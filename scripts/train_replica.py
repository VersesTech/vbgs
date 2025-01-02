# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig, OmegaConf

import json
import copy
import rich

from pathlib import Path
from tqdm import tqdm

import numpy as np
import random

import jax
import jax.numpy as jnp
import jax.random as jr

import vbgs
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.model.utils import random_mean_init, store_model
from vbgs.model.train import fit_gmm_step

from vbgs.data.replica import ReplicaDataIterator
from vbgs.model.reassign import reassign


from model_volume import get_volume_delta_mixture

from vbgs.render.volume import render_gsplat

from PIL import Image
from vbgs.metrics import calc_psnr, calc_mse
import matplotlib.pyplot as plt

def evaluate(splat, cameras, eval_frames, intrinsics, shape, store_path):
    psnrs = []
    mses = []
    for i in range(len(cameras)):
        x = np.array(Image.open(eval_frames[i])) / 255.0
        x_hat = render_gsplat(*splat, cameras[i], intrinsics, *shape)

        psnrs.append(calc_psnr(x, x_hat))
        mses.append(calc_mse(x, x_hat))

    np.savez(store_path, psnr=np.array(psnrs), mse=np.array(mses))
    return np.array(psnrs), np.array(mses)


def no_reassign(*x):
    return x[0]


def fit_continual(
    data_path,
    n_components,
    init_random=False,
    key=None,
    eval_every=1,
    batch_size=5000,
    use_reassign=True,
    reassign_fraction=0.05,
    seed=0,
):
    np.random.seed(seed)
    random.seed(seed)

    if key is None:
        key = jr.PRNGKey(0)

    reassign_fn = reassign if use_reassign else no_reassign

    # Some subsampling
    x_data = None
    data_params = create_normalizing_params(
        [-5, 5], [-5, 5], [-5, 5], [0, 1], [0, 1], [0, 1]
    )
    if not init_random:
        data_iter = ReplicaDataIterator(data_path, None)
        data_iter.indices = np.arange(0, 2000, 10)

        all_data = []
        for frame in data_iter:
            all_data.append(frame)

        x_data = jnp.concatenate(all_data, axis=0)

        x_data, data_params = normalize_data(x_data, None)
        print("Normalizing data parameters: ")
        print(data_params, end="\n\n")

    data_iter = ReplicaDataIterator(data_path, data_params)
    eval_cameras = [data_iter.load_camera_params(i)[1] for i in jnp.arange(0, len(data_iter), 2)]
    eval_frames = [data_iter.get_frame(i)[0] for i in jnp.arange(0, len(data_iter), 2)]
    # data_iter.indices = np.arange(0, 2000, 10)
    data_iter.indices = np.arange(0, 2000, 400)

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=x_data,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=init_random,
        add_noise=True,
    )

    del x_data

    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=n_components,
        mean_init=mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=n_components,
        position_event_shape=(3, 1),
    )

    model = copy.deepcopy(prior_model)

    metrics = dict(
        {"psnr": {"mean": [], "std": []}, "mse": {"mean": [], "std": []}}
    )
    prior_stats, space_stats, color_stats = None, None, None
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        prior_model = reassign_fn(
            prior_model, model, x, batch_size, reassign_fraction
        )
        model, prior_stats, space_stats, color_stats = fit_gmm_step(
            prior_model,
            model,
            data=x[::4],
            batch_size=batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
        )

        if step % eval_every == 0:
            store_model(
                model, data_params, f"model_{step:02d}.npz", use_numpy=True
            )            
            p, m = evaluate(
                model.extract_model(data_params),
                eval_cameras,
                eval_frames,
                data_iter.intrinsics,
                (data_iter.h, data_iter.w),
                f"results_{step:02d}.npz",
            )


            metrics["psnr"]["mean"].append(p.mean())
            metrics["psnr"]["std"].append(p.std())
            metrics["mse"]["mean"].append(m.mean())
            metrics["mse"]["std"].append(m.std())

            if step % 10 == 0:
                print(f"PSNR: {p.mean():.2f} +- {p.std():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, (k, v) in enumerate(metrics.items()):
        y, std = np.array(metrics[k]["mean"]), np.array(metrics[k]["std"])
        ax[i].plot(y, label=k)
        ax[i].fill_between(
            np.arange(len(y)), y - 1.96 * std, y + 1.96 * std, alpha=0.25
        )
    plt.legend()
    plt.savefig("metrics.png")

    # Make sure the final model is stored as well
    store_model(model, data_params, f"model_{step:02d}.json")
    return metrics


def run_experiment(
    key,
    data_path,
    n_components,
    init_random,
    batch_size,
    use_reassign,
    reassign_fraction,
    device,
):
    # Fit continual VBEM
    key, subkey = jr.split(key)
    metrics = fit_continual(
        data_path,
        n_components,
        key=subkey,
        init_random=init_random,
        batch_size=batch_size,
        use_reassign=use_reassign,
        reassign_fraction=reassign_fraction,
    )
    rich.print(metrics)

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="replica",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    jax.config.update("jax_default_device", jax.devices()[int(cfg.device)])

    root_path = Path(vbgs.__file__).parent.parent

    # Minor hack to launch everything at once
    data_path = cfg.data.data_path
    if "room0_depth_estimate" in data_path:
        data_path = data_path.replace("_depth_estimated", "").replace(
            "Replica", "Replica-depth_estimated"
        )

    results = run_experiment(
        key=jr.PRNGKey(0),
        n_components=cfg.model.n_components,
        data_path=root_path / Path(data_path),
        init_random=cfg.model.init_random,
        batch_size=cfg.train.batch_size,
        use_reassign=cfg.model.use_reassign,
        reassign_fraction=float(cfg.model.reassign_fraction),
        device=cfg.device,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
