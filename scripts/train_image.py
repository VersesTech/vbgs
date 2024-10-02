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

import rich
import json
import copy
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import jax.random as jr
import jax.numpy as jnp

from pathlib import Path

from tqdm import tqdm

import datasets as ds

import vbgs

from vbgs.model.utils import random_mean_init
from vbgs.data.image import image_to_data
from vbgs.data.utils import normalize_data
from vbgs.model.train import fit_gmm
from vbgs.metrics import calc_mse, calc_psnr
from vbgs.render.image import render_img

from model_image import get_image_model


resources_path = Path(vbgs.__file__).parent.parent / "resources"


def run_experiment(
    n_components,
    init_random,
    beta=0,
    learning_rate=1,
    batch_size=1,
    dof=1,
    n_iters=1,
    scale=None,
    seed=123,
):
    def fit(key, img):
        data = image_to_data(img)
        x, data_params = normalize_data(data)

        key, subkey = jr.split(key)
        mean_init = random_mean_init(
            subkey,
            x,
            component_shape=(n_components,),
            event_shape=(5, 1),
            init_random=init_random,
            add_noise=False,
        )

        model = get_image_model(
            key,
            n_components=n_components,
            mean_init=mean_init,
            beta=beta,
            learning_rate=learning_rate,
            dof_offset=dof,
            position_scale=scale,
        )

        initial_model = copy.deepcopy(model)
        for i in range(n_iters):
            model = fit_gmm(initial_model, model, x)

        mu, si = model.denormalize(data_params)

        rendered_img = render_img(mu, si, model.prior.alpha, img.shape[:2])

        mse_ = calc_mse(img.astype(np.float32), rendered_img.clip(0, 1.0))
        psnr = calc_psnr(img.astype(np.float32), rendered_img.clip(0, 1.0))

        n_used = int((model.prior.alpha > model.prior.prior_alpha.min()).sum())
        metrics = {"mse": mse_.item(), "psnr": psnr.item(), "n_used": n_used}
        return key, metrics

    key = jr.PRNGKey(seed)

    # Evaluate on validation set (10k images)
    dataset = ds.load_dataset("Maysee/tiny-imagenet", split="valid")["image"]

    # Track some metrics
    metrics = {}
    for image in tqdm(dataset):
        img = jnp.array(image) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        key, subkey = jr.split(key)
        subkey, m = fit(subkey, img)

        metrics = {k: metrics.get(k, []) + [v] for k, v in m.items()}

    rich.print(metrics)

    with open("results.json", "w") as f:
        json.dump(metrics, f)

    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="imagenet")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    results = run_experiment(
        n_components=cfg.model.n_components,
        n_iters=cfg.train.n_iters,
        learning_rate=cfg.train.learning_rate,
        beta=cfg.train.beta,
        init_random=cfg.model.init_random,
        seed=cfg.seed,
        batch_size=cfg.train.batch_size,
        dof=cfg.model.dof,
        scale=cfg.model.scale,
    )

    results.update({"config": OmegaConf.to_container(cfg)})
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
