import hydra
from omegaconf import DictConfig, OmegaConf

import json
import copy
import rich

from pathlib import Path
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import vbgs
from vbgs.model.utils import random_mean_init
from vbgs.data.utils import create_normalizing_params, normalize_data
from optisplat.delta_mixture import fit_delta_gmm_step

from optisplat.delta_mixture import get_volume_delta_mixture

from optisplat.data import (
    BlenderDataIterator,
    load_blender_data,
    load_blender_val,
)

from optisplat.volume_rendering import (
    get_projected_mixture,
    patched_render_with_occlusion,
    create_model_mesh,
)

from vbgs.camera import construct_world_to_image_matrix
from vbgs.metrics import calc_mse, calc_psnr

from PIL import Image


def evaluate(mu, si, alpha, imgs, poses, intrinsics, step):
    print("Running evaluation...")
    results = {"mse": [], "psnr": []}
    for idx in tqdm(np.arange(0, 200, 10)):
        ground_truth, c2w = imgs[idx], poses[idx]

        w2i = np.array(
            construct_world_to_image_matrix(
                jnp.array(c2w), jnp.array(intrinsics), True
            )
        )

        nm, nc = get_projected_mixture(w2i, np.array(mu), np.array(si))

        rendered_img = patched_render_with_occlusion(
            nm, nc, jnp.array(alpha), ground_truth.shape[:2], patch_size=100
        )

        results["mse"].append(float(calc_mse(rendered_img, ground_truth)))
        results["psnr"].append(float(calc_psnr(rendered_img, ground_truth)))

        step_dir = Path(f"step_{step:02d}")
        step_dir.mkdir(exist_ok=True, parents=True)
        Image.fromarray((255 * rendered_img).astype(np.uint8)).save(
            step_dir / f"render_{idx:02d}.png"
        )

    return results


def fit_continual(
    data_path,
    n_components,
    subsample=None,
    init_random=False,
    key=None,
    eval_every=None,
    batch_size=5000,
):
    # Load the imgs & poses & intrinsics for evaluation purposes
    imgs, poses, intrinsics = load_blender_val(data_path, "test")

    if key is None:
        key = jr.PRNGKey(0)

    if not init_random:
        # Essentially, if not init random, we load n_components points from the
        # point cloud, to initialize the model components on. Then we can just
        # do either the continual or non continual learning scheme with this
        # script. Note that in a real continual setting, init on data won't be
        # possible, hence we need a proper create_normalizing params
        data, image_shape = load_blender_data(data_path)
        data, data_params = normalize_data(data)

        np.random.seed(0)
        idcs = np.arange(data.shape[0])
        np.random.shuffle(idcs)
        x_data = data[idcs[:n_components]]
        del data

    else:
        x_data = None
        data_params = create_normalizing_params(
            [-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]
        )

    data_iter = BlenderDataIterator(
        data_path, data_params, subsample=subsample
    )

    # TODO: remove this: just a way to reduce processed data.
    print(len(data_iter), "Frames to be processed.")

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=x_data,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=init_random,
        add_noise=False,
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

    metrics = dict({})
    prior_stats, space_stats, color_stats = None, None, None
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        model, prior_stats, space_stats, color_stats = fit_delta_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
        )

        mu, si = model.denormalize(data_params, clip_val=None)
        alpha = model.prior.alpha.reshape(-1)

        # Storing the model as a json at each step!
        model_dict = {
            "mu": mu.tolist(),
            "si": si.tolist(),
            "alpha": alpha.tolist(),
        }
        with open(f"model_{step:02d}.json", "w") as f:
            json.dump(model_dict, f, indent=2)

        # Run evaluation at these intervals
        if eval_every is not None and step % eval_every == 0:
            # Computing the PSNR's and MSE's of the renders
            res = evaluate(mu, si, alpha, imgs, poses, intrinsics, step=step)
            n_used = int(
                (model.prior.alpha > model.prior.prior_alpha.min()).sum()
            )
            res.update({"n_used": n_used})
            for k, v in res.items():
                metrics[k] = metrics.get(k, []) + [v]

            # Creating the model mesh
            used_components = alpha > alpha.min()
            mu = mu[used_components]
            si = si[used_components]
            create_model_mesh(mu, si * np.eye(6).reshape(1, 6, 6) * 3).export(
                f"mesh_{step:02d}.obj", "obj"
            )

    return metrics


def run_experiment(
    key, data_path, n_components, subsample, init_random, batch_size
):
    # Fit continual VBEM
    key, subkey = jr.split(key)
    metrics = fit_continual(
        data_path,
        n_components,
        subsample=subsample,
        key=subkey,
        init_random=init_random,
        batch_size=batch_size,
    )
    rich.print(metrics)

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="benchmark_blender",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    jax.config.update("jax_default_device", jax.devices()[int(cfg.device)])

    root_path = Path(vbgs.__file__).parent.parent

    results = run_experiment(
        key=jr.PRNGKey(0),
        n_components=cfg.model.n_components,
        data_path=root_path / Path(cfg.data.data_path),
        subsample=cfg.data.subsample_factor,
        init_random=cfg.model.init_random,
        batch_size=cfg.train.batch_size,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
