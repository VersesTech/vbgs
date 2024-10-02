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

import jax.numpy as jnp
import jax.random as jr

import numpy as np

import copy
import rich

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import json

import datasets

from vbgs.model.utils import random_mean_init
from vbgs.model.train import fit_delta_gmm_step
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.data.image import to_patches, image_to_data
from vbgs.render.image import render_img
from vbgs.metrics import calc_mse, calc_psnr

from model_image import get_image_model


def fit_continual(image_patches, img, n_components, key=None):
    if key is None:
        key = jr.PRNGKey(0)

    data_params = create_normalizing_params(
        [0, 64], [0, 64], [0, 1], [0, 1], [0, 1]
    )

    event_shape = (5, 1)

    component_shape = (n_components,)
    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        subkey,
        None,
        component_shape,
        event_shape,
        init_random=True,
        add_noise=False,
    )

    prior_model = get_image_model(
        key,
        component_shape[0],
        mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=15 * jnp.sqrt(component_shape[0]),
    )

    mu, si = prior_model.denormalize(data_params)
    rendered_img = render_img(mu, si, prior_model.prior.alpha, img.shape[:2])

    info = {"psnr": [], "mse": []}

    model = copy.deepcopy(prior_model)
    prior_stats, likelihood_stats, delta_stats = None, None, None
    for patch in image_patches:
        x, _ = normalize_data(patch, data_params)
        model, pi_stats, l_stats, d_stats = fit_delta_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=x.shape[0],
            space_stats=likelihood_stats,
            prior_stats=prior_stats,
            color_stats=delta_stats,
        )

        prior_stats = pi_stats
        likelihood_stats = l_stats
        delta_stats = d_stats

        mu, si = model.denormalize(data_params)
        rendered_img = render_img(mu, si, model.prior.alpha, img.shape[:2])

        info["mse"].append(calc_mse(rendered_img, img))
        info["psnr"].append(calc_psnr(rendered_img, img))

    return info


def run_experiment(n_components, key):
    # Evaluate on validation set (10k images)
    dataset = datasets.load_dataset("Maysee/tiny-imagenet", split="valid")

    all_metrics = {}
    for image in tqdm(dataset["image"]):
        img = np.array(image) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        data = image_to_data(img)
        img_patches, masks = to_patches(data, img)

        # Fit continual VBEM
        key, subkey = jr.split(key)
        metrics_vbem = fit_continual(img_patches, img, n_components, subkey)
        metrics_c_vbem = {f"cont_vbem_{k}": v for k, v in metrics_vbem.items()}

        # Fit 1 pass VBEM
        key, subkey = jr.split(key)
        metrics_vbem = fit_continual([data], img, n_components, subkey)
        metrics_b_vbem = {f"base_vbem_{k}": v for k, v in metrics_vbem.items()}

        # aggregate the metrics we care about
        metrics = dict({})
        metrics.update(metrics_c_vbem)
        metrics.update(metrics_b_vbem)
        all_metrics = {
            k: all_metrics.get(k, []) + [v] for k, v in metrics.items()
        }

    rich.print(all_metrics)

    with open("results.json", "w") as f:
        json.dump(all_metrics, f)

    return all_metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="imagenet",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    results = run_experiment(
        n_components=cfg.model.n_components, key=jr.PRNGKey(0)
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
