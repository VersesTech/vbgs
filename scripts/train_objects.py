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

import numpy as np
import hydra

import json
import copy
from functools import partial
from typing import Optional, Tuple

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import jax
from jax import jit
import jax.random as jr
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
from PIL import Image

import vbgs
from vbgs.model.utils import random_mean_init, store_model
from vbgs.model.train import fit_gmm_step
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.data.blender import BlenderDataIterator
from vbgs.finetune import finetune
from vbgs.render.volume import render_gsplat


from model_volume import get_volume_delta_mixture


def fit_continual(
    data_path,
    n_components,
    subsample=None,
    init_random=False,
    key=None,
    batch_size=5000,
):
    if key is None:
        key = jr.PRNGKey(0)

    data_iter = BlenderDataIterator(data_path, data_params=None, subsample=subsample)
    if not init_random:
        # Essentially, if not init random, we load n_components points from the
        # point cloud, to initialize the model components on. Then we can just
        # do either the continual or non continual learning scheme with this
        # script. Note that in a real continual setting, init on data won't be
        # possible, hence we need a proper create_normalizing params
        data = np.zeros((0, 6))
        for d in data_iter:
            data = np.concatenate([data, d])
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
        data_path, data_params=data_params, subsample=subsample
    )

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
    data_iter._frames = data_iter._frames[::40]
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        model, prior_stats, space_stats, color_stats = fit_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
        )

        store_model(model, data_params, f"model_{step:02d}.json")

    return model, metrics, data_params


def run_experiment(
    key,
    data_path,
    n_components,
    subsample,
    init_random,
    batch_size: int,
    do_finetune: Optional[Tuple] = None,
):
    # Fit continual VBEM
    key, subkey = jr.split(key)
    model, _, data_params = fit_continual(
        data_path,
        n_components,
        subsample=subsample,
        key=subkey,
        init_random=init_random,
        batch_size=batch_size,
    )
    if do_finetune is not None:
        params = model.extract_model(data_params)
        iterator_fn = lambda k: BlenderDataIterator(
            data_path, data_params=data_params, subsample=subsample, key=k
        )
        finetune(
            *params,
            iterator_fn,
            do_finetune[0],
            key,
            bs=do_finetune[1],
            log_every=do_finetune[2],
        )
    return {}


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="blender",
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
        do_finetune=(
            cfg.train.finetune
            if cfg.train.finetune is None
            else (
                cfg.train.finetune_iters,
                cfg.train.finetune_batch_size,
                cfg.train.finetune_log,
            )
        ),
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
