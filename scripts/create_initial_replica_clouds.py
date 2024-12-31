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

# from vbgs.data.habitat import HabitatDataIterator
from vbgs.data.replica import ReplicaDataIterator
from vbgs.model.reassign import reassign


from model_volume import get_volume_delta_mixture

from vbgs.render.volume import (
    readCamerasFromTransforms,
    render_img,
    vbgs_model_to_splat,
)

from PIL import Image
from vbgs.metrics import calc_psnr, calc_mse
import matplotlib.pyplot as plt

from plyfile import PlyData, PlyElement


def store_ply(path, xyz, rgb):
    # Make sure rgb is an uint8 (we might push in floats).
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = (rgb * 255).astype("u1")

    # Define the dtype for the structured array.
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements = np.array(list(map(tuple, attributes)), dtype=dtype)

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def evaluate(model, cameras, store_path):
    psnrs = []
    mses = []
    for i in range(len(cameras)):
        x = np.array(Image.open(cameras[i].image_path)) / 255.0
        x_hat = render_img(model, cameras, i, 0, scale=1.4)

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
    device=0,
):
    # the torch stuff should always be on cuda:0
    torch_device = "cuda:0"
    np.random.seed(seed)
    random.seed(seed)

    if key is None:
        key = jr.PRNGKey(0)

    reassign_fn = reassign if use_reassign else no_reassign

    # ====
    eval_cameras = readCamerasFromTransforms(
        Path(
            str(data_path)
            .replace(
                "Replica", "Replica-blender"
            )  # Go to the location of the blender transforms file
            .replace(
                "-depth_estimated", ""
            )  # The depth estimated frames use the same validation set
        ),
        "transforms_eval.json",
        True,
    )[::2]  # Evaluate on 100 frames

    # ============
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
    data_iter.indices = np.arange(0, 2000, 10)

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=x_data,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=init_random,
        add_noise=True,
    )

    s = data_params["stdevs"]
    o = data_params["offset"]
    xi = mean_init[..., 0] * s + o

    scene = data_path.name
    print(scene)
    p = Path(f"/home/shared/Replica-blender/{scene}") / "100K_initial_pts.ply"
    store_ply(p, xi[:, :3], (xi[:, 3:] * 255).astype(np.uint8))


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
        device=device,
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
