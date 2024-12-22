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

from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

import vbgs
import jax.numpy as jnp
from vbgs.model.utils import load_model
from vbgs.data.blender import BlenderDataIterator
from vbgs.data.replica import ReplicaDataIterator
from vbgs.render.volume import (
    # readCamerasFromTransforms,
    render_gsplat,
)

from vbgs.data.habitat import HabitatDataIterator


def show_replica():
    root_path = Path(vbgs.__file__).parent.parent
    data_path = Path("/home/shared/Replica/room0")
    model_path = Path("/home/shared/vbgs-results/room0_trained.npz")
    mu, si, alpha = load_model(model_path)

    data_iter = ReplicaDataIterator(data_path)
    p0 = data_iter.poses[:200]

    render_gsplat(mu, si, alpha, p0, data_iter.intrinsics, 640, 1200)


def show_blender():
    root_path = Path(vbgs.__file__).parent.parent
    # blender_data_path = root_path / "../../data/blender/lego"
    blender_data_path = (root_path / "../data/blender/lego").resolve()
    # load the data in our format
    data_iter = BlenderDataIterator(blender_data_path, "transforms_val.json")
    rich.print(data_iter._frames[0])

    # Load the cameras in the gaussian-splatting format
    # cameras = readCamerasFromTransforms(
    #     blender_data_path, "transforms_val.json", True
    # )
    with open(blender_data_path / "transforms_val.json") as f:
        transforms_val = json.load(f)
        cameras = jnp.array([x["transform_matrix"] for x in transforms_val["frames"]])
    # Load the trained model
    splat_path = (
        "data/blender-dataset/lego/nc:10000/subs:None_randinit:True/model_12.json"
    )
    # model = vbgs_model_to_splat(root_path / splat_path)
    mu, si, alpha = load_model(root_path / splat_path)
    x_hat = render_gsplat(
        mu,
        si,
        alpha,
        cameras[0],
        data_iter._intrinsics,
        480,
        640,
    )
    i = 0
    # x_hat = render_img(model, cameras, i, 1)
    x = Image.open(str(blender_data_path / data_iter._frames[i]["file_path"]) + ".png")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(x)
    ax[0].set_title("Ground truth")

    ax[1].imshow(x_hat)
    ax[1].set_title("Predicted")

    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]
    # plt.show()
    plt.savefig("output.png")


def show_habitat():
    root_path = Path(vbgs.__file__).parent.parent

    habitat_data_path = (
        "/home/shared/habitat_processed/habitat-test-scenes/van-gogh-room/"
    )

    # load the data in our format
    data_iter = HabitatDataIterator(habitat_data_path, "", None)
    rich.print(data_iter._frames[0])

    # Load the cameras in the gaussian-splatting format
    cameras = readCamerasFromTransforms(
        root_path / "resources/large-datasets/van-gogh-room",
        "transforms_eval_200.json",
        True,
    )

    # Load the trained model
    splat_path = "data/rooms/van-gogh-room_shuffle:True/nc:100000/randinit:True_reassign:True/model_199.json"
    # model = vbgs_model_to_splat(root_path / splat_path)
    mu, si, alpha = load_model(root_path / splat_path)

    i = 0
    # x_hat = render_img(model, cameras, i, 1, scale=2)
    x_hat = render_gsplat(
        mu,
        si,
        alpha,
    )
    x = Image.open(data_iter._frames[i])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(x)
    ax[0].set_title("Ground truth")

    ax[1].imshow(x_hat)
    ax[1].set_title("Predicted")

    [a.set_xticks([]) for a in ax.flatten()]
    [a.set_yticks([]) for a in ax.flatten()]
    plt.show()


if __name__ == "__main__":
    show_blender()
    # show_habitat()
