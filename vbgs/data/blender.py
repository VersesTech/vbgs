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

import json
from pathlib import Path
from PIL import Image

import jax
import jax.numpy as jnp
import jax.random as jr


from functools import partial

from vbgs.camera import transform_uvd_to_points
from vbgs.data.utils import normalize_data


class BlenderDataIterator:
    def __init__(
        self,
        data_path,
        file="transforms_test.json",
        data_params=None,
        subsample=None,
    ):
        self._data_params = data_params

        self._subsample = subsample

        self._data_path = Path(data_path)
        with open(data_path / file) as f:
            data = json.load(f)

        color_path = data["frames"][0]["file_path"]
        if ".png" not in color_path:
            color_path += ".png"

        shape = jnp.array(Image.open(data_path / color_path)).shape
        self.h, self.w = shape[:2]

        # For the blender dataset fx = fy
        angle_x = data["camera_angle_x"]
        fx = shape[1] / (2 * jnp.tan(angle_x / 2))
        fy = fx
        intrinsics = jnp.eye(4)
        intrinsics = intrinsics.at[0, 0].set(fx)
        intrinsics = intrinsics.at[1, 1].set(fy)
        intrinsics = intrinsics.at[0, 2].set(shape[1] / 2 - 0.5)
        intrinsics = intrinsics.at[1, 2].set(shape[0] / 2 - 0.5)

        self.intrinsics = intrinsics
        self.c = int(intrinsics[0, 2]), int(intrinsics[1, 2])
        self.f = float(intrinsics[0, 0]), float(intrinsics[1, 1])

        self._frames = data["frames"]
        self._index = 0
        self._r = self._compute_distance_to_depth(angle_x, shape)

        self.key = jr.PRNGKey(0)

    @staticmethod
    def _compute_distance_to_depth(angle_x, shape):
        # Map from a distance map to a depth map!
        uv = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]))
        uv = jnp.concatenate(
            [jnp.expand_dims(u, -1) for u in uv]
            + [jnp.ones(shape=(shape[1], shape[0], 1))],
            axis=-1,
        )
        uv = uv - shape[0] / 2
        uv = uv.at[..., 0].set(uv[..., 0] * angle_x / 2)
        uv = uv.at[..., 1].set(uv[..., 1] * angle_x / 2)
        uvr = uv.reshape(-1, 3)
        uvr = uvr / jnp.linalg.norm(uvr, axis=-1, keepdims=True)
        fwd = jnp.array([0, 0, -1])
        fwd = fwd / jnp.linalg.norm(fwd)
        r = jax.vmap(partial(jnp.dot, fwd))(uvr)
        r = r.reshape(uv.shape[:2])
        return r

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._frames):
            item = self._get_frame(self._index)
            self._index += 1
            return item
        else:
            raise StopIteration

    def load_camera_params(self, idx):
        frame = self._frames[idx]
        cam2world = jnp.array(frame["transform_matrix"])
        return cam2world, self.intrinsics

    def get_camera_frame(self, idx):
        frame = self._frames[idx]
        color_path = frame["file_path"]
        if ".png" not in color_path:
            color_path += ".png"
        color_image = jnp.array(Image.open(self._data_path / color_path)) * 1.0

        depth_path = frame.get("depth_path", None)
        if depth_path is None:
            depth_path = f"{frame['file_path']}_depth_*.png"
            depth_path = list(self._data_path.glob(depth_path))[0]
            # Depth image processing specific to blender dataset
            depth_im = jnp.array(Image.open(depth_path))
            depth_image = 8 * (1.0 - (depth_im[..., 0] / 255.0))
            depth_image *= self._r
            depth_image *= depth_im[..., 0] > 0
        else:
            depth_image = jnp.array(Image.open(depth_path)) / 5000

        return color_image, depth_image

    def _compute_cloud(self, i):
        frame = self._frames[i]
        color, depth = self.get_camera_frame(i)

        camera_to_world = np.array(frame["transform_matrix"])

        points, rgb = transform_uvd_to_points(
            color[..., :3],
            depth,
            camera_to_world,
            self.intrinsics,
            from_opengl=True,
            filter_zero=True,
        )

        data = jnp.concatenate([points, rgb], axis=1)
        return data

    def _get_frame(self, i):
        frame = self._frames[i]
        cloud_path = self._data_path / f"{frame['file_path']}.npz"
        if not cloud_path.exists():
            data = self._compute_cloud(i)
            np.savez(cloud_path, data)
        else:
            data = np.load(cloud_path)["arr_0"]

        if self._data_params is not None:
            data, _ = normalize_data(data, self._data_params)

        if self._subsample is not None:
            self.key, subkey = jr.split(self.key)
            data = jr.permutation(subkey, data, independent=False)
            data = data[: self._subsample]

        return np.array(data)
