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

        shape = jnp.array(
            Image.open(data_path / f"{data['frames'][0]['file_path']}.png")
        ).shape

        # For the blender dataset fx = fy
        angle_x = data["camera_angle_x"]
        fx = shape[0] / (2 * jnp.tan(angle_x / 2))
        fy = fx
        intrinsics = jnp.eye(4)
        intrinsics = intrinsics.at[0, 0].set(fx)
        intrinsics = intrinsics.at[1, 1].set(fy)
        intrinsics = intrinsics.at[:2, 2].set(shape[0] / 2)

        self._intrinsics = intrinsics
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
            + [jnp.ones(shape=(*shape[:2], 1))],
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

    def _compute_cloud(self, i):
        frame = self._frames[i]
        color_path = f"{frame['file_path']}.png"
        depth_path = f"{frame['file_path']}_depth_*.png"

        depth_path = list(self._data_path.glob(depth_path))[0]

        color = jnp.array(Image.open(self._data_path / color_path))

        depth_im = jnp.array(Image.open(depth_path))
        depth = 8 * (1.0 - (depth_im[..., 0] / 255.0))
        depth *= self._r
        depth *= depth_im[..., 0] > 0

        camera_to_world = np.array(frame["transform_matrix"])

        points, rgb = transform_uvd_to_points(
            color[..., :3],
            depth,
            camera_to_world,
            self._intrinsics,
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
