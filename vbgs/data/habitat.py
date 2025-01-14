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

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import json

from vbgs.camera import transform_uvd_to_points
from vbgs.data.utils import normalize_data

from vbgs.data.depth import load_depth_model, predict_depth

rot = np.eye(4)
rot[1, 1] = -1
rot[2, 2] = -1


def load_camera_params(im_path):
    with open(str(im_path).replace(".jpeg", "_camera_params.json"), "r") as f:
        d = json.load(f)

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = np.array(d["camera_intrinsics"])[:3, :3]

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.array(d["R_cam2world"])
    extrinsics[:3, -1] = d["t_cam2world"]

    if "habitat" in str(im_path):
        extrinsics = extrinsics @ rot

    depth_scale = d.get("scale", 1)

    return intrinsics, extrinsics, depth_scale


class TUMDataIterator:
    def __init__(self, path, data_params=None):
        path = Path(path)
        self._frames = sorted(list(path.glob("*.npz")))
        self._data_params = data_params
        self._index = 0

    def _get_frame(self, idx):
        cloud_path = self._frames[idx]
        data = np.load(cloud_path)["arr_0"]

        if self._data_params is not None:
            data, _ = normalize_data(data, self._data_params)
        return data

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


class HabitatDataIterator:
    def __init__(
        self,
        path,
        idx,
        data_params,
        estimate_depth=False,
        device="cuda:0",
        from_opengl=True,
    ):
        path = Path(path)
        if idx == "":
            frames = list(path.glob("*.jpeg"))
            frames = [f for f in frames if "depth" not in str(f)]
            self._frames = sorted(
                frames,
                key=lambda x: int(str(x).replace(".jpeg", "").split("/")[-1]),
            )
        else:
            self._frames = sorted(list(path.glob(f"{idx}_*.jpeg")))
        self._data_params = data_params

        self._index = 0

        self._estimate_depth = estimate_depth
        if self._estimate_depth:
            self._depth_model = load_depth_model("dav2", device)

        self._from_opengl = from_opengl
        self.intrinsics, *_ = load_camera_params(self._frames[0])
        self.c = int(self.intrinsics[0, 2]), int(self.intrinsics[1, 2])
        self.f = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])

    def _get_frame_rgbd(self, index):
        im = self._frames[index]

        rgb = cv2.imread(im)[..., [2, 1, 0]]

        intrinsics, extrinsics, depth_scale = load_camera_params(im)

        if self._estimate_depth:
            d = predict_depth(rgb, *self._depth_model)
        else:
            depth_path = str(im).replace(".jpeg", "_depth.exr")
            if os.path.exists(depth_path):
                d = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            else:
                d = str(im).replace(".jpeg", "_depth.jpeg")
                d = cv2.imread(d, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                d = d / 255.0

            d = depth_scale * d

        return rgb, d, intrinsics, extrinsics

    def _get_frame(self, index):
        rgb, d, intrinsics, extrinsics = self._get_frame_rgbd(index)

        points, rgb = transform_uvd_to_points(
            rgb,
            d,
            extrinsics,
            intrinsics,
            from_opengl=self._from_opengl,
            filter_zero=False,
        )

        data = jnp.concatenate([points, rgb], axis=1)

        if self._data_params is not None:
            data, _ = normalize_data(data, self._data_params)

        return data

    def get_camera_params(self, index):
        im = self._frames[index]
        return load_camera_params(im)

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
