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

rot = np.eye(4)
rot[1, 1] = -1
rot[2, 2] = -1


def load_camera_params(im_path):
    with open(str(im_path).replace(".jpeg", "_camera_params.json"), "r") as f:
        d = json.load(f)
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = np.array(d["camera_intrinsics"])

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.array(d["R_cam2world"])
    extrinsics[:3, -1] = d["t_cam2world"]

    return intrinsics, extrinsics @ rot


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
    def __init__(self, path, idx, data_params):
        path = Path(path)
        if idx == "":
            self._frames = sorted(list(path.glob("*.jpeg")))
        else:
            self._frames = sorted(list(path.glob(f"{idx}_*.jpeg")))
        self._data_params = data_params

        self._index = 0

    def _get_frame(self, index):
        im = self._frames[index]

        rgb = cv2.imread(im)[..., [2, 1, 0]]
        d = str(im).replace(".jpeg", "_depth.exr")
        d = cv2.imread(d, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        intrinsics, extrinsics = load_camera_params(im)

        points, rgb = transform_uvd_to_points(
            rgb,
            d,
            extrinsics,
            intrinsics,
            from_opengl=True,
            filter_zero=False,  # True
        )

        data = jnp.concatenate([points, rgb], axis=1)

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
