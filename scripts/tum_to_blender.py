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

import json
import numpy as np
from pathlib import Path
from PIL import Image

import vbgs
from vbgs.data.habitat import HabitatDataIterator, load_camera_params

from optisplat.data import load_3D_TUM_new, parse_tum_pose, FR3_INTRINSICS

from optisplat.geom_util import rot_x
import jax.numpy as jnp


def read_json(f):
    with open(f, "r") as fp:
        d = json.load(fp)
    return d


def parse_tum(iterator):
    for datum in iterator:
        pose = np.eye(4)
        if len(datum) == 3:
            pose = parse_tum_pose(datum[2])
            pose = pose @ rot_x(np.pi)
        rgb_path = datum[0]
        depth_path = datum[1]
        yield rgb_path, depth_path, FR3_INTRINSICS, pose


def build_loader(datasetpath):
    return parse_tum(load_3D_TUM_new(datasetpath, use_pose=True))


if __name__ == "__main__":
    root_path = Path(vbgs.__file__).parent.parent

    datapath = Path(
        "/home/shared/rgbd_dataset_freiburg3_long_office_household"
    )
    out_path = Path(
        "/home/shared/rgbd_dataset_freiburg3_long_office_household/blender"
    )
    out_path.mkdir(exist_ok=True, parents=True)

    data_iter = build_loader(datapath)

    frames = []
    frames_val = []

    x = list(data_iter)
    train_indices = np.arange(len(x))[::10]
    val_indices = np.arange(len(x))[5:][::10]
    print(len(train_indices), len(val_indices))

    for i, (frame, depth_frame, intrinsics, extrinsics) in enumerate(x):
        if (i not in val_indices) and (i not in train_indices):
            continue

        image = Image.open(frame)

        fx = intrinsics[0][0]
        fy = intrinsics[1][1]

        fov_x = 2 * np.arctan2(intrinsics[0][2], fx)
        fov_y = 2 * np.arctan2(intrinsics[1][2], fy)

        head = {
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "width": 2 * intrinsics[0][2],
            "height": 2 * intrinsics[1][2],
            "rotation": None,
        }

        f = {
            "depth_path": str(depth_frame),
            "file_path": str(frame),
            "transform_matrix": extrinsics.tolist(),
        }
        f.update(head)

        if i in train_indices:
            frames.append(f)
        elif i in val_indices:
            frames_val.append(f)

    with open(out_path / "transforms_train.json", "w") as fp:
        f_ = {"frames": frames}
        f_.update(head)
        print(f_.keys())
        json.dump(f_, fp, indent=2)

    # we changed the script to fit the test data for blender so now we have to use this naming
    with open(out_path / "transforms_eval.json", "w") as fp:
        f_val = {"frames": frames_val}
        f_val.update(head)
        json.dump(f_val, fp, indent=2)
