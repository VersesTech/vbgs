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
from vbgs.data.replica import ReplicaDataIterator
import jax.numpy as jnp
from vbgs.camera import opengl_to_frame


def read_json(f):
    with open(f, "r") as fp:
        d = json.load(fp)
    return d


if __name__ == "__main__":
    root_path = Path(vbgs.__file__).parent.parent

    test_scene_path = Path("/home/shared/Replica/")
    scenes = [i.name for i in test_scene_path.glob("*")]
    scenes = [i for i in scenes if "raw" not in str(i)]

    for s in scenes:
        data_path = test_scene_path / s
        print(data_path)

        indices = np.arange(0, 200, 10).tolist()
        val_indices = np.arange(5, 205, 10).tolist()

        data_iter = ReplicaDataIterator(data_path)

        frames = []
        frames_val = []
        for i, pc in enumerate(data_iter):
            frame, _ = data_iter.get_frame(i)
            print(frame)
            intrinsics, extrinsics  = data_iter.load_camera_params(i)

            extrinsics = jnp.dot(extrinsics, opengl_to_frame)

            image = Image.open(frame)

            fx = intrinsics[0][0]
            fy = intrinsics[1][1]

            fov_x = 2 * np.arctan2(intrinsics[0][2], fx)
            fov_y = 2 * np.arctan2(intrinsics[1][2], fy)

            f = {
                "camera_angle_x": fov_x,
                "camera_angle_y": fov_y,
                "width": 2 * intrinsics[0][2],
                "height": 2 * intrinsics[1][2],
                "file_path": str(frame),
                "rotation": None,
                "transform_matrix": extrinsics.tolist(),
            }

            if i in indices:
                frames.append(f)
            if i in val_indices:
                frames_val.append(f)

        out_path = root_path / f"resources/large-datasets/{s}"
        out_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / "transforms_train.json", "w") as fp:
            json.dump(
                {
                    "frames": frames,
                    "camera_angle_x": frames_val[0]["camera_angle_x"],
                    "camera_angle_y": frames_val[0]["camera_angle_y"],
                    "width": 2 * intrinsics[0][2],
                    "height": 2 * intrinsics[1][2],
                },
                fp,
                indent=2,
            )

        # we changed the script to fit the test data for blender so now we have to use this naming
        with open(out_path / "transforms_test.json", "w") as fp:
            json.dump(
                {
                    "frames": frames,
                    "camera_angle_x": frames_val[0]["camera_angle_x"],
                    "camera_angle_y": frames_val[0]["camera_angle_y"],
                    "width": 2 * intrinsics[0][2],
                    "height": 2 * intrinsics[1][2],
                },
                fp,
                indent=2,
            )

        # we changed the script to fit the test data for blender so now we have to use this naming
        with open(out_path / "transforms_eval.json", "w") as fp:
            json.dump(
                {
                    "frames": frames_val,
                    "camera_angle_x": frames_val[0]["camera_angle_x"],
                    "camera_angle_y": frames_val[0]["camera_angle_y"],
                    "width": 2 * intrinsics[0][2],
                    "height": 2 * intrinsics[1][2],
                },
                fp,
                indent=2,
            )
