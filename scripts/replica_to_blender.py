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
    scenes = [i for i in scenes if "." not in str(i)]
    scenes = ["room0"] + [i for i in scenes if i != "room0"]
    print(scenes)

    for s in scenes:
        print(s)
        data_path = test_scene_path / s

        indices = np.arange(0, 2000, 10).tolist()
        val_indices = np.arange(5, 2000, 10).tolist()

        data_iter = ReplicaDataIterator(data_path)

        frames = []
        frames_val = []
        for i, pc in enumerate(data_iter):
            if i not in indices and i not in val_indices:
                continue

            frame, _ = data_iter.get_frame(i)
            intrinsics, extrinsics = data_iter.load_camera_params(i)

            image = Image.open(frame)

            fx = intrinsics[0][0]
            fy = intrinsics[1][1]

            fov_x = 2 * np.arctan2(intrinsics[0][2], fx)
            fov_y = 2 * np.arctan2(intrinsics[1][2], fy)

            f = {
                "camera_angle_x": float(fov_x),
                "camera_angle_y": float(fov_y),
                "width": 2 * float(intrinsics[0][2]),
                "height": 2 * float(intrinsics[1][2]),
                "file_path": str(frame),
                "rotation": None,
                "transform_matrix": extrinsics.tolist(),
            }

            if i in indices:
                frames.append(f)
            if i in val_indices:
                frames_val.append(f)

        out_path = Path("/home/shared/Replica-blender") / s
        out_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / "transforms_train.json", "w") as fp:
            json.dump(
                {
                    "frames": frames,
                    "camera_angle_x": frames[0]["camera_angle_x"],
                    "camera_angle_y": frames[0]["camera_angle_y"],
                    "width": frames[0]["width"],
                    "height": frames[0]["height"],
                },
                fp,
                indent=2,
            )

        # we changed the script to fit the test data for blender so now we have to use this naming
        with open(out_path / "transforms_test.json", "w") as fp:
            json.dump(
                {
                    "frames": frames,
                    "camera_angle_x": frames[0]["camera_angle_x"],
                    "camera_angle_y": frames[0]["camera_angle_y"],
                    "width": frames[0]["width"],
                    "height": frames[0]["height"],
                },
                fp,
                indent=2,
            )

        # we changed the script to fit the test data for blender so now we have to use this naming
        with open(out_path / "transforms_eval.json", "w") as fp:
            json.dump(
                {
                    "frames": frames_val,
                    "camera_angle_x": frames[0]["camera_angle_x"],
                    "camera_angle_y": frames[0]["camera_angle_y"],
                    "width": frames[0]["width"],
                    "height": frames[0]["height"],
                },
                fp,
                indent=2,
            )
