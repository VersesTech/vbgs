from vbgs.data.habitat import HabitatDataIterator, load_camera_params

from pathlib import Path

import numpy as np
import shutil
import cv2

if __name__ == "__main__":
    name = "room0"
    data_path = Path(f"/home/shared/splatam/{name}")

    depth_loc = Path(f"/home/shared/splatam-d_estimated/{name}")
    depth_loc.mkdir(exist_ok=True, parents=True)

    for f in data_path.glob("*"):
        shutil.copyfile(f, depth_loc / f.name)

    estimated_iter = HabitatDataIterator(
        str(depth_loc), "", None, estimate_depth=True, from_opengl=False
    )

    for i in range(len(estimated_iter._frames)):
        _, d, _, _ = estimated_iter._get_frame_rgbd(i)
        *_, s = load_camera_params(estimated_iter._frames[i])

        filename = str(estimated_iter._frames[i]).replace(
            ".jpeg", "_depth.jpeg"
        )

        depth_image = (255 * (d / s)).astype(np.uint8)
        cv2.imwrite(filename, depth_image)
