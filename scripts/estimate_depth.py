from vbgs.data.replica import ReplicaDataIterator


from pathlib import Path

import numpy as np
import shutil
import cv2
from vbgs.data.depth import load_depth_model, predict_depth

from PIL import Image

from tqdm import trange

if __name__ == "__main__":
    name = "room0"
    data_path = Path(f"/home/shared/Replica/{name}")

    depth_loc = Path(f"/home/shared/Replica-depth_estimated/{name}")
    depth_loc.parent.mkdir(exist_ok=True, parents=True)

    if not depth_loc.exists():
        shutil.copytree(str(data_path), str(depth_loc))

    depth_model = load_depth_model("dav2", "cuda:0")
    data_iter = ReplicaDataIterator(str(depth_loc), None, None)

    for i in trange(len(data_iter)):
        idx = str(i).zfill(6)
        filename = data_iter._datapath / f"frame{idx}.jpg"

        color = np.array(
            Image.open(filename).resize(
                (data_iter.w, data_iter.h), Image.Resampling.NEAREST
            )
        )
        color = color.astype(np.uint8)
        d = predict_depth(color.copy(), *depth_model)

        d_out = d * data_iter.depth_scale
        d_out = d_out.astype(np.uint16)

        outfile = (
            str(filename).replace("frame", "depth").replace(".jpg", ".png")
        )
        cv2.imwrite(outfile, d_out)
