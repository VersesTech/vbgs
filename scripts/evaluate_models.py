from pathlib import Path

import vbgs
from vbgs.render.volume import (
    readCamerasFromTransforms,
    render_img,
    vbgs_model_to_splat,
)
from vbgs.metrics import calc_mse, calc_psnr
from PIL import Image
import numpy as np

import json
import pickle

from tqdm import trange


def crop_center(x):
    return x[170 : 170 + 340, 300:900]


def evaluate(model, cameras):
    psnrs = []
    mses = []

    imgs = {"gt": [], "predict": []}
    for i in range(100):
        x_hat = render_img(model, cameras, i, 0, scale=1.0)
        x = np.array(Image.open(cameras[i].image_path)) / 255.0

        x, x_hat = crop_center(x), crop_center(x_hat)

        imgs["gt"].append(x)
        imgs["predict"].append(x_hat)

        psnrs.append(calc_psnr(x, x_hat))
        mses.append(calc_mse(x, x_hat))

    return {"psnr": psnrs, "mse": mses}, imgs


if __name__ == "__main__":
    scene = "room0"

    mode = "depth_estimate"
    # mode = "ground_truth"

    root_path = Path(vbgs.__file__).parent.parent
    for reassign in [True, False]:
        model_path = root_path / Path(
            # f"scripts/data/sweep/splatam_rooms_estimated_depth/room0_shuffle:True/nc:100000/randinit:True_reassign:{reassign}"
            f"scripts/data/sweep/splatam_rooms_{mode}/{scene}_shuffle:True/nc:100000/randinit:True_reassign:{reassign}"
        )

        store_path = (
            root_path
            / f"data/evaluated_replica_{mode}/{scene}_VBGS_100K_Reassign:{reassign}"
        )
        store_path.mkdir(exist_ok=True, parents=True)

        cameras = readCamerasFromTransforms(
            f"/home/toon.vandemaele/projects/iclr-rebuttal/vbgs-internal/resources/large-datasets/{scene}",
            "transforms_eval.json",
            True,
        )

        for i in trange(200):
            model_i = vbgs_model_to_splat(
                model_path / f"model_{i:02d}.npz", device="cuda:0"
            )
            results, imgs = evaluate(model_i, cameras)
            with open(store_path / f"{i:02d}_results.json", "w") as fp:
                json.dump(results, fp)
            # with open(store_path / f"{i:02d}_imgs.pickle", "wb") as fp:
            #     pickle.dump(imgs, fp)
