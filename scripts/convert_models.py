import pickle
import json
import jax.numpy as jnp

import os
from pathlib import Path
import tqdm

import numpy as np

if __name__ == "__main__":
    p = "/home/toon.vandemaele/projects/iclr-rebuttal/vbgs-internal/scripts/data/sweep/"
    p = Path(p)

    files = list(p.glob("**/model_*.json"))
    print(files)
    for f in tqdm.tqdm(files):
        pa = str(f).replace(".json", ".npz")
        if os.path.exists(pa):
            continue

        with open(f, "r") as fp:
            d = json.load(fp)

        mu = np.asarray(d["mu"])
        si = np.asarray(d["si"])
        alpha = np.asarray(d["alpha"])

        with open(pa, "wb") as fp:
            np.savez(fp, mu=mu, si=si, alpha=alpha)

        del d 
        os.remove(str(f))
