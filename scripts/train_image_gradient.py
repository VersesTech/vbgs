import numpy as np
import torch
import json

from tqdm import tqdm

import datasets

import hydra
from omegaconf import DictConfig, OmegaConf

from vbgs.model.baseline import SimpleTrainer
from vbgs.metrics import calc_mse, calc_psnr


def run_experiment(
    n_components,
    fixed_depth=False,
    init_on_data=False,
    device="cuda:0",
    lr=0.1,
):
    min_opacity = 0.005

    def fit(img):
        img = torch.tensor(img, dtype=torch.float32)
        trainer = SimpleTrainer(
            img.to(device),
            num_points=n_components,
            fixed_depth=fixed_depth,
            init_from_data=init_on_data,
            device=device,
        )

        f, times, opacities = trainer.train(100, lr=lr)

        # The result from the gradient is a uint8
        pred = (f[-1].astype(np.float32) / 255.0).clip(0, 1.0)
        img_np = img.cpu().numpy()
        n_used = (opacities[-1] > min_opacity).sum()

        return {
            "mse": calc_mse(img_np, pred).item(),
            "psnr": calc_psnr(img_np, pred).item(),
            "times": np.sum(times),
            "n_used": float(n_used),
        }

    # Evaluate on validation set (10k images)
    dataset = datasets.load_dataset("Maysee/tiny-imagenet", split="valid")

    # Track some metrics
    all_metrics = {}
    for image in tqdm(dataset["image"]):
        img = np.array(image) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        m = fit(img)
        all_metrics = {k: all_metrics.get(k, []) + [v] for k, v in m.items()}

    return all_metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="gradient_imagenet",
)
def main(cfg: DictConfig) -> None:
    results = run_experiment(
        n_components=cfg.model.n_components,
        fixed_depth=cfg.model.fixed_depth,
        init_on_data=cfg.model.init_on_data,
        device=f"cuda:{cfg.device}",
        lr=cfg.train.lr,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
