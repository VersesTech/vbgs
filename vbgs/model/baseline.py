import math
import os
import time
import numpy as np
from PIL import Image

import torch
from gsplat import rasterization
from torch import Tensor, optim

import jax.numpy as jnp
from vbgs.camera import transform_uvd_to_points


def inv_sigmoid(x):
    return torch.log(x / (1 - x))


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        device: str = "cuda:0",
        init_from_data=False,
        sub_factor=1,
        fixed_depth=False,
    ):
        self._fixed_depth = fixed_depth
        self.sub_factor = sub_factor
        self.device = torch.device(device)
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        self.intrinsics = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        self._init_gaussians(init_from_data)

    def _init_from_data(self):
        img = jnp.asarray(self.gt_image.detach().cpu().numpy())
        intr = np.eye(4)
        intr[:3, :3] = self.intrinsics.detach().cpu().numpy()
        pts, cols = transform_uvd_to_points(
            img,
            jnp.ones_like(img[..., 0]),
            self.viewmat.detach().cpu().numpy(),
            intr,
            from_opengl=False,
            filter_zero=False,
        )

        idcs = np.random.choice(
            np.arange(4096), self.num_points, replace=False
        )

        return torch.tensor(
            np.array(pts[idcs]), dtype=torch.float32, device=self.device
        ), torch.tensor(
            np.array(cols[idcs]), dtype=torch.float32, device=self.device
        )

    def _init_gaussians(self, init_from_data):
        """Random gaussians"""

        self.rgbs = torch.rand(self.num_points, 3, device=self.device)
        self.scales = 1e-1 * torch.rand(self.num_points, 3, device=self.device)
        if not init_from_data:
            bd = 2
            self._means = bd * (
                torch.rand(self.num_points, 3, device=self.device) - 0.5
            )
        else:
            self._means, self.rgbs = self._init_from_data()
            self.rgbs = inv_sigmoid(self.rgbs)

        if self._fixed_depth:
            self._means = self._means[:, :2]
            self._depths = torch.ones_like(self._means[:, :1])
            # self.scales = self.scales * 1e-3

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    @property
    def means(self):
        if self._fixed_depth:
            return torch.concatenate([self._means, self._depths], axis=-1)
        else:
            return self._means

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        B_SIZE: int = 14,
        verbose=False,
        fit_mask=None,
    ):
        if fit_mask is None:
            # A mask that can be used to filter out parts of the image for the
            # loss, i.e. when 'simulating' continual learning
            fit_mask = torch.ones_like(self.gt_image)

        optimizer = optim.Adam(
            [self.rgbs, self._means, self.scales, self.opacities, self.quats],
            lr,
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        opacities = []
        times = [0] * 3  # project, rasterize, backward
        times_tot = []

        for iter in range(iterations):
            start = time.time()
            bt = time.time()
            renders, _, _ = rasterization(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                self.intrinsics[None],
                self.W,
                self.H,
                packed=False,
            )
            out_img = renders[0]

            torch.cuda.synchronize()
            times[1] += time.time() - start

            loss = mse_loss(
                fit_mask[:: self.sub_factor, :: self.sub_factor]
                * out_img[:: self.sub_factor, :: self.sub_factor],
                fit_mask[:: self.sub_factor, :: self.sub_factor]
                * self.gt_image[:: self.sub_factor, :: self.sub_factor],
            )
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            et = time.time() - bt
            times_tot.append(et)

            if iter % 1 == 0:
                frames.append(
                    (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                )
                opacities.append(self.opacities.detach().cpu().numpy())

        if save_imgs:
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)

            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            frames[-1].save(f"{out_dir}/render.png")
            print("saved in ", f"{out_dir}/render.png")

            frames = frames[::20]
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        if verbose:
            print(f"Total(s):\nProject: {times[0]:.3f}, ", end="")
            print(f"Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}")
            print(f"Per step(s):\nProject: {times[0]/iterations:.5f} ", end="")
            print(f"Rasterize: {times[1]/iterations:.5f}, ", end="")
            print(f"Backward: {times[2]/iterations:.5f}")

        return frames, times_tot, opacities
