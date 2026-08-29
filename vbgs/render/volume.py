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

import numpy as np
import sys
import json

from pathlib import Path

import jax
import jax.numpy as jnp

import vbgs

root_path = Path(vbgs.__file__).parent.parent

# Gaussian splatting imports
import torch

sys.path.append(str(root_path / "../gaussian-splatting"))
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, network_gui
from scene.dataset_readers import readCamerasFromTransforms, CameraInfo
from utils.camera_utils import loadCam
from arguments import PipelineParams
from gaussian_renderer import render as render_cuda
from argparse import ArgumentParser
from utils.sh_utils import RGB2SH, SH2RGB


parser = ArgumentParser(description="Training script parameters")
pipe = PipelineParams(parser)


class CustomArgs:
    resolution = -1
    data_device = "cuda:0"


cargs = CustomArgs()


def render_img(model, cams, idx, bg=0, scale=1.41):
    custom_cam = loadCam(cargs, id=0, cam_info=cams[idx], resolution_scale=1.0)
    net_image = render_cuda(
        custom_cam, model, pipe, bg * torch.ones(3).to("cuda:0"), scale
    )["render"]
    img_ours_cu = net_image.detach().cpu().permute(1, 2, 0).numpy()
    return img_ours_cu.clip(0, 1)


def rot_mat_to_quat(m):
    w = jnp.sqrt((1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    w4 = 4.0 * w
    x = (m[2, 1] - m[1, 2]) / w4
    y = (m[0, 2] - m[2, 0]) / w4
    z = (m[1, 0] - m[0, 1]) / w4
    q = jnp.array([w, x, y, z])
    return q


def covariance_to_scaling_rotation(covariance):
    """Unpack NIW spatial covariance into 3DGS scales and quaternions.

    3DGS expects Sigma = R diag(s^2) R^T. The previous Cholesky path treated a
    row-normalized L as R, which is incorrect for non-axis-aligned Gaussians.
    Use the symmetric eigenframe instead: s = sqrt(lambda), R = V (det +1).
    """
    cov = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    scales = jnp.sqrt(jnp.clip(eigenvalues, 1e-18, None))
    # Ensure a proper rotation (det = +1), not a reflection.
    det = jnp.linalg.det(eigenvectors)
    sign = jnp.where(det < 0.0, -1.0, 1.0)
    eigenvectors = eigenvectors.at[:, :, 0].multiply(sign[:, None])
    wxyz = jax.vmap(rot_mat_to_quat)(eigenvectors)
    return np.array(scales), np.array(wxyz)


def construct_covariance(lower, device="cuda:0"):
    cov = torch.zeros((lower.shape[0], 3, 3), device=device)

    # fill in lower triangle
    cov[:, 0:3, 0] = lower[:, :3]
    cov[:, 1:3, 1] = lower[:, 3:5]
    cov[:, 2:3, 2] = lower[:, 5:]

    # make symmetrical
    cov[:, 0, 1:3] = cov[:, 1:3, 0]
    cov[:, 1, 2] = cov[:, 2, 1]

    return cov


def vbgs_model_to_splat(model_path, device="cuda:0", dtype=torch.float32):
    with open(model_path, "r") as f:
        d = json.load(f)

    mu, si = np.array(d["mu"]), np.array(d["si"])
    alpha = np.array(d["alpha"])

    scaling, rotation = covariance_to_scaling_rotation(si[:, :3, :3])
    mask = scaling.sum(axis=-1) > -1

    model = GaussianModel(3)
    model.max_sh_degree = 0
    model._xyz = torch.tensor(mu[:, :3], dtype=dtype, device=device)
    model._features_dc = torch.tensor(
        RGB2SH(mu[mask, 3:].clip(0, 1)), dtype=dtype, device=device
    ).unsqueeze(1)
    model._features_rest = torch.empty(0).to(device=device, dtype=dtype)
    model._opacity = torch.tensor(
        (alpha[mask] > 0.000001), dtype=dtype, device=device
    )
    model.opacity_activation = lambda x: x
    model._scaling = torch.tensor(scaling[mask], dtype=dtype, device=device)
    model.scaling_activation = lambda x: x
    model._rotation = torch.tensor(rotation[mask], dtype=dtype, device=device)
    model.rotation_activation = lambda x: x
    return model
