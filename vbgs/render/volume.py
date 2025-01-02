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

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxsplat as jsplat

import vbgs

root_path = Path(vbgs.__file__).parent.parent


def opengl_to_colmap_frame(cam):
    return cam.at[:3, 1:3].set(cam[:3, 1:3] * -1)


def render_gsplat(
    mu, 
    si, 
    alpha,
    cam_to_world,
    intrinsics,
    height,
    width,
    bg=None,
    glob_scale=1.0,
    clip_thresh=0.01,
    block_size=16,
    from_opengl=True
):
    """Uses the gsplats rasterization code to render a vbgs splat.

    Args:
        mu: The 6D means of the gaussians. [N, 6]
        si: The corresponding covariances of the gaussians. [N, 6, 6]
        alpha: The assignments. [N]
        cam_to_world: A camera pose to render from. [4, 4]
        intrinsics: Camera intrinsics. [3, 3]
        height: The desired frame height
        width: The desired frame width
        bg: [Optional] The backgroundcolor, will be black if unset.
    """
    scales, quats = covariance_to_scaling_rotation(si[:, :3, :3])
    colors = mu[:, 3:]
    center_points = mu[:, :3]
    c = int(intrinsics[0, 2]), int(intrinsics[1, 2])
    f = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    alpha = alpha[..., None] > 0.01
    if bg is None:
        bg = jnp.zeros(3)
    if from_opengl:
        cam_to_world = opengl_to_colmap_frame(cam_to_world)
    
    world_to_cam = jnp.linalg.inv(cam_to_world)
    return jsplat.render(
        center_points.astype(jnp.float32),
        scales.astype(jnp.float32),
        quats.astype(jnp.float32),
        colors.astype(jnp.float32),
        alpha.astype(jnp.float32),
        viewmat=world_to_cam.astype(jnp.float32),
        background=bg.astype(jnp.float32),
        img_shape=(height, width),
        glob_scale=glob_scale,
        f=f,
        c=c,
        clip_thresh=clip_thresh,
        block_size=block_size,
    )


def rot_mat_to_quat(m):
    w = jnp.sqrt((1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    w4 = 4.0 * w
    x = (m[2, 1] - m[1, 2]) / w4
    y = (m[0, 2] - m[2, 0]) / w4
    z = (m[1, 0] - m[0, 1]) / w4
    q = jnp.array([w, x, y, z])
    return q


def covariance_to_scaling_rotation(covariance):
    # Decompose into L @ L.T
    mat_L = jax.vmap(jnp.linalg.cholesky)(covariance)

    # Decompose into R @ S
    scales = jax.vmap(lambda x: jnp.linalg.norm(x, axis=-1))(mat_L)

    # Rotation is basically the normalized matrix left over
    rotation = mat_L / jnp.expand_dims(scales, -1)

    rec = jax.vmap(lambda r, s: jnp.dot(r, jnp.dot(s, jnp.dot(s.T, r.T))))
    scale_mat = jnp.eye(3).reshape((1, 3, 3)) * scales.reshape(-1, 3, 1)
    res = rec(rotation, scale_mat)

    # Convert to quaternion
    wxyz = jax.vmap(rot_mat_to_quat)(rotation)
    return np.array(scales), np.array(wxyz)
