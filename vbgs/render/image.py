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

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

from functools import partial


@jax.jit
def select(uv, mean, radii):
    dist = jnp.linalg.norm(mean - uv)
    return dist < radii


def select_gaussians(mean, covariance, alpha, patch_center, patch_radius):
    # Basically filters out the Gaussians that are not in the patch
    U, s, Vt = jax.vmap(jnp.linalg.svd)(covariance[:, :2, :2])
    # Assume circle shape
    radii = 3 * jnp.array(jnp.sqrt(s)).mean(axis=-1, keepdims=True)[:, 0]
    s = jax.vmap(partial(select, patch_center))(
        mean[:, :2, 0], radii + 2 * patch_radius
    )
    s = s.astype(jnp.bool_)
    return mean[s], covariance[s], alpha[s]


@jax.jit
def render_patch(uv, mean, covariance, alpha):
    pdf_fun = jax.vmap(
        lambda x: mvn.pdf(x, mean[:, :2, 0], covariance[:, :2, :2])
    )
    uv_flat = uv.reshape(-1, 2)
    pdf_i = pdf_fun(uv_flat)

    # Instabilities can lead to nan's, i.e. dismiss this component
    pdf_i = jnp.nan_to_num(pdf_i, nan=0)

    pdf_i = pdf_i * alpha.reshape(1, -1)  # (n_pix, n_comp)
    pdf_i = pdf_i / (pdf_i.sum(axis=-1, keepdims=True) + 1e-8)

    pdf_i = pdf_i.reshape((*pdf_i.shape, 1))

    rgb = mean[:, -3:, 0]
    pix = (pdf_i * rgb).sum(axis=1)
    return jnp.clip(pix, 0, 1).reshape(*uv.shape[:2], 3)


def render_img(mean, cova, alpha, shape, patch_size=100):
    """
    mean: the mean of the MVN
    cova: the covariance matrix of the MVN
    inv_stdev: 1/stdev used to normalize the data
    shape: the shape of the image you want to render
    """
    # Normalize pixels indices (0-centered and multiply by 1/stdev)
    u = np.arange(shape[1])
    v = np.arange(shape[0])
    u, v = jnp.meshgrid(u, v)
    uv = jnp.concatenate(
        [u.reshape(*u.shape, 1), v.reshape(*v.shape, 1)], axis=-1
    )

    img = np.zeros(shape=(*shape, 3))
    for i in np.arange(0, shape[0], patch_size):
        for j in np.arange(0, shape[1], patch_size):
            # summarize over components
            uv_i = uv[i : i + patch_size, j : j + patch_size]

            patch_center = uv[i + patch_size // 2, j + patch_size // 2]
            mi, ci, ai = select_gaussians(
                jnp.expand_dims(mean, -1),
                cova,
                alpha,
                patch_center,
                patch_size / 2,
            )
            patch = render_patch(uv_i, mi, ci, ai)
            img[i : i + patch_size, j : j + patch_size] = patch

    return img
