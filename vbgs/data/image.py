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

import jax
import jax.numpy as jnp


@jax.jit
def image_to_data(img):
    """
    Load an image together with the UV values, such that it can be processed
    by the GMM.
    """
    u, v = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    data = jnp.concatenate(
        [
            (u.reshape(-1, 1)),
            (v.reshape(-1, 1)),
            img[..., 0].reshape(-1, 1),
            img[..., 1].reshape(-1, 1),
            img[..., 2].reshape(-1, 1),
        ],
        axis=1,
    )
    return data


def to_patches(data, img, patch_side=8):
    data = data.reshape((*img.shape[:2], 5))
    patches, masks = [], []
    for a in range(0, img.shape[0], patch_side):
        for b in range(0, img.shape[1], patch_side):
            patches.append(
                data[a : a + patch_side, b : b + patch_side].reshape(-1, 5)
            )

            mask = jnp.zeros(img.shape)
            mask = mask.at[a : a + patch_side, b : b + patch_side].set(1.0)

            masks.append(mask)
    return patches, masks
