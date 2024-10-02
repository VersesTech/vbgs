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

import json

import jax.random as jr
import jax.numpy as jnp


def store_model(model, data_params, filename):
    mu, si = model.denormalize(data_params, clip_val=None)
    alpha = model.prior.alpha.reshape(-1)

    # Storing the model as a json
    model_dict = {
        "mu": mu.tolist(),
        "si": si.tolist(),
        "alpha": alpha.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(model_dict, f, indent=2)


def random_mean_init(
    key, x, component_shape, event_shape, init_random=False, add_noise=True
):
    """
    Sample a mean init for initializing the GMM.
    """
    _, param_init_key = jr.split(key)
    if init_random or x is None:
        mean_init = jr.uniform(
            param_init_key,
            component_shape + event_shape,
            minval=-1.70,
            maxval=1.70,
        )
        # initialize the color values on zeros (for normal distribution)
        # this is a good thing. At the center
        mean_init = mean_init.at[:, -3:].set(0)
    else:
        # Initialize the components around the points from the data
        idcs = jr.randint(
            param_init_key, component_shape, minval=0, maxval=len(x)
        )

        mean_init = jnp.zeros(component_shape + event_shape)
        mean_init = mean_init.at[:].set(x[idcs].reshape((-1, *event_shape)))

    if add_noise:
        key, subkey = jr.split(param_init_key)
        mean_init = (
            mean_init + jr.normal(subkey, shape=mean_init.shape) * 0.025
        )

    return mean_init


def transform_mvn(scale, offset, mean, cova):
    A = jnp.diag(scale)
    new_mean = A.dot(mean) + offset
    new_cova = jnp.dot(A, jnp.dot(cova, A.T))
    return new_mean, new_cova
