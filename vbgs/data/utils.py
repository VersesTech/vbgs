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

import jax.numpy as jnp


def normalize_data(data, data_params=None):
    """
    Normalize the data such that it has mean 0 and standard deviation 1

    :param data: jnp array of data to be normalized with shape (n_data, n_dims)
    :param data_params: a dictionary containing the normalizing parameters
                        'offset': translation component
                        'stdevs': scaling component.
                        If the value is None, these are computed from the data
    :returns data: jnp array of normalized data with shape (n_data, n_dims)
    :returns data_params: a dictionary containing the normalizing parameters
                          used to normalize this data.
    """
    if data_params is None:
        offset = data.mean(axis=0, keepdims=True)
        data = data - offset
        stdevs = data.std(axis=0, keepdims=True)
        data /= stdevs
    else:
        offset = data_params["offset"].reshape(1, -1)
        stdevs = data_params["stdevs"].reshape(1, -1)
        data = data - offset
        data /= stdevs

    data_params = {"stdevs": stdevs[0], "offset": offset[0]}
    return data, data_params


def create_normalizing_params(*ranges):
    """
    Create a params_dict to normalize the data, given the data range provided
    of shape.
    """
    mi = jnp.array([r[0] for r in ranges])
    ma = jnp.array([r[1] for r in ranges])
    offsets = (mi + ma) / 2
    stdevs = jnp.sqrt((1 / 12) * (ma - mi) ** 2)
    return {"stdevs": stdevs, "offset": offsets}
