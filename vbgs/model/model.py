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

from functools import partial
from collections import namedtuple

import equinox

import jax
import jax.numpy as jnp

from vbgs.vi.conjugate.mvn import MultivariateNormal
from vbgs.vi.models.mixture import Mixture

from vbgs.model.utils import transform_mvn


Splat = namedtuple("Splat", ["mu", "si", "alpha"])


class DeltaMixture(equinox.Module):
    """
    A small compositional class to allow for the use of previously written code
    that calls `model.likelihood` and `model.prior`
    """

    mixture: Mixture
    delta: MultivariateNormal

    def __init__(self, mixture, delta):
        self.mixture = mixture
        self.delta = delta

    @property
    def likelihood(self):
        return self.mixture.likelihood

    @property
    def prior(self):
        return self.mixture.prior

    def denormalize(self, params, clip_val=None):
        """
        Invert the normalization step applied to the data, such that
        the model is now in the space of the original data.

        :param params: normalizing params in a dictionary as created by
                       `normalize_data` or `create_normalizing_params`
        :param clip_val: minimum value to have on the diagonal of the
                         covariance matrices. Defaults to None.
                         For 3D recommended None, for 2D recommended 0.05.
        :returns mu: the denormalized means of the spatial and color components
                     concatenated as a multivariate Normal.
        :returns si: the denormalized covariances of the spatial and color
                     components, as a multivariate Normal.
        """
        mu_uv = self.mixture.likelihood.mean[:, :, 0]
        si_uv = self.mixture.likelihood.expected_sigma()

        mu_rgb = self.delta.likelihood.mean[:, :, 0]
        si_rgb = jnp.eye(3).reshape(-1, 3, 3)

        n = self.mixture.likelihood.event_shape[0] + self.delta.event_shape[0]
        mu = jnp.zeros((mu_uv.shape[0], n))
        mu = mu.at[:, :-3].set(mu_uv)
        mu = mu.at[:, -3:].set(mu_rgb)

        si = jnp.zeros((mu_uv.shape[0], n, n))
        si = si.at[:, :-3, :-3].set(si_uv)
        si = si.at[:, -3:, -3:].set(si_rgb)

        mu, si = jax.vmap(
            partial(
                transform_mvn,
                params["stdevs"].flatten(),
                params["offset"].flatten(),
            )
        )(mu, si)

        if clip_val is not None:
            si_diag = jnp.diagonal(si, axis1=1, axis2=2).clip(
                clip_val, jnp.inf
            )
            si = jax.vmap(lambda x, y: jnp.fill_diagonal(x, y, inplace=False))(
                si, si_diag
            )

        return mu, si

    def extract_model(self, data_params):
        mu, si = self.denormalize(data_params, clip_val=None)
        alpha = self.prior.alpha.reshape(-1)
        return Splat(mu, si, alpha)