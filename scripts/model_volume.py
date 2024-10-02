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
import jax.random as jr

from vbgs.vi.conjugate.mvn import MultivariateNormal
from vbgs.vi.conjugate.multinomial import Multinomial
from vbgs.vi.models.mixture import Mixture
from vbgs.vi.utils import ArrayDict

from vbgs.model.model import DeltaMixture


def get_volume_delta_mixture(
    key,
    n_components,
    mean_init,
    position_event_shape=(2, 1),
    color_event_shape=(3, 1),
    beta=1,
    learning_rate=1,
    dof_offset=1,
    position_scale=None,
    default_event_dim=2,
):
    component_shape = (n_components,)
    if position_scale is None:
        position_scale = jnp.sqrt(n_components)

    key, subkey = jr.split(key)

    # Create prior parameters
    # -----------------------

    # Likelihood (Multivariate Normal)
    likelihood_prior_params = MultivariateNormal.init_default_params(
        component_shape,
        position_event_shape,
        position_scale * 15,
        dof_offset=dof_offset,
        default_event_dim=default_event_dim,
    )

    likelihood_prior_params = ArrayDict(
        mean=likelihood_prior_params.mean,
        kappa=likelihood_prior_params.kappa / 1e3,
        u=likelihood_prior_params.u * 100,
        n=likelihood_prior_params.n,
    )
    likelihood_params = ArrayDict(
        # Initialize the likelihood parameters on mean init
        mean=mean_init[:, :-3, :],
        # We trust the position prior
        kappa=likelihood_prior_params.kappa / 1e3,
        # But we increase the range of it
        u=likelihood_prior_params.u,
        n=likelihood_prior_params.n,
    )

    # Delta prior
    # we approximate the delta distribution with an MVN with a very narrow var
    delta_prior_params = MultivariateNormal.init_default_params(
        component_shape,
        color_event_shape,
        scale=1e5,
        dof_offset=dof_offset,
        default_event_dim=default_event_dim,
    )
    delta_prior_params = ArrayDict(
        mean=delta_prior_params.mean,
        kappa=delta_prior_params.kappa / 1e2,
        # We want to initialize with a large variance
        u=delta_prior_params.u / 100,
        n=delta_prior_params.n,
    )

    delta_params = ArrayDict(
        mean=mean_init[:, -3:, :],
        kappa=delta_prior_params.kappa,
        # We want to initialize with a large variance
        u=delta_prior_params.u * 1e5,
        n=delta_prior_params.n,
    )

    # Create the models
    # -----------------
    key, subkey = jr.split(key)
    prior = Multinomial(
        batch_shape=(),
        event_shape=component_shape,
        initial_count=1 / component_shape[0],
        init_key=subkey,
    )

    key, subkey = jr.split(key)
    likelihood = MultivariateNormal(
        batch_shape=component_shape,
        event_shape=position_event_shape,
        event_dim=len(position_event_shape),
        dof_offset=dof_offset,
        init_key=subkey,
        params=likelihood_params,
        prior_params=likelihood_prior_params,
    )

    key, subkey = jr.split(key)
    delta = MultivariateNormal(
        batch_shape=component_shape,
        event_shape=color_event_shape,
        event_dim=len(color_event_shape),
        dof_offset=dof_offset,
        init_key=subkey,
        params=delta_params,
        prior_params=delta_prior_params,
        fixed_precision=True,  # Crucial!
    )

    opts = {"lr": learning_rate, "beta": beta}
    mixture = Mixture(likelihood, prior, pi_opts=opts, likelihood_opts=opts)
    return DeltaMixture(mixture, delta)
