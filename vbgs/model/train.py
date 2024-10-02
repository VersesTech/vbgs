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
from jax.nn import softmax
from jax.scipy.special import logsumexp

import numpy as np

from vbgs.vi import utils


def get_likelihood_sst(self, data, weights):
    """
    Computes the sufficient statistics for the likelihood
    """
    x = data[0] if isinstance(data, tuple) else data

    counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
    shape = counts_shape + (1,) * self.event_dim
    counts = jnp.ones(counts_shape)
    sample_dims = self.get_sample_dims(x)

    weights = (
        self.expand_event_dims(weights)
        if weights is not None
        else jnp.ones(shape)
    )

    likelihood_stats = self.likelihood.statistics(data)

    param_stats = self.map_stats_to_params(likelihood_stats, counts)
    summed_stats = self.sum_stats_over_samples(
        param_stats, weights, sample_dims
    )

    return summed_stats, {
        "likelihood_stats": likelihood_stats,
        "counts": counts,
        "param_stats": param_stats,
        "weights": weights,
        "summmed_stats": summed_stats,
    }


def fit_gmm(initial_model, model, data):
    data = jnp.expand_dims(data, -1)
    d = model.mixture.expand_to_categorical_dims(data)

    # spatial and color information is stored concatenated in the final dim
    ds, dc = d[:, :, :-3], d[:, :, -3:]
    space_logprob = model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = model.delta.expected_log_likelihood(dc)
    prior_logprob = model.mixture.prior.log_mean()
    logprob = space_logprob + color_logprob + prior_logprob
    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))
    posteriors = softmax(logprob, mixdims)

    cat_i = initial_model.mixture.expand_to_categorical_dims(data)
    ps = initial_model.mixture._to_stats(
        posteriors, initial_model.mixture.get_sample_dims(cat_i)
    )
    ss, _ = get_likelihood_sst(
        initial_model.mixture.likelihood, cat_i[:, :, :-3], posteriors
    )
    cs, _ = get_likelihood_sst(
        initial_model.delta, cat_i[:, :, -3:], posteriors
    )
    model.prior.update_from_statistics(ps, **initial_model.mixture.pi_opts)
    model.likelihood.update_from_statistics(
        ss, **initial_model.mixture.likelihood_opts
    )
    model.delta.update_from_statistics(
        cs, **initial_model.mixture.likelihood_opts
    )
    return model


@jax.jit
def compute_elbo_delta(model, data):
    """
    Computes the ELBO for the provided data.
    Note: if the data is too large to process, you want to batch around this
          function (see `fit_gmm_step`)

    :param model : Mixture model instance which represents the joint
                   distribution of space, color and the latent z
    :param data : the data points to consider
    :returns elbo: array of elbo's for each data point
    :returns posteriors: array of the posterior distribution q(z) for all data
    """
    d = model.mixture.expand_to_categorical_dims(data)

    # spatial and color information is stored concatenated in the final dim
    ds, dc = d[:, :, :-3], d[:, :, -3:]

    space_logprob = model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = model.delta.expected_log_likelihood(dc)
    prior_logprob = model.mixture.prior.log_mean()

    logprob = space_logprob + color_logprob + prior_logprob

    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))

    elbo_contrib = logsumexp(logprob, mixdims)

    prior_kl = model.mixture.prior.kl_divergence()
    space_kl = model.mixture.likelihood.kl_divergence().sum(mixdims)
    color_kl = model.delta.kl_divergence().sum(mixdims)

    elbo = elbo_contrib - space_kl - prior_kl - color_kl

    # elbo = elbo.sum(sample_dims)

    posteriors = softmax(logprob, mixdims)
    return elbo, posteriors


def fit_gmm_step(
    initial_model,
    model,
    data,
    batch_size,
    prior_stats=None,
    space_stats=None,
    color_stats=None,
):
    """
    Compute a single update step for the `DeltaMixture` using the assignments
    upon the initial model, but adding the sst to the model.

    :param initial_model: DeltaMixture before having applied a single udpate
    :param model: DeltaMixture of the model having applied the previous updates
                  is used to apply the upate upon
    :param data: The data to fit the model to. Preferrably a numpy array, to
                 only populate the GPU when it's necessary.
    :param batch_size: size of a single batch processed by GPU
    :param prior_stats: The collected sufficient statistics of the prior. None
                        at step 0, after that it should contain the prior_stats
                        returned at the previous step.
    :param space_stats: The sufficient statistics of the spatial likelihood.
                        None at step 0, after that it should contain the
                        space_stats returned at the previous step.
    :param color_stats: The sufficient statistics of the color likelihood. None
                        at step 0, after that it should contain the color_stats
                        returned at the previous step.
    :returns model: DeltaMixture model after updating
    """
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    for batch_idx in range(n_batches):
        xi = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        xi = jnp.expand_dims(jnp.array(xi), -1)

        size = xi.shape[0]
        if size < batch_size:
            # Concat zeros, so that the posteriors are still computed with jit
            # NOTE: the elbo will not be correct since it's contributing the
            # log likelihood of the augmented zeros. But since we don't use the
            # elbo here, it is not a problem.
            xi = jnp.concatenate(
                [xi, jnp.zeros((batch_size - size, *xi.shape[1:]))],
                axis=0,
            )
            _, posteriors = compute_elbo_delta(initial_model, xi)
            xi = xi[:size]
            posteriors = posteriors[:size]
        else:
            _, posteriors = compute_elbo_delta(initial_model, xi)

        cat_i = initial_model.mixture.expand_to_categorical_dims(xi)

        # Compute the sufficient statistics using optimus methods
        ps = model.mixture._to_stats(
            posteriors, initial_model.mixture.get_sample_dims(cat_i)
        )

        ss, _ = get_likelihood_sst(
            initial_model.mixture.likelihood, cat_i[:, :, :-3], posteriors
        )

        cs, _ = get_likelihood_sst(
            initial_model.delta, cat_i[:, :, -3:], posteriors
        )

        # Aggregate the sufficient statistics in loop
        if batch_idx == 0 and prior_stats is None:
            prior_stats = ps
            space_stats = ss
            color_stats = cs
        else:
            prior_stats = utils.apply_add(ps, prior_stats)
            space_stats = utils.apply_add(ss, space_stats)
            color_stats = utils.apply_add(cs, color_stats)

    model.mixture.prior.update_from_statistics(
        prior_stats, **initial_model.mixture.pi_opts
    )
    model.mixture.likelihood.update_from_statistics(
        space_stats, **initial_model.mixture.likelihood_opts
    )
    model.delta.update_from_statistics(
        color_stats, **initial_model.mixture.likelihood_opts
    )

    return model, prior_stats, space_stats, color_stats
