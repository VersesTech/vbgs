
from jax.nn import softmax
import jax.numpy as jnp


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