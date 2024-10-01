import time
from functools import partial

import equinox

import jax
import jax.numpy as jnp
import jax.random as jr

from vbgs.vi.conjugate.multinomial import Multinomial
from vbgs.vi.conjugate.mvn import MultivariateNormal
from vbgs.vi.models.mixture import Mixture
from vbgs.vi.utils import ArrayDict

from vbgs.data.utils import normalize_data
from vbgs.data.image import image_to_data
from vbgs.model.utils import random_mean_init, transform_mvn
from vbgs.render import render_img


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


def fit_delta_vbem_image(
    key,
    n_components,
    img,
    n_iters,
    init_random,
    position_scale,
    ret_model=False,
):
    data = image_to_data(img)
    x, data_params = normalize_data(data)

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=x,
        component_shape=(n_components,),
        event_shape=(5, 1),
        init_random=init_random,
        add_noise=True,
    )

    key, subkey = jr.split(key)

    frames, times = [], []
    x = jnp.array(x)
    for i in range(n_iters):
        model = get_image_delta_mixture(
            key=subkey,
            n_components=n_components,
            mean_init=mean_init,
            beta=0,
            learning_rate=1,
            dof_offset=1,
            position_scale=position_scale,
        )

        bt = time.time()
        model = fit_gmm(model, model, x)
        et = time.time()
        times.append(et - bt)

        mu, si = model.denormalize(data_params)
        rendered_img = render_img(mu, si, model.prior.alpha, img.shape[:2])
        frames.append(rendered_img)

    if ret_model:
        return frames, times, (model, data_params)
    else:
        return frames, times
