import rich
import json
import copy
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import jax.random as jr
import jax.numpy as jnp

from pathlib import Path

from tqdm import tqdm

import datasets as ds

import vbgs

from vbgs.vi.conjugate.mvn import MultivariateNormal
from vbgs.vi.conjugate.multinomial import Multinomial
from vbgs.vi.models.mixture import Mixture
from vbgs.vi.utils import ArrayDict

from vbgs.model.model import DeltaMixture
from vbgs.model.utils import random_mean_init
from vbgs.data.image import image_to_data
from vbgs.data.utils import normalize_data
from vbgs.model.train import fit_gmm
from vbgs.metrics import calc_mse, calc_psnr
from vbgs.render import render_img


resources_path = Path(vbgs.__file__).parent.parent / "resources"


def get_model(
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
        # Seemed to work well on 3D
        # Note: 15 * sqrt(n_components) is good if init random,
        # if init on data, this can be more narrow (i.e. 1 * sqrt(n_comp))
        position_scale = 15 * jnp.sqrt(n_components)

    key, subkey = jr.split(key)

    # Create prior parameters
    # -----------------------

    # Likelihood (Multivariate Normal)
    likelihood_prior_params = MultivariateNormal.init_default_params(
        component_shape,
        position_event_shape,
        position_scale,
        dof_offset=dof_offset,
        default_event_dim=default_event_dim,
    )

    likelihood_prior_params = ArrayDict(
        mean=likelihood_prior_params.mean,
        kappa=likelihood_prior_params.kappa / 1e2,
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
        scale=1e4,
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


def run_experiment(
    n_components,
    init_random,
    beta=0,
    learning_rate=1,
    batch_size=1,
    dof=1,
    n_iters=1,
    scale=None,
    seed=123,
):
    def fit(key, img):
        data = image_to_data(img)
        x, data_params = normalize_data(data)

        key, subkey = jr.split(key)
        mean_init = random_mean_init(
            subkey,
            x,
            component_shape=(n_components,),
            event_shape=(5, 1),
            init_random=init_random,
            add_noise=False,
        )

        model = get_model(
            key,
            n_components=n_components,
            mean_init=mean_init,
            beta=beta,
            learning_rate=learning_rate,
            dof_offset=dof,
            position_scale=scale,
        )

        initial_model = copy.deepcopy(model)
        for i in range(n_iters):
            model = fit_gmm(initial_model, model, x)

        mu, si = model.denormalize(data_params)

        rendered_img = render_img(mu, si, model.prior.alpha, img.shape[:2])

        mse_ = calc_mse(img.astype(np.float32), rendered_img.clip(0, 1.0))
        psnr = calc_psnr(img.astype(np.float32), rendered_img.clip(0, 1.0))

        n_used = int((model.prior.alpha > model.prior.prior_alpha.min()).sum())
        metrics = {"mse": mse_.item(), "psnr": psnr.item(), "n_used": n_used}
        return key, metrics

    key = jr.PRNGKey(seed)

    # Evaluate on validation set (10k images)
    dataset = ds.load_dataset("Maysee/tiny-imagenet", split="valid")["image"]

    # Track some metrics
    metrics = {}
    for image in tqdm(dataset):
        img = jnp.array(image) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        key, subkey = jr.split(key)
        subkey, m = fit(subkey, img)

        metrics = {k: metrics.get(k, []) + [v] for k, v in m.items()}

    rich.print(metrics)

    with open("results.json", "w") as f:
        json.dump(metrics, f)

    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="imagenet")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    results = run_experiment(
        n_components=cfg.model.n_components,
        n_iters=cfg.train.n_iters,
        learning_rate=cfg.train.learning_rate,
        beta=cfg.train.beta,
        init_random=cfg.model.init_random,
        seed=cfg.seed,
        batch_size=cfg.train.batch_size,
        dof=cfg.model.dof,
        scale=cfg.model.scale,
    )

    results.update({"config": OmegaConf.to_container(cfg)})
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
