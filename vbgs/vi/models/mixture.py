from typing import Union, Optional, Tuple
from jaxtyping import Array

from jax.tree_util import tree_map
import jax.numpy as jnp

from vbgs.vi.conjugate.base import Conjugate
from vbgs.vi.distribution import Distribution
from vbgs.vi.models.base import Model
from vbgs.vi.utils import ArrayDict


class Mixture(Model):
    pytree_data_fields = ("likelihood", "prior")
    pytree_aux_fields = (
        "pi_opts",
        "likelihood_opts",
        "assign_unused_opts",
        "batch_shape_prior",
        "event_shape_prior",
        "flattened_batch_shape",
    )

    def __init__(
        self,
        likelihood: Conjugate,
        prior: Conjugate,
        pi_opts: Optional[dict] = None,
        likelihood_opts: Optional[dict] = None,
        assign_unused_opts: Optional[dict] = None,
    ):
        assert prior.batch_dim + prior.event_dim == likelihood.batch_dim
        super().__init__(
            likelihood.default_event_dim,
            prior.batch_shape,
            likelihood.event_shape,
        )

        self.likelihood = likelihood
        self.prior = prior

        self.pi_opts = (
            pi_opts if pi_opts is not None else {"lr": 1.0, "beta": 0.0}
        )
        self.likelihood_opts = (
            likelihood_opts
            if likelihood_opts is not None
            else {"lr": 1.0, "beta": 0.0}
        )
        self.assign_unused_opts = (
            assign_unused_opts
            if assign_unused_opts is not None
            else {"d_alpha_thr": 1.0, "fill_value": 1.0}
        )

        self.batch_shape_prior = prior.batch_shape
        self.event_shape_prior = prior.event_shape
        flattened_batch_shape = 1
        for i in self.batch_shape_prior:
            flattened_batch_shape *= i
        self.flattened_batch_shape = (flattened_batch_shape,)

    def get_sample_dims(self, data):
        if type(data) is tuple:
            sample_dims = tuple(
                range(
                    len(data[0].shape)
                    - self.batch_dim
                    - self.prior.event_dim
                    - self.likelihood.event_dim
                )
            )
        else:
            sample_dims = tuple(
                range(
                    len(data.shape)
                    - self.batch_dim
                    - self.prior.event_dim
                    - self.likelihood.event_dim
                )
            )
        return sample_dims

    def get_sample_shape(self, data):
        sample_dims = self.get_sample_dims(data)
        return sample_dims

    def expand_to_categorical_dims(self, data: Array) -> Array:
        mix_dims = tuple(
            range(
                -self.prior.event_dim - self.likelihood.event_dim,
                -self.likelihood.event_dim,
            )
        )
        if type(data) is tuple:
            data = tree_map(lambda d: jnp.expand_dims(d, mix_dims), data)
        else:
            data = jnp.expand_dims(data, mix_dims)
        return data

    def expand_to_categorical_dims_for_probs(
        self, inputs: Union[Tuple[Distribution], Distribution]
    ) -> Union[Tuple[Distribution], Distribution]:
        mix_dims = tuple(range(-self.prior.event_dim, 0))
        if isinstance(inputs, tuple):
            expanded_inputs = tree_map(
                lambda x: x.expand_batch_shape(mix_dims),
                inputs,
                is_leaf=lambda x: isinstance(x, Distribution),
            )
        else:
            expanded_inputs = inputs.expand_batch_shape(mix_dims)
        return expanded_inputs

    def _to_stats(self, posterior: Array, sample_dims: int) -> ArrayDict:
        return ArrayDict(
            eta=ArrayDict(eta_1=posterior.sum(sample_dims)), nu=None
        )
