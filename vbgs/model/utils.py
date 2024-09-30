import jax.random as jr
import jax.numpy as jnp


def random_mean_init(
    key, x, component_shape, event_shape, init_random=False, add_noise=True
):
    """
    Sample a mean init for initializing the GMM.
    """
    _, param_init_key = jr.split(key)
    if init_random or x is None:
        # mean_init = jr.normal(param_init_key, component_shape + event_shape)
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
