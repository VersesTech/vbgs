import jax
import jax.numpy as jnp


@jax.jit
def image_to_data(img):
    """
    Load an image together with the UV values, such that it can be processed
    by the GMM.
    """
    u, v = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    data = jnp.concatenate(
        [
            (u.reshape(-1, 1)),
            (v.reshape(-1, 1)),
            img[..., 0].reshape(-1, 1),
            img[..., 1].reshape(-1, 1),
            img[..., 2].reshape(-1, 1),
        ],
        axis=1,
    )
    return data


def to_patches(data, img, patch_side=8):
    data = data.reshape((*img.shape[:2], 5))
    patches, masks = [], []
    for a in range(0, img.shape[0], patch_side):
        for b in range(0, img.shape[1], patch_side):
            patches.append(
                data[a : a + patch_side, b : b + patch_side].reshape(-1, 5)
            )

            mask = jnp.zeros(img.shape)
            mask = mask.at[a : a + patch_side, b : b + patch_side].set(1.0)

            masks.append(mask)
    return patches, masks
