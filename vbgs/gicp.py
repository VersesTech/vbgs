from scipy.spatial import KDTree
import jax.numpy as jnp
import numpy as np
import jax
from typing import Tuple
from functools import partial
import jaxlie as jl


def kd_tree_nn(
    points: jax.Array, test_points: jax.Array, k: int = 1
) -> Tuple[jax.Array, jax.Array]:
    """
    Uses a KD-tree to find the k nearest neighbors to a test point in 3D space.

    Parameters:
        points: [n, d] Array of points.
        test_points: [m, d] points to query
        k: The number of nearest neighbors to find.

    Returns:
        distances: [m, k] Distances to the k nearest neighbors.
        indices: [m, k] Indices of the k nearest neighbors.
    """
    m, d = np.shape(test_points)
    k = int(k)
    args = (points, test_points, k)

    distance_shape_dtype = jax.ShapeDtypeStruct(shape=(m, k), dtype=points.dtype)
    index_shape_dtype = jax.ShapeDtypeStruct(shape=(m, k), dtype=jnp.int32)

    return jax.pure_callback(
        _kd_tree_nn_host, (distance_shape_dtype, index_shape_dtype), *args
    )


def _kd_tree_nn_host(
    points: jax.Array, test_points: jax.Array, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses a KD-tree to find the k nearest neighbors to a test point in 3D space.

    Parameters:
        points: [n, d] Array of points.
        test_points: [m, d] points to query
        k: The number of nearest neighbors to find.

    Returns:
        distances: [m, k] Distances to the k nearest neighbors.
        indices: [m, k] Indices of the k nearest neighbors.
    """
    points, test_points = jax.tree.map(np.asarray, (points, test_points))
    k = int(k)
    tree = KDTree(points, compact_nodes=False, balanced_tree=False)
    if k == 1:
        distances, indices = tree.query(test_points, k=[1])  # unsqueeze k
    else:
        distances, indices = tree.query(test_points, k=k)
    return distances, indices.astype(jnp.int32)


def associate(source, target):
    _, index = kd_tree_nn(target, source, k=1)
    return index


def compute_cost(transform, source, target, cov_source, cov_target, correspondences):
    apply_transform = lambda t, x: (t @ jnp.ones(4).at[:3].set(x))[:3]
    transformed_source = jax.vmap(partial(apply_transform, transform))(source)

    residuals = transformed_source - target[correspondences]
    combined_cov = cov_source + cov_target[correspondences]
    weights = jnp.linalg.inv(combined_cov)

    # Compute cost
    cost_step = lambda r, w: r @ w @ r.T
    cost = jax.vmap(cost_step)(residuals, weights)
    cost = jnp.sum(cost)
    return cost


def optimize_transform(
    transform,
    source,
    target,
    cov_source,
    cov_target,
    correspondences,
    lr=1e-4,
    iters=10,
):
    cost_grad = jax.value_and_grad(
        lambda x: compute_cost(
            x, source, target, cov_source, cov_target, correspondences
        )
    )

    def scan_step(transform, x):
        cost, grad = cost_grad(transform)
        transform = (
            transform @ jl.SE3.exp(-lr * jl.SE3.from_matrix(grad).log()).as_matrix()
        )
        return transform, cost

    transform, costs = jax.lax.scan(scan_step, transform, jnp.zeros(iters))
    return transform, costs[-1]


def icp(source, target, cov_source, cov_target, assoc_iters=2, trans_iters=10):
    """Calculate the transform mapping source onto target."""
    # Calculate initial guess.
    mean_source = jnp.mean(source, axis=0)
    mean_target = jnp.mean(target, axis=0)
    H = (source - mean_source).T @ (target - mean_target)
    U, _, VH = jnp.linalg.svd(H)
    R = VH.T @ U.T
    if jnp.linalg.det(R) < 0:
        # reflective case
        R = R.at[:3, 2].set(R[:3, 2] * -1)

    t = mean_target - R @ mean_source  # target = R @ source + t
    transform = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(t)
    apply_transform = lambda t, x: (t @ jnp.ones(4).at[:3].set(x))[:3]
    for _ in range(assoc_iters):
        assoc = associate(
            jax.vmap(partial(apply_transform, transform))(source),
            target,
        )
        transform, cost = optimize_transform(
            transform,
            source,
            target,
            cov_source,
            cov_target,
            assoc,
            iters=trans_iters,
        )
    return transform
