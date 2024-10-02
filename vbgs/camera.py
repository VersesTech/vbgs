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

# TODO: Map all numpy code to jax.numpy
import numpy as np

from functools import partial

# Frame here is the image frame format used to construct the camera matrix
opengl_to_frame = jnp.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


def to_homogenous_coords(coords):
    """
    Convert coordinates to homogeneous coordinates by adding an extra dimension
    :param coords: jnp array of shape (n_samples, n_dims)
    :return hom_coords: jnp array of shape (n_samples, n_dims+1)
    """
    hom_coords = jnp.ones((coords.shape[0], coords.shape[1] + 1))
    hom_coords = hom_coords.at[:, : coords.shape[1]].set(coords)
    return hom_coords


def generate_uv(img_shape):
    """
    Generates a map of u,v coordinates for a given image
    :param img_shape: shape of the image for which to generate the uv map
    :returns u: a matrix of img_shape with the u-values
    :returns v: a matrix of img_shape with the v-values
    """
    u, v = jnp.meshgrid(jnp.arange(img_shape[1]), jnp.arange(img_shape[0]))
    return u, v


def construct_world_to_image_matrix(
    camera_to_world, camera_to_image, from_opengl=True
):
    """
    Returns the camera matrix, i.e. the world_to_image transform
    :param camera_to_world : camera extrinsic matrix (i.e. the camera pose in world frame)
    :param camera_to_image : camera intrinsic matrix (i.e. the projection of points to image frame)
    :param from_opengl: whether the viewpose is represented in OpenGL frame (y up, z backward)
    :returns world_to_image: camera matrix
    """
    if from_opengl:
        # OpenGL to standard projection form
        camera_to_world = jnp.dot(camera_to_world, opengl_to_frame)
    world_to_camera = jnp.linalg.inv(camera_to_world)
    # Projection Matrix
    world_to_image = jnp.dot(camera_to_image, world_to_camera)
    return world_to_image


def construct_image_to_world_matrix(
    camera_to_world, camera_to_image, from_opengl=True
):
    """
    Returns the camera matrix, i.e. the world_to_image transform
    :param camera_to_world : camera extrinsic matrix (i.e. the camera pose in world frame)
    :param camera_to_image : camera intrinsic matrix (i.e. the projection of points to image frame)
    :param from_opengl: whether the viewpose is represented in OpenGL frame (y up, z backward)
    :returns image_to_world: camera matrix
    """
    if from_opengl:
        # OpenGL to standard projection form
        camera_to_world = jnp.dot(camera_to_world, opengl_to_frame)

    image_to_camera = jnp.linalg.inv(camera_to_image)
    image_to_world = jnp.dot(camera_to_world, image_to_camera)
    return image_to_world


def transform_uvd_to_points(
    rgb,
    depth,
    camera_to_world,
    camera_to_image,
    from_opengl=True,
    filter_zero=True,
):
    """
    :param rgb: RGB values of the projected image
    :param d: depth values of the projected image
    :param camera_to_world : camera extrinsic matrix (i.e. the camera pose in world frame)
    :param camera_to_image : camera intrinsic matrix (i.e. the projection of points to image frame)
    :param from_opengl: whether the viewpose is represented in OpenGL frame (y up, z backward)
    :param filter_zero: whether to remove the viewpoints where zero depth is measured
    :returns cloud: coordinates (xyz) in allocentric/global reference frame (n, 3)
    :returns color: corresponding color (rgb) values (n, 3)
    """
    image_to_world = construct_image_to_world_matrix(
        camera_to_world, camera_to_image, from_opengl=from_opengl
    )

    tf = jax.vmap(partial(jnp.dot, image_to_world))

    # d * [u,v,1]^T = K @ [RT] @ [x,y,z,1]^T
    uv = generate_uv(depth.shape[:2])
    uvd = jnp.ones((depth.shape[0], depth.shape[1], 4))
    uvd = uvd.at[..., 0].set(depth * uv[0])
    uvd = uvd.at[..., 1].set(depth * uv[1])
    uvd = uvd.at[..., 2].set(depth)

    # Only return the [x,y,z] components
    cloud = tf(uvd.reshape(-1, 4))[:, :3]
    normalizer = 255.0 if rgb.max() > 1 else 1
    rgb = rgb.reshape(-1, 3) / normalizer
    # cloud, rgb = jax.lax.cond(
    #     filter_zero,
    #     lambda c, r, d: (c[d], r[d]),
    #     lambda c, r, d: (c, r),
    #     cloud,
    #     rgb,
    #     depth.nonzero(),
    # )
    if filter_zero:
        cloud = cloud[depth.reshape(-1) > 0]
        rgb = rgb[depth.reshape(-1) > 0]

    return cloud, rgb


def transform_points_to_uvd(
    points, camera_to_world, camera_to_image, from_opengl=True
):
    """
    :param points: xyz values for a point cloud
    :param camera_to_world : camera extrinsic matrix (i.e. the camera pose in world frame)
    :param camera_to_image : camera intrinsic matrix (i.e. the projection of points to image frame)
    :param from_opengl: whether the viewpose is represented in OpenGL frame (y up, z backward)
    :returns uv: A list of pixel coordinates to which each observation would map
    :returns color: corresponding color (rgb) values (n, 3)
    """
    # Projection Matrix
    world_to_image = construct_world_to_image_matrix(
        camera_to_world, camera_to_image, from_opengl
    )

    tf = jax.vmap(partial(jnp.dot, world_to_image))

    points = to_homogenous_coords(points)

    # Only return the [x,y,z] components
    uvd = tf(points)
    uvd = uvd.at[:, 0].set(uvd[:, 0] / uvd[:, 2])
    uvd = uvd.at[:, 1].set(uvd[:, 1] / uvd[:, 2])

    return uvd[:, :3]


def look_at_matrix(from_vec, to_vec, earth_normal=None):
    """
    Computes a homogeneous matrix to look from the from_vec to the
    to_vector, both in batched format (B, 3) or normal format (3,)
    The earth_normal should be a tensor of shape (3,)
    """
    forward = from_vec - to_vec
    forward /= np.linalg.norm(forward)

    if earth_normal is None:
        earth_normal = np.array([0, 0, 1.0])

    earth_normal /= np.linalg.norm(earth_normal)

    right = np.cross(earth_normal, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = forward
    mat[:3, 3] = from_vec

    det = np.linalg.det(mat[:3, :3])
    mat[:3, :3] /= det

    return mat
