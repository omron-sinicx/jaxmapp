"""Utilities for roadmaps

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from chex import Array


@partial(jax.jit, static_argnames=("num_max_steps"))
def valid_linear_move(
    src: Array,
    dst: Array,
    max_speed: Array,
    rad: Array,
    sdf: Array,
    num_max_steps: int = 20,
) -> bool:
    """
    Validate if dst is reachable from src in a single timestep

    Args:
        src (Array): source position
        dst (Array): destination
        max_speed (Array): agent's max speed
        rad (Array): agent's radius
        sdf (Array): singed distance function of the map
        num_max_steps (int, optional): number of steps to check from the src. Defaults to 20.

    Returns:
        bool: validity (true = no collisions)
    """

    dist = jnp.linalg.norm(dst - src)
    uv = (dst - src) / (dist + 1e-10)
    pts = (
        src
        + uv * jnp.arange(1, num_max_steps + 1).reshape((num_max_steps, 1)) * rad / 2.0
    )
    map_size = sdf.shape[0]
    pts_int = jnp.minimum((pts * map_size).astype(int), map_size - 1)
    dist_to_obs = jax.vmap(lambda p: sdf[p[0], p[1]] - rad)(pts_int).flatten()
    dist_to_src = jnp.linalg.norm(pts - src, axis=1).flatten()
    dist_to_obs = dist_to_obs * (dist_to_src < dist)
    return (
        (jnp.min(dist_to_obs) >= 0)
        & (dist <= max_speed)
        & (jnp.min(dst) > rad) * (jnp.max(dst) < 1 - rad)
    )


def compute_linear_move_matrix(
    vertices: Array,
    max_speed: float,
    rad: float,
    sdf: Array,
) -> Array:
    """
    Compute edge matrix based on valid_linear_move

    Args:
        vertices (Array): a list of vertices
        max_speed (float): agent's max speed
        rad (float): agent's radius
        sdf (Array): signed distance function of the map

    Returns:
        Array: edge matrix
    """

    edges = jax.vmap(
        jax.vmap(
            valid_linear_move,
            in_axes=(None, 0, None, None, None),
        ),
        in_axes=(0, None, None, None, None),
    )(vertices, vertices, max_speed, rad, sdf)

    return edges
