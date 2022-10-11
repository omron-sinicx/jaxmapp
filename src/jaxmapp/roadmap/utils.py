"""Utilities for roadmaps

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import Array


@jax.jit
def valid_linear_move(
    src: Array,
    dst: Array,
    max_speed: Array,
    rad: Array,
    sdf: Array,
) -> bool:
    """
    Validate if dst is reachable from src in a single timestep

    Args:
        src (Array): source position
        dst (Array): destination
        max_speed (Array): agent's max speed
        rad (Array): agent's radius
        sdf (Array): singed distance function of the map

    Returns:
        bool: validity (true = no collisions)
    """

    map_size = sdf.shape[0]
    dist = jnp.linalg.norm(dst - src)
    uv = (dst - src) / (dist + 1e-10)

    def cond(pts):
        p = jnp.minimum((pts * map_size).astype(int), map_size - 1)
        return (
            (dist <= max_speed)  # each move should be completed with a single step
            & (
                jnp.linalg.norm(pts - src) < dist
            )  # pts should be an intermediate point between src and dst
            & (sdf[p[0], p[1]] > rad)  # agent does not collide with obstacles at pts
        )

    def body(pts):
        return pts + uv * rad

    pts = jax.lax.while_loop(cond, body, src)
    return jnp.linalg.norm(pts - src) >= dist


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
