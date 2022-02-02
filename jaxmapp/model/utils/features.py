"""Utilities for feature extraction

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from chex import Array


@jax.jit
def get_normed_vec_mag(arr_vec: Array) -> Array:
    """Compute
       from [[x1, y1], [x2, y2], ...]
       to   [[normed_x1, normed_y1, mag], [normed_x2, normed_y2, mag], ...]

    Args:
        arr_vec (Array): un-normalized vector (2D)

    Returns:
         Array: normalized vector (3D)
    """
    vec_mag = jnp.sqrt(jnp.sum(arr_vec ** 2, axis=1)).reshape(-1, 1)
    vec_mag_avoid_zero = jnp.where(vec_mag == 0, 1, vec_mag)
    arr_vec = arr_vec / vec_mag_avoid_zero
    return jnp.hstack((arr_vec, vec_mag))


@jax.jit
def get_arr_others_info(
    arr_current_locs: Array,
    goals: Array,
    arr_prev_locs: Array,
    max_speeds: Array,
    rads: Array,
) -> Array:
    """
    Get locational features of all agents

    Args:
        arr_current_locs (Array): array concatenating the current locations of all agents
        goals (Array): array concatenating the goal locations of all agents
        arr_prev_locs (Array): array concatenating the previous locations of all agents
        max_speeds (Array): maximum speeds of agents
        rads (Array): size of agents

    Returns:
        Array: localtional features of all agents
    """
    ref = jnp.hstack((arr_current_locs, goals, arr_prev_locs)).reshape(-1, 2)
    arr_relative_locs = jax.vmap(lambda x: ref - x)(arr_current_locs)
    num_agents = len(arr_current_locs)
    arr_relative_locs = get_normed_vec_mag(arr_relative_locs.reshape(-1, 2)).reshape(
        num_agents * num_agents, -1
    )

    arr_others_info = jnp.hstack(
        (
            arr_relative_locs,
            jnp.repeat(rads, num_agents).reshape(-1, 1),
            jnp.repeat(max_speeds, num_agents).reshape(-1, 1),
        )
    )
    return arr_others_info


@partial(jax.jit, static_argnums=(1,))
def get_self_info(arr_others_info: Array, num_agents: int) -> Array:
    """Extract self information from the result of get_arr_others_info"""

    self_info = jax.vmap(
        lambda i: arr_others_info.reshape(num_agents, num_agents, -1)[i, i, :]
    )(jnp.arange(num_agents))
    return self_info[..., 3:]


@partial(jax.jit, static_argnums=(3, 4))
def get_fov_features(
    current_pos: Array,
    occupancy_map: Array,
    cost_to_go_map: Array,
    fov_size: int,
    flatten: bool,
) -> Array:
    """
    Generate FOV-related features

    Args:
        current_pos (Array): Current position of an agent
        occupancy_map (Array): occupancy map
        cost_to_go_map (Array): cost-to-go map
        fov_size (int): size of FOV from which features are extracted
        flatten (bool): whether to flatten the output matrix

    Returns:
        Array: FOV-related features
    """
    map_size = occupancy_map.shape[0]
    map_size = occupancy_map.shape[0]
    buf = int((fov_size - 1) / 2)

    pos_y = jax.lax.cond(
        current_pos[0] < 1,
        lambda _: (current_pos[0] * map_size).astype(int),
        lambda _: map_size - 1,
        None,
    )
    pos_x = jax.lax.cond(
        current_pos[1] < 1,
        lambda _: (current_pos[1] * map_size).astype(int),
        lambda _: map_size - 1,
        None,
    )

    omap_padded = jnp.ones((map_size + buf * 2, map_size + buf * 2))
    omap_padded = 1 - jax.lax.dynamic_update_slice(
        omap_padded, occupancy_map, [buf, buf]
    )
    omap_fov = jax.lax.dynamic_slice(omap_padded, (pos_y, pos_x), (fov_size, fov_size))

    cmap_padded = jnp.ones((map_size + buf * 2, map_size + buf * 2)) * jnp.inf
    cmap_padded = jax.lax.dynamic_update_slice(cmap_padded, cost_to_go_map, [buf, buf])
    cmap_fov = jax.lax.dynamic_slice(cmap_padded, (pos_y, pos_x), (fov_size, fov_size))
    cmap_fov = (cmap_fov < cmap_fov[buf, buf]).astype(float)

    fov_feature = jnp.stack((omap_fov, cmap_fov), 0)
    if flatten:
        fov_feature = fov_feature.flatten()

    return fov_feature
