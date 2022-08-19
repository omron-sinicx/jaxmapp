"""Utilities for sampling vertices

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


@partial(jax.jit, static_argnames=("num_samples", "num_max_trials"))
def sample_random_pos(
    key: PRNGKey,
    num_samples: int,
    rads: Array,
    sdf: Array,
    no_overlap: bool = False,
    num_max_trials: int = 100,
) -> Array:
    """
    Sample a list of vertices from free area

    Args:
        key (PRNGKey): jax.random.PRNGKey
        num_samples (int): number of vertices to sample
        rads (Array): agent's radius
        sdf (Array): signed distance function of the map
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Array: list of random valid positions
    """

    if rads.size == 1:
        rads = jnp.ones(num_samples) * rads
    key0, key1 = jax.random.split(key)
    carried_pos = jax.random.uniform(key0, (num_samples, 2))
    loop_carry = [key1, carried_pos, rads, sdf]
    pos = jax.lax.fori_loop(
        0,
        num_samples,
        partial(
            _sample_random_pos, no_overlap=no_overlap, num_max_trials=num_max_trials
        ),
        loop_carry,
    )[1]
    return pos


def _sample_random_pos(
    i: int, loop_carry: list, no_overlap: bool = False, num_max_trials: int = 100
) -> list:
    """Compiled function of sample_random_pos"""

    def cond(while_carry):
        (
            target_pos,
            target_rad,
            sdf,
            carreid_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry[1:]
        map_size = sdf.shape[0]
        pos_int = jnp.minimum((target_pos * map_size).astype(int), map_size - 1)
        return (
            (
                jnp.any(
                    jnp.linalg.norm(carreid_pos - target_pos, axis=1)
                    - target_rad
                    - rads
                    < 0
                )
                & no_overlap
            )
            | (sdf[pos_int[0], pos_int[1]] - target_rad < 0)
        ) & (num_trials < num_max_trials)

    def body(while_carry):
        (
            key,
            target_pos,
            target_rad,
            sdf,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry
        num_trials = num_trials + 1
        key0, key = jax.random.split(key)
        target_pos = (
            jax.random.uniform(key0, shape=(2,)) * (1 - 2 * target_rad) + target_rad
        )
        return [
            key,
            target_pos,
            target_rad,
            sdf,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ]

    key, carried_pos, rads, sdf = loop_carry
    num_trials = 0
    target_pos = carried_pos[i]
    target_rad = rads[i]

    while_carry = [
        key,
        target_pos,
        target_rad,
        sdf,
        carried_pos,
        rads,
        no_overlap,
        num_trials,
    ]

    while_carry = jax.lax.while_loop(cond, body, while_carry)
    key = while_carry[0]
    pos = while_carry[1]
    num_trials = while_carry[-1]
    pos = jax.lax.cond(
        num_trials < num_max_trials, lambda _: pos, lambda _: pos * jnp.inf, None
    )
    carried_pos = carried_pos.at[i].set(pos)

    return [key, carried_pos, rads, sdf]
