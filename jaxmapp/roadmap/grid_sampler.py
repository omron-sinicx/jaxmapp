"""Grid sampler

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, dataclass

from .sampler import DefaultSampler
from .utils import compute_linear_move_matrix


@partial(jax.jit, static_argnames={"grid_size"})
def generate_grid(grid_size: int, rad: Array, sdf: Array) -> tuple[Array, Array]:
    """
    Compiled function for defining grid

    Args:
        grid_size (int): grid resolution
        rad (Array): agent's radius
        sdf (Array): signed distance function of the map

    Returns:
        tuple[Array, Array]: grid vertices and their validity
    """

    pos = jnp.vstack(
        [
            x.flatten()
            for x in jnp.meshgrid(
                jnp.linspace(0, 1, grid_size),
                jnp.linspace(0, 1, grid_size),
            )
        ]
    ).T
    map_size = sdf.shape[0]
    pos_int = (pos * map_size).astype(int)
    validity = jax.vmap(lambda p: sdf[p[0], p[1]] > rad)(pos_int)
    return pos, validity


@dataclass
class GridSampler(DefaultSampler):
    """Grid sampler"""

    def build_sample_vertices(self):
        if self.share_roadmap:

            def sample_fn(key, num_samples, instance):
                grid_size = int(jnp.sqrt(num_samples))
                pos, validity = generate_grid(
                    grid_size, instance.rads[0], instance.obs.sdf
                )
                pos = pos[validity]
                return pos

            return sample_fn

        else:

            def vmapped_sample_fn(key, num_samples, instance):
                grid_size = int(jnp.sqrt(num_samples))
                pos, validity = jax.vmap(generate_grid, in_axes=(None, 0, None))(
                    grid_size, instance.rads, instance.obs.sdf
                )
                return pos

            return vmapped_sample_fn
