"""Random sampler

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import jax
from chex import dataclass

from ..utils import sample_random_pos
from .sampler import DefaultSampler
from .utils import compute_linear_move_matrix


@dataclass
class RandomSampler(DefaultSampler):
    """Random sampler"""

    def build_sample_vertices(self):
        if self.share_roadmap:
            return lambda key, num_samples, instance: jax.jit(
                sample_random_pos, static_argnames={"num_samples"}
            )(key, num_samples, instance.rads[0], instance.obs.sdf)
        else:

            def vmapped_fn(key, num_samples, instance):
                keys = jax.random.split(key, instance.num_agents)
                return jax.jit(
                    jax.vmap(sample_random_pos, in_axes=(0, None, 0, None)),
                    static_argnames={"num_samples"},
                )(keys, num_samples, instance.rads, instance.obs.sdf)

            return vmapped_fn

    def build_check_connectivity(self):
        if self.share_roadmap:
            return lambda vertices, instance: jax.jit(compute_linear_move_matrix)(
                vertices,
                instance.max_speeds[0],
                instance.rads[0],
                instance.obs.sdf,
            )
        else:
            return lambda vertices, instance: jax.jit(
                jax.vmap(compute_linear_move_matrix, in_axes=(0, 0, 0, None))
            )(
                vertices,
                instance.max_speeds,
                instance.rads,
                instance.obs.sdf,
            )
