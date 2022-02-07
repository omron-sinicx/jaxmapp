"""Random sampler

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import jax
from chex import Array, dataclass

from ..utils import sample_random_pos
from .sampler import DefaultSampler


@dataclass
class RandomSampler(DefaultSampler):
    """Random sampler"""

    def build_sample_vertices(self):

        if self.share_roadmap:

            def sample_vertices(key, num_samples, instance) -> Array:
                return sample_random_pos(
                    key, num_samples, instance.rads[0], instance.obs.sdf
                )

            return jax.jit(sample_vertices, static_argnames={"num_samples"})
        else:

            def sample_vertices(key, num_samples, instance) -> Array:
                keys = jax.random.split(key, instance.num_agents)
                return jax.jit(
                    jax.vmap(sample_random_pos, in_axes=(0, None, 0, None)),
                    static_argnames={"num_samples"},
                )(keys, num_samples, instance.rads, instance.obs.sdf)

            return sample_vertices
