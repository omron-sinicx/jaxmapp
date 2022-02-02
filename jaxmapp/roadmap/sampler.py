"""Base class for sampler

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import time
from logging import getLogger

import numpy as np
from chex import Array, PRNGKey, dataclass

from ..env.instance import Instance
from .timed_roadmap import TimedRoadmap

logger = getLogger(__name__)


@dataclass
class DefaultSampler:
    """Default class for sampler"""

    num_samples: int
    max_T: int
    share_roadmap: bool
    timed_roadmap: bool = False
    verbose: bool = False

    def __post_init__(self):
        self.sample_vertices = self.build_sample_vertices()
        self.check_connectivity = self.build_check_connectivity()
        self.construct_trms = self.build_construct_trms()

    def build_sample_vertices(self):
        """
        Returns a callable function `sample_vertices` that will be used to sample valid vertices as follows.
        ```python
        vertices = sampler.sample_vertices(key, num_samples, instance)
        ```

        Note:
            The output of `sample_vertices`, should be a collection of vertices with its shape given as follows.
            - If `share_roadmap == True`, output shape should be `(num_samples, 2)`
            - If `share_roadmap == False`, output shape should be `(num_agents, num_samples, 2)` which stacks vertices adaptively sampled for each agent
            - If `timed_roadmap == True`, output shape should be `(num_agents, max_T, num_samples, 2)` which gives time-varying vertices for max_T steps for each agent

        """

        def sample_vertices(
            key: PRNGKey, num_samples: int, instance: Instance
        ) -> Array:
            """
            Sample valid vertices

            Args:
                key (PRNGKey): jax.random.PRNGKey
                num_samples (int): number of vertices
                instance (Instance): problem instance

            Returns:
                Array: list of valid vertices stacked over agents
            """
            raise NotImplementedError()

        return sample_vertices

    def build_check_connectivity(self):
        """
        Returns a callable function `check_connectivity` that will be used to check reachability between vertices as follows.
        ```python
        edges = sampler.check_connectivity(vertices, instance)
        ```

        Note:
            The output shape of `check_connectivity`, is expected as follows.
            - If `share_roadmap == True`, output shape should be `(num_samples, num_sample)`
            - If `share_roadmap == False`, output shape should be `(num_agents, num_samples, num_samples)`
            - If `timed_roadmap == True`, output shape should be `(num_agents, max_T, num_samples, num_samples)` where `edges[:, t, :, :]` describes the connectivity of `vertices[:, t]` to `vertices[:, t + 1]`.

        """

        def check_connectivity(vertices: Array, instance: Instance) -> Array:
            """
            Compute edges between vertices

            Args:
                vertices (Array): a list of vertices
                instance (Instance): problem instance

            Returns:
                Array: edge matrices stacked over agents
            """
            raise NotImplementedError()

        return check_connectivity

    def build_construct_trms(self):
        """
        Returns a callable function `construct_trms` that will be used to construct TimedRoadmaps as follows.
        ```python
        trms = sampler.construct_trms(key, instance)
        ```
        """

        def construct_trms(key: PRNGKey, instance: Instance) -> list[TimedRoadmap]:
            """
            Construct timed roadmaps

            Args:
                key (PRNGKey): jax.random.PRNGKey
                ins (Instance): problem instance

            Returns:
                list[TimedRoadmap]: list of timed roadmaps
            """

            t_start = time.time()
            num_samples = self.num_samples
            max_T = self.max_T

            vertices = self.sample_vertices(key, num_samples, instance)
            edges = self.check_connectivity(vertices, instance)

            trms = []

            for i in range(instance.num_agents):
                trm = TimedRoadmap(np.array(instance.starts[i], np.float64))
                if self.timed_roadmap:
                    generate_fn = trm.generate_from_timed_roadmap
                    vertices_i = vertices[i]
                    edges_i = edges[i]
                elif self.share_roadmap:
                    generate_fn = trm.generate_from_nontimed_roadmap
                    vertices_i = vertices
                    edges_i = edges
                else:
                    generate_fn = trm.generate_from_nontimed_roadmap
                    vertices_i = vertices[i]
                    edges_i = edges[i]

                generate_fn(
                    vertices_i,
                    edges_i,
                    instance.starts[i],
                    instance.goals[i],
                    instance.max_speeds[i],
                    instance.rads[i],
                    instance.obs.sdf,
                    max_T,
                )
                trms.append(trm)

            elapsed = time.time() - t_start
            if self.verbose == 1:
                logger.info(f"Built roadmap, {elapsed} sec")
            return trms

        return construct_trms
