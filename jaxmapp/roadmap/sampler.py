"""Base class for sampler

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import time
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, dataclass

from ..env.instance import Instance
from .timed_roadmap import TimedRoadmap
from .utils import compute_linear_move_matrix, valid_linear_move

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

        if self.timed_roadmap:

            def check_connectivity(vertices: Array, instance: Instance) -> Array:
                def _check_connectivity(vertices_i, agent_id, max_speed, rad, sdf):
                    def inner_loop_body(t, inner_loop_carry):
                        pos_carry_i, instance, i, connectivities = inner_loop_carry
                        pos1 = pos_carry_i[t - 1]
                        pos2 = pos_carry_i[t]
                        connectivity = jax.vmap(
                            jax.vmap(
                                valid_linear_move, in_axes=(None, 0, None, None, None)
                            ),
                            in_axes=(0, None, None, None, None),
                        )(
                            pos1,
                            pos2,
                            max_speed,
                            rad,
                            sdf,
                        )
                        connectivities = connectivities.at[t - 1].set(connectivity)
                        inner_loop_carry = [pos_carry_i, instance, i, connectivities]
                        return inner_loop_carry

                    max_T, num_samples = vertices_i.shape[:2]
                    edges = jnp.zeros((max_T, num_samples, num_samples))
                    loop_carry = [vertices_i, instance, agent_id, edges]
                    loop_carry = jax.lax.fori_loop(
                        1, self.max_T + 1, inner_loop_body, loop_carry
                    )
                    edges = loop_carry[-1]

                    return edges

                return jax.vmap(
                    jax.jit(_check_connectivity), in_axes=(0, 0, 0, 0, None)
                )(
                    vertices,
                    jnp.arange(instance.num_agents),
                    instance.max_speeds,
                    instance.rads,
                    instance.obs.sdf,
                )

            return check_connectivity

        else:

            if self.share_roadmap:

                def check_connectivity(vertices, instance):
                    return compute_linear_move_matrix(
                        vertices,
                        instance.max_speeds[0],
                        instance.rads[0],
                        instance.obs.sdf,
                    )

                return jax.jit(check_connectivity)
            else:

                def check_connectivity(vertices, instance):
                    return jax.vmap(
                        compute_linear_move_matrix, in_axes=(0, 0, 0, None)
                    )(
                        vertices,
                        instance.max_speeds,
                        instance.rads,
                        instance.obs.sdf,
                    )

                return jax.jit(check_connectivity)

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
