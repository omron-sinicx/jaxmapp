"""Timed roadmaps

Author: Keisuke Okumura, Ryo Yonetani

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

from .utils import valid_linear_move


class TimedVertex(NamedTuple):
    """vertex (space-time pair)"""

    t: int  # time
    index: int  # index within the timed roadmap
    pos: Array  # position

    def __eq__(self, other) -> bool:
        if type(other) is not TimedVertex:
            return False
        return (
            self.t == other.t
            and self.index == other.index
            and jnp.array_equal(self.pos, other.pos)
        )

    def __getstate__(self):
        """called when pickling, reducing volume"""
        state = (self.t, self.index, self.pos)
        return state

    def __setstate__(self, state):
        """called when un-pickling"""
        self._replace(t=state[0], index=state[1], pos=state[2])


class TimedRoadmap:
    """timed roadmaps

    Attributes:
        V (list[list[TimedVertex]]): vertices [t][i]
        E (list[list[list[int]]]): adj_list [t][i] -> index list of t+1
        dim (int): dimension of the instance
    """

    def __init__(self, loc_start: Array):
        self.V: list[list[TimedVertex]] = [[TimedVertex(t=0, index=0, pos=loc_start)]]
        self.E: list[list[list[int]]] = [[[]]]
        self.dim = loc_start.shape[0]

    def extend_layer(self, t: int) -> None:
        """extend timed roadmap until a certain timestep

        Args:
            t (int): timestep
        """
        while t > len(self.V) - 1:
            self.V.append([])
            self.E.append([])

    def generate_from_nontimed_roadmap(
        self,
        vertices: Array,
        edges: Array,
        start: Array,
        goal: Array,
        max_speed: Array,
        rad: Array,
        sdf: Array,
        max_T: int,
    ) -> None:
        """
        Generate timed roadmap from non-timed vertices and edges

        Args:
            vertices (Array; (num_samples, 2)): a list of vertices
            edges (Array; (num_samples, num_samples)): edge matrix
            start (Array): start position
            goal (Array): goal position
            max_speed (Array): agent's max speed
            rad (Array): agent's radius
            sdf (Array): signed distance function of the map
            max_T (int): maximum number of time steps
        """

        edges_s = jax.vmap(
            valid_linear_move,
            in_axes=(0, None, None, None, None),
        )(vertices, start, max_speed, rad, sdf)
        edges_g = jax.vmap(
            valid_linear_move,
            in_axes=(0, None, None, None, None),
        )(vertices, goal, max_speed, rad, sdf)
        edges = jnp.vstack(
            (jnp.hstack((edges, edges_g.reshape(-1, 1))), jnp.hstack((edges_g, True)))
        )
        edges = [x.nonzero()[0].tolist() for x in edges]

        vertices = jnp.vstack((vertices, goal))

        self.E[0][0] = np.array(edges_s.nonzero()[0]).tolist()

        self.extend_layer(max_T)
        for t in range(1, max_T + 1):
            self.V[t] = [
                TimedVertex(t=t, index=i, pos=np.array(p, np.float64))
                for i, p in enumerate(vertices)
            ]
            self.E[t] = edges
        self.E[max_T] = [[] for _ in range(len(vertices))]

    def generate_from_timed_roadmap(
        self,
        vertices: Array,
        edges: Array,
        start: Array,
        goal: Array,
        max_speed: Array,
        rad: Array,
        sdf: Array,
        max_T: int,
    ) -> None:
        """
        Generate timed roadmap from timed vertices and edges

        Args:
            vertices (Array; (max_T, num_samples, 2)): list of vertices stacked over time
            edges (Array; (max_T, num_samples, num_samples)): edge matrix stacked over time
            start (Array): start position
            goal (Array): goal position
            max_speed (Array): agent's max speed
            rad (Array): agent's radius
            sdf (Array): signed distance function of the map
            max_T (int): maximum number of time steps
        """

        self.extend_layer(max_T)
        vertices_t = vertices[0]
        edges_s = jax.vmap(valid_linear_move, in_axes=(0, None, None, None, None))(
            vertices_t,
            start,
            max_speed,
            rad,
            sdf,
        )
        self.E[0][0] = np.array(edges_s.nonzero()[0]).tolist()

        for t in range(1, max_T + 1):
            vertices_t = vertices[t - 1]
            edges_t = edges[t - 1]
            vertices_t_next = vertices[t]
            edges_g = jax.vmap(
                valid_linear_move,
                in_axes=(0, None, None, None, None),
            )(vertices_t, goal, max_speed, rad, sdf)
            edges_g_inv = jax.vmap(
                valid_linear_move,
                in_axes=(None, 0, None, None, None),
            )(goal, vertices_t_next, max_speed, rad, sdf)
            edges_t = jnp.vstack(
                (
                    jnp.hstack((edges_t, edges_g.reshape(-1, 1))),
                    jnp.hstack((edges_g_inv, True)),
                )
            )
            edges_t = [x.nonzero()[0].tolist() for x in edges_t]

            vertices_t = jnp.vstack((vertices_t, goal))

            self.V[t] = [
                TimedVertex(t=t, index=i, pos=np.array(p, np.float64))
                for i, p in enumerate(vertices_t)
            ]
            self.E[t] = edges_t
        self.E[max_T] = [[] for _ in range(len(vertices))]
