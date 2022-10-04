"""Definition of problem instance

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, dataclass
from CMap2D import CMap2D

from ..utils import sample_random_pos
from .obstacle import ObstacleMap, ObstacleSphere


@dataclass
class Instance:
    """Problem instance of multi-agent path planning"""

    num_agents: int
    starts: Array
    goals: Array
    max_speeds: Array
    rads: Array
    goal_rads: Array
    obs: ObstacleMap

    def calc_cost_to_go_maps(self) -> np.ndarray:
        """
        Calculate cost-to-go maps via dijkstra search

        Returns:
            np.ndarray: a stack of cost-to-go maps
        """

        map_size = self.obs.get_size()
        cmap2d = CMap2D()
        cmap2d.from_array(np.array(self.obs.occupancy), (0, 0), 1.0 / map_size)
        cost_to_go_maps = []
        for goal, rad in zip(self.goals, self.rads):
            rad_int = math.ceil(rad * map_size)
            cost_to_go = cmap2d.dijkstra(
                np.array(goal * map_size, np.int64), connectedness=4
            )
            cost_to_go[cmap2d.as_sdf() < rad] = np.inf
            cost_to_go[:rad_int] = np.inf
            cost_to_go[:, :rad_int] = np.inf
            cost_to_go[-rad_int:] = np.inf
            cost_to_go[:, -rad_int:] = np.inf
            cost_to_go_maps.append(cost_to_go)
        cost_to_go_maps = np.stack(cost_to_go_maps)

        return cost_to_go_maps

    def to_numpy(self) -> Instance:
        """
        Convert instance to numpy format, which should be called when feeding the instance to planner.

        Returns:
            Instance: instance with all attributes converted to numpy array
        """

        starts = [np.array(x, np.float64) for x in self.starts]
        goals = [np.array(x, np.float64) for x in self.goals]
        max_speeds = np.array(self.max_speeds, np.float64)
        rads = np.array(self.rads, np.float64)
        goal_rads = np.array(self.goal_rads, np.float64)
        obs = ObstacleMap(np.array(self.obs.occupancy), np.array(self.obs.sdf))
        return Instance(
            num_agents=self.num_agents,
            starts=starts,
            goals=goals,
            max_speeds=max_speeds,
            rads=rads,
            goal_rads=goal_rads,
            obs=obs,
        )

    def to_jnumpy(self) -> Instance:
        """
        Convert instance to numpy format, which is necessary when feeding the instance to ML models.

        Returns:
            Instance: instance with all attributes converted to jax.numpy array
        """

        starts = jnp.array(self.starts)
        goals = jnp.array(self.goals)
        max_speeds = jnp.array(self.max_speeds)
        rads = jnp.array(self.rads)
        goal_rads = jnp.array(self.goal_rads)
        obs = ObstacleMap(jnp.array(self.obs.occupancy), jnp.array(self.obs.sdf))

        return Instance(
            num_agents=self.num_agents,
            starts=starts,
            goals=goals,
            max_speeds=max_speeds,
            rads=rads,
            goal_rads=goal_rads,
            obs=obs,
        )


@dataclass
class InstanceGenerator:
    """Default class for instance generators"""

    num_agents_min: int
    num_agents_max: int
    max_speeds_cands: list[float]
    rads_cands: list[float]

    def __post_init__(self):

        if not isinstance(self.max_speeds_cands, list):
            self.max_speeds_cands = [i for i in self.max_speeds_cands]
        if not isinstance(self.rads_cands, list):
            self.rads_cands = [i for i in self.rads_cands]

        self._generate = self.build_compiled_generate_fn()

    def build_compiled_generate_fn(self):
        def _generate_ins_with_num_agents_max(
            key: PRNGKey,
            num_agents_max: int,
            max_speeds_cands: Array,
            rads_cands: Array,
            obs: ObstacleMap,
        ) -> Instance:
            """
            Generate instance of [num_agents_max] agents

            Args:
                key (PRNGKey): jax.random.PRNGKey
                num_agents_max (int): number of (maximum) agents
                max_speeds_cands (Array): a list of candidates for agent's max speeds
                rads_cands (Array): a list of candidates for agent's radius
                obs (ObstacleMap): obstacle map

            Returns:
                Instance: MAPP instance of [num_agents_max] agents
            """

            # set radius and max_speeds
            rads_cands = jnp.array(rads_cands)
            max_speeds_cands = jnp.array(max_speeds_cands)

            key_rads, key_speeds, key = jax.random.split(key, 3)
            choices = jax.random.choice(
                key_rads, len(rads_cands), shape=(num_agents_max,)
            )
            rads = rads_cands[choices]
            choices = jax.random.choice(
                key_speeds, len(max_speeds_cands), shape=(num_agents_max,)
            )
            max_speeds = max_speeds_cands[choices]

            key_start_goal, key = jax.random.split(key)
            starts_goals = sample_random_pos(
                key_start_goal,
                num_agents_max * 2,
                jnp.tile(rads, 2),  # [rad_a, rad_b] -> [rad_a, rad_b, rad_a, rad_b]
                obs.sdf,
                no_overlap=True,
                num_max_trials=100,
            )
            starts = starts_goals[:num_agents_max]
            goals = starts_goals[num_agents_max:]

            return Instance(
                num_agents=num_agents_max,
                starts=starts,
                goals=goals,
                max_speeds=max_speeds,
                rads=rads,
                goal_rads=0.01 * jnp.ones(num_agents_max),
                obs=obs,
            )

        def _sample_agents(ins: Instance, num_agents: int) -> Instance:
            """
            Reduce number of agents.
            This was necessary to avoid jit-compile everytime when the number of agents changes.

            Args:
                ins (Instance): Original problem instance
                num_agents (int): number of agents to keep

            Returns:
                Instance: MAPP instance of [num_agents] agents
            """

            n = num_agents
            jls = jax.lax.slice

            return Instance(
                num_agents=n,
                starts=jls(ins.starts, (0, 0), (n, 2)),
                goals=jls(ins.goals, (0, 0), (n, 2)),
                max_speeds=jls(ins.max_speeds, (0,), (n,)),
                rads=jls(ins.rads, (0,), (n,)),
                goal_rads=jls(ins.goal_rads, (0,), (n,)),
                obs=ins.obs,
            )

        def _generate(key: PRNGKey, num_agents: int, obs: ObstacleMap) -> Instance:
            """
            Compiled function for generating MAPP instance for a given obstacle map.

            Args:
                key (PRNGKey): jax.random.PRNGKey
                num_agents (int): number of agents
                obs (ObstacleMap): obstacle map

            Returns:
                Instance: MAPP instance
            """

            ins = jax.jit(
                _generate_ins_with_num_agents_max,
                static_argnames={"num_agents_max"},
            )(
                key,
                self.num_agents_max,
                self.max_speeds_cands,
                self.rads_cands,
                obs,
            )
            ins = jax.jit(_sample_agents, static_argnames={"num_agents"})(
                ins, num_agents
            )

            return ins

        return jax.jit(_generate, static_argnames={"num_agents"})


@dataclass
class InstanceGeneratorCircleObs(InstanceGenerator):
    map_size: int
    num_obs: int
    obs_size_lower_bound: float = 0.05
    obs_size_upper_bound: float = 0.08

    def generate(
        self,
        key: PRNGKey,
    ) -> Instance:
        """
        Generate an instance with some circle obstacles

        Args:
            key (PRNGKey): jax.random.PRNGKey

        Returns:
            Instance: MAPP problem instance with circle obstacles
        """

        if self.num_obs == 0:
            return self.generate_wo_obs(key)

        key0, key = jax.random.split(key)
        num_agents = int(
            jax.random.randint(
                key0, (1,), minval=self.num_agents_min, maxval=self.num_agents_max + 1
            )
        )

        circle_obs = jax.random.uniform(key, shape=(self.num_obs, 3))
        circle_obs = circle_obs.at[:, 2].multiply(
            (self.obs_size_upper_bound - self.obs_size_lower_bound) / 2
        )
        circle_obs = circle_obs.at[:, 2].add(self.obs_size_lower_bound / 2)
        circle_obs = [ObstacleSphere(pos=o[:2], rad=o[2]) for o in circle_obs]
        occupancy = np.dstack([x.draw_2d(self.map_size) for x in circle_obs]).max(-1)
        cmap2d = CMap2D()
        cmap2d.from_array(occupancy, (0, 0), 1.0 / self.map_size)
        sdf = cmap2d.as_sdf()

        obs = ObstacleMap(occupancy, sdf)

        ins = self._generate(key, num_agents, obs)
        assert not (
            jnp.any(jnp.isinf(ins.starts)) | jnp.any(jnp.isinf(ins.goals))
        ), "Invalid problem instance. try different parameters."
        return ins

    def generate_wo_obs(self, key: PRNGKey) -> Instance:
        """
        Generate an instance with no obstacles

        Args:
            key (PRNGKey): jax.random.PRNGKey

        Returns:
            Instance: MAPP problem instance with no obstacles
        """

        key0, key = jax.random.split(key)
        num_agents = int(
            jax.random.randint(
                key0, (1,), minval=self.num_agents_min, maxval=self.num_agents_max + 1
            )
        )

        occupancy = jnp.zeros((self.map_size, self.map_size))
        sdf = jnp.ones((self.map_size, self.map_size))
        obs = ObstacleMap(occupancy, sdf)

        ins = self._generate(key, num_agents, obs)
        assert not (
            jnp.any(jnp.isinf(ins.starts)) | jnp.any(jnp.isinf(ins.goals))
        ), "Invalid problem instance. try different parameters."
        return ins


@dataclass
class InstanceGeneratorImageInput(InstanceGenerator):
    image: np.ndarray

    def generate(self, key: PRNGKey) -> Instance:
        """
        Generate an instance from obstacle image

        Args:
            key (PRNGKey): jax.random.PRNGKey

        Returns:
            Instance: MAPP problem instance
        """

        assert self.image.shape[0] == self.image.shape[1]
        key0, key = jax.random.split(key)
        num_agents = int(
            jax.random.randint(
                key0, (1,), minval=self.num_agents_min, maxval=self.num_agents_max + 1
            )
        )
        map_size = self.image.shape[0]
        occupancy = self.image
        cmap2d = CMap2D()
        cmap2d.from_array(occupancy, (0, 0), 1.0 / map_size)
        sdf = cmap2d.as_sdf()
        obs = ObstacleMap(occupancy, sdf)

        ins = self._generate(key, num_agents, obs)
        assert not (
            jnp.any(jnp.isinf(ins.starts)) | jnp.any(jnp.isinf(ins.goals))
        ), "Invalid problem instance. try different parameters."
        return ins
