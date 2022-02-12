"""Baseclass for multi-agent path planner

Author: Keisuke Okumura

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

import heapq
import time
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from logging import getLogger
from typing import Optional

import numpy as np
import timeout_decorator
from check_continuous_collision import check_continuous_collision

from ..env import Instance
from ..roadmap import TimedRoadmap, TimedVertex
from .result import Result
from .utils import get_cost, get_travel_dist

logger = getLogger(__name__)


def continuous_collide_spheres(
    from1: np.ndarray,
    to1: np.ndarray,
    rad1: float,
    from2: np.ndarray,
    to2: np.ndarray,
    rad2: float,
) -> bool:
    """Detect collision between two dynamic spheres

    Args:
        from1 (np.ndarray): 'from' position of sphere 1
        to1 (np.ndarray): 'to' position of sphere 1
        rad1 (float): radius of sphere 1
        from2 (np.ndarray): 'from' position of sphere 2
        to2 (np.ndarray): 'to' position of sphere 2
        rad2 (float): radius of sphere 2

    Returns:
        bool: true -> collide
    """
    # use cython
    return check_continuous_collision(
        from1[0],
        from1[1],
        to1[0],
        to1[1],
        rad1,
        from2[0],
        from2[1],
        to2[0],
        to2[1],
        rad2,
    )


class Planner(metaclass=ABCMeta):
    """Template of multi-agent path planner

    Attributes:
        ins (Instance): instance
        trms (list[TimedRoadmap]): timed roadmaps (assumed to be consistent)
        verbose (int): print additional info
        solved (bool): whether to be solved
        solution (list[list[TimedVertex]]): paths
        elapsed (float): planning time
        result (Result): planning result
        time_limit (float): time limit of planning
        agent_collision (bool): true -> neglect inter-agent collision
        cnt_static_collide (int): number of vertex collision check
        cnt_continuous_collide (int): number of edge collision check
        lowlevel_expanded (int): number of expanded vertices
        lowlevel_explored (int): number of explored vertices
    """

    def __init__(
        self,
        verbose: int = 0,
        **kwargs,
    ):
        self.verbose: int = verbose
        self.time_limit: float = (
            30 if "time_limit" not in kwargs.keys() else kwargs["time_limit"]
        )
        self.agent_collision: bool = (
            True
            if "agent_collision" not in kwargs.keys()
            else kwargs["agent_collision"]
        )
        self.init_status()

    def init_status(self):
        self.solved: bool = False
        self.solution: list[list[TimedVertex]] = []
        self.elapsed: float = 0
        self.result: Optional[Result] = None
        # to measure the effort of collision check
        self.cnt_static_collide: int = 0
        self.cnt_continuous_collide: int = 0

        # search effort
        self.lowlevel_expanded: int = 0  # expanded node
        self.lowlevel_explored: int = 0  # explored node

    def info(self, msg: str):
        """print useful info"""
        if self.verbose > 0:
            logger.info(msg)

    def debug(self, msg: str):
        """print useful info"""
        if self.verbose > 0:
            logger.debug(msg)

    def solve(self, ins: Instance, trms: list[TimedRoadmap]) -> Result:
        """main function"""
        self.info(f"_solve is started, time_limit: {self.time_limit} sec")
        self.init_status()
        self.ins = ins
        self.trms = trms

        @timeout_decorator.timeout(self.time_limit)
        def solve_until_timeout() -> None:
            self._solve()

        # solve
        t_start = time.time()
        try:
            solve_until_timeout()
        except timeout_decorator.TimeoutError:
            print("timeout")
        self.elapsed = time.time() - t_start
        self.info(f"_solve is done, {self.elapsed} sec")

        # validate solution
        if self.validate() != self.solved:
            logger.error("inconsistent planner")

        # set result
        if self.solved:
            return Result(
                solved=self.solved,
                num_agents=ins.num_agents,
                paths=self.solution,
                name_planner=self.get_name(),
                elapsed_planner=self.elapsed,
                sum_of_costs=self.get_sum_of_costs(self.solution),
                maximum_costs=self.get_maximum_costs(self.solution),
                sum_of_travel_dists=self.get_sum_of_travel_dists(self.solution),
                maximum_travel_dists=self.get_maximum_travel_dists(self.solution),
                cnt_static_collide=self.cnt_static_collide,
                cnt_continuous_collide=self.cnt_continuous_collide,
                lowlevel_expanded=self.lowlevel_expanded,
                lowlevel_explored=self.lowlevel_explored,
            )
        else:
            return Result(
                solved=self.solved,
                num_agents=ins.num_agents,
                name_planner=self.get_name(),
                elapsed_planner=self.elapsed,
                cnt_static_collide=self.cnt_static_collide,
                cnt_continuous_collide=self.cnt_continuous_collide,
                lowlevel_expanded=self.lowlevel_expanded,
                lowlevel_explored=self.lowlevel_explored,
            )

    @abstractmethod
    def _solve(self) -> None:
        """child class should complement this part"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """solver name"""
        pass

    def validate(self) -> bool:
        """validate obtained solution

        Returns:
            bool: true -> valid, false -> invalid
        """
        if not self.solved:
            return False
        # check starts
        if not all(
            [
                all(path[0].pos == self.ins.starts[i])
                for i, path in enumerate(self.solution)
            ]
        ):
            logger.error("start location is invalid")
            return False
        # check goals
        if not all(
            [
                np.linalg.norm(path[-1].pos - self.ins.goals[i])
                <= self.ins.goal_rads[i]
                for i, path in enumerate(self.solution)
            ]
        ):
            logger.error("goal location is invalid")
            return False

        T = self.solution[0][-1].t
        for t in range(1, T + 1):
            for i in range(self.ins.num_agents):
                loc_i_t = self.solution[i][t].pos
                loc_i_t_prev = self.solution[i][t - 1].pos
                rad_i = self.ins.rads[i]
                # check continuity
                if np.linalg.norm(loc_i_t - loc_i_t_prev) > self.ins.max_speeds[i]:
                    logger.error(
                        f"invalid move at t={t-1}:" f"{loc_i_t_prev} -> {loc_i_t}"
                    )
                    return False
                # check collisions
                for j in range(i + 1, self.ins.num_agents):
                    if self.collide_dynamic_agents(
                        loc_i_t_prev,
                        loc_i_t,
                        rad_i,
                        self.solution[j][t - 1].pos,
                        self.solution[j][t].pos,
                        self.ins.rads[j],
                    ):
                        logger.error("conflict")
                        return False
        return True

    def collide_dynamic_agents(
        self,
        from1: np.ndarray,
        to1: np.ndarray,
        rad1: float,
        from2: np.ndarray,
        to2: np.ndarray,
        rad2: float,
    ) -> bool:
        """detect collision between two moving agents

        Args:
            from1 (np.ndarray): 'from' position of agent 1
            to1 (np.ndarray): 'to' position of agent 1
            rad1 (float): radius of agent 1
            from2 (np.ndarray): 'from' position of agent 2
            to2 (np.ndarray): 'to' position of agent 2
            rad2 (float): radius of agent 2

        Returns:
            bool: true -> collide
        """

        res = self.__collide_dynamic_agents(
            from1,
            to1,
            rad1,
            from2,
            to2,
            rad2,
        )
        self.cnt_continuous_collide += 1
        return res

    def __collide_dynamic_agents(
        self,
        from1: np.ndarray,
        to1: np.ndarray,
        rad1: float,
        from2: np.ndarray,
        to2: np.ndarray,
        rad2: float,
    ) -> bool:
        """private function, detect collision between two moving agents

        Args:
            from1 (np.ndarray): 'from' position of agent 1
            to1 (np.ndarray): 'to' position of agent 1
            rad1 (float): radius of agent 1
            from2 (np.ndarray): 'from' position of agent 2
            to2 (np.ndarray): 'to' position of agent 2
            rad2 (float): radius of agent 2

        Returns:
            bool: true -> collide
        """
        if not self.agent_collision:
            return False

        # accurate check
        return continuous_collide_spheres(
            from1,
            to1,
            rad1,
            from2,
            to2,
            rad2,
        )

    def collide_static_agents(
        self, pos1: np.ndarray, size1: float, pos2: np.ndarray, size2: float
    ) -> bool:
        """detect collision between two static agents

        Args:
            pos1 (np.ndarray): 'from' position of agent 1
            size1 (float): radius of agent 1
            pos2 (np.ndarray): 'from' position of agent 2
            size2 (float): radius of agent 2

        Returns:
            bool: true -> collide
        """

        res = self.__collide_static_agents(pos1, size1, pos2, size2)
        self.cnt_static_collide += 1
        return res

    def __collide_static_agents(
        self, pos1: np.ndarray, size1: float, pos2: np.ndarray, size2: float
    ) -> bool:
        """private func, detect collision between two static agents (sphere)

        Args:
            pos1 (np.ndarray): 'from' position of agent 1
            size1 (float): radius of agent 1
            pos2 (np.ndarray): 'from' position of agent 2
            size2 (float): radius of agent 2

        Returns:
            bool: true -> collide
        """
        if not self.agent_collision:
            return False
        # return collide_spheres(pos1, size1, pos2, size2)
        # fast check
        return sum((pos1 - pos2) ** 2) <= (size1 + size2) ** 2

    def get_single_agent_path(
        self,
        agent: int,
        get_f_value: Callable[[TimedVertex], float],
        check_fin: Callable[[TimedVertex], bool],
        insert: Callable[[TimedVertex, list[list]], None],
        valid_successor: Callable[[TimedVertex, TimedVertex], bool],
        **kwargs,
    ) -> Optional[list[TimedVertex]]:
        """template of single-agent path finding (A*-like)

        See also prioritized_planning.py

        Args:
        agent (int): target agent
        get_f_value (Callable[[TimedVertex], float]): f-value function
        check_fin (Callable[[TimedVertex], bool]): goal checker
        insert (Callable[[TimedVertex, list[list]], None]):
            insert expanded node to OPEN list
        valid_successor: Callable[[TimedVertex, TimedVertex], bool]:
            check whether the successor is valid
        **kwargs:

        Returns:
            Optional[list[TimedVertex]]: path or None
        """
        # A* search
        trm = self.trms[agent]
        T = len(trm.V) - 1  # makespan

        # closed list
        CLOSE = [[False] * len(trm.V[t]) for t in range(T + 1)]
        # store parent index
        PARENT = [[None] * len(trm.V[t]) for t in range(T + 1)]
        # open list
        OPEN: list[list] = []

        # insert initial node
        insert(trm.V[0][0], OPEN)
        self.lowlevel_expanded += 1

        # main loop
        while len(OPEN) > 0:
            # pop
            v = heapq.heappop(OPEN)[-1]
            self.lowlevel_explored += 1

            # check goal condition
            if check_fin(v):
                # backtrack
                path = []
                while v.t > 0:
                    path.append(v)
                    v = trm.V[v.t - 1][PARENT[v.t][v.index]]
                path.append(v)  # start
                path.reverse()
                return path

            # expand
            if v.t == T:  # no successors
                continue
            for j in trm.E[v.t][v.index]:
                u = trm.V[v.t + 1][j]
                if CLOSE[u.t][u.index]:
                    continue
                # check collision
                if not valid_successor(v, u):
                    continue
                # update CLOSE, this is special for timed roadmap
                CLOSE[u.t][u.index] = True
                # update parent
                PARENT[u.t][u.index] = v.index
                insert(u, OPEN)
                self.lowlevel_expanded += 1

        return None

    def get_costs_metric(self, paths: list[list[TimedVertex]], fn: Callable) -> float:
        """compute cost metric

        Args:
            paths (list[list[TimedVertex]]): paths
            fn (Callable): e.g., max, sum

        Returns:
            float: cost
        """
        return fn(
            [
                get_cost(
                    path,
                    self.ins.goals[i],
                    self.ins.goal_rads[i],
                )
                for i, path in enumerate(paths)
            ]
        )

    def get_sum_of_costs(self, paths: list[list[TimedVertex]]) -> float:
        """sum-of-costs"""
        return self.get_costs_metric(paths, sum)

    def get_maximum_costs(self, paths: list[list[TimedVertex]]) -> float:
        """aka. makespan"""
        return self.get_costs_metric(paths, max)

    def get_travel_dists_metric(self, paths: list[list[TimedVertex]], fn: Callable):
        """compute distance metric

        Args:
            paths (list[list[TimedVertex]]): paths
            fn (Callable): e.g., max, sum

        Returns:
            float: cost
        """
        return fn([get_travel_dist(path) for i, path in enumerate(paths)])

    def get_sum_of_travel_dists(self, paths: list[list[TimedVertex]]) -> float:
        """sum of travel distances"""
        return self.get_travel_dists_metric(paths, sum)

    def get_maximum_travel_dists(self, paths: list[list[TimedVertex]]) -> float:
        """maximum travel distances"""
        return self.get_travel_dists_metric(paths, max)

    def print_solution(self) -> None:
        """print solution"""
        if not self.solved:
            print("failed to solve")
            return
        for i, path in enumerate(self.solution):
            print(
                f"agent={i}, start={self.ins.starts[i]}, "
                f"goal={self.ins.goals[i]}, "
                f"rad={self.ins.rads[i]}, "
                f"max_speed={self.ins.max_speeds[i]}, "
                f"goal_rad={self.ins.goal_rads[i]}"
            )
            for v in path:
                print(v)
            print()
