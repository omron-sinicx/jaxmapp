"""Implementation of a standard prioritized planning

Author: Keisuke Okumura

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

ref
- Silver, D. (2005).
  Cooperative Pathfinding.
  Aiide, 1, 117-122.

- Erdmann, M., & Lozano-Perez, T. (1987).
  On multiple moving objects.
  Algorithmica, 2(1), 477-521.
"""

from __future__ import annotations

import heapq

import numpy as np

from ..roadmap.timed_roadmap import TimedVertex
from .planner import Planner


class PrioritizedPlanning(Planner):
    def __init__(
        self,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(verbose, **kwargs)
        self.verbose: int = verbose

    def get_name(self):
        return "PrioritizedPlanning"

    def _solve(self) -> None:
        T = len(self.trms[0].V) - 1  # makespan
        required_timestep = 1

        for agent in range(self.ins.num_agents):
            self.debug(f"agent-{agent} starts planning")
            goal_pos = self.ins.goals[agent]
            max_speed = self.ins.max_speeds[agent]
            rad = self.ins.rads[agent]
            trm = self.trms[agent]

            def get_f_value(v: TimedVertex) -> float:
                # Note: the value is scaled for the time axis
                return v.t + np.linalg.norm(goal_pos - v.pos) / max_speed

            def check_fin(v: TimedVertex) -> bool:
                # the last vertex is goal
                return v.t >= required_timestep and v == trm.V[v.t][-1]

            def insert(v: TimedVertex, OPEN: list[list]) -> None:
                # tie-break: f -> g -> random
                heapq.heappush(OPEN, [get_f_value(v), v.t, np.random.rand(), v])

            def valid_successor(v_from: TimedVertex, v_to: TimedVertex) -> bool:
                # TODO: develop smart collision checker
                return not any(
                    [
                        self.collide_dynamic_agents(
                            v_from.pos,
                            v_to.pos,
                            rad,
                            self.solution[i][v_from.t].pos,
                            self.solution[i][v_to.t].pos,
                            self.ins.rads[i],
                        )
                        for i in range(agent)
                    ]
                )

            path = self.get_single_agent_path(
                agent, get_f_value, check_fin, insert, valid_successor
            )

            if path is None:  # failed to solve
                self.solution.clear()
                self.info(f"agent-{agent} failed to find paths")
                return

            # update required_timestep (for collision check)
            required_timestep = max(len(path) - 1, required_timestep)

            # format new path, extending by goals
            path += [trm.V[t][-1] for t in range(len(path), T + 1)]

            # update solution
            self.solution.append(path)

        self.solved = True
