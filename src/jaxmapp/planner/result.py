"""Result of planning

Author: Keisuke Okumura

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..roadmap import TimedVertex


@dataclass(frozen=True)
class Result:
    solved: bool  # true -> solved
    num_agents: int
    paths: list[list[TimedVertex]] = field(default_factory=list)  # solution paths
    name_planner: str = ""  # planner name
    elapsed_planner: float = 0  # runtime of planner
    sum_of_costs: float = 0
    maximum_costs: float = 0  # aka. makespan
    sum_of_travel_dists: float = 1
    maximum_travel_dists: float = 0

    # collision check, planner
    cnt_static_collide: int = 0
    cnt_continuous_collide: int = 0

    # planning effort
    lowlevel_expanded: int = 0
    lowlevel_explored: int = 0

    def get_dict_wo_paths(self) -> dict[Any, Any]:
        dic = asdict(self)
        del dic["paths"]
        return dic

    def __repr__(self):
        status = "solved" if self.solved else "failed"
        soc_normed = self.sum_of_costs / self.num_agents
        return f"Status: {status}, makespan: {self.maximum_costs:0.2f}, sum-of-costs: {self.sum_of_costs:0.2f}, sum-of-costs (normed): {soc_normed:0.2f}"
