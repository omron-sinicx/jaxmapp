"""Utilities for planning

Author: Keisuke Okumura

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

import numpy as np

from ..roadmap import TimedVertex


def get_cost(path: list[TimedVertex], goal: np.ndarray, goal_rad: float) -> float:
    """Compute path cost

    Args:
        path (list[TimedVertex])
        goal (np.ndarray): goal
        goal_rad (float): goal radius

    Returns:
        float: cost

    Note:
        In precisely, this is approximation.
    """
    cost = len(path) - 1
    while cost - 1 >= 0 and np.linalg.norm(path[cost - 1].pos - goal) <= goal_rad:
        cost -= 1
    return cost


def get_travel_dist(path: list[TimedVertex]) -> float:
    """Compute travel distance

    Args:
        path (list[TimedVertex]): path

    Returns:
        float: distance
    """

    return float(
        sum(
            [
                np.linalg.norm(path[i + 1].pos - path[i].pos)
                for i in range(len(path) - 1)
            ]
        )
    )
