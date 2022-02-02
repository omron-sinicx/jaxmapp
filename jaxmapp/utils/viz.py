"""Utilities for visualization

Author: Keisuke Okumura

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

import pickle
import re
from glob import glob
from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jaxmapp.roadmap.timed_roadmap import TimedRoadmap

from ..env.instance import Instance
from ..planner import Result, get_cost

COLORS = list(mcolors.TABLEAU_COLORS)


def simple_plot_2d(
    ins: Instance,
    res: Optional[Result] = None,
    ax: matplotlib.axes._subplots.AxesSubplot = None,
) -> Optional[np.ndarray]:
    """visualize instance and solution

    Args:
        ins (Instance): problem instance
        res (Result): result data
        ax (matplotlib.axes._subplots.AxesSubplot) : subplot ax created by plt.subplots

    Returns:
        Optional[np.ndarray]: numpy array of the plot
    """
    solution = res.paths if res is not None else None

    if ax is None:
        ax = plt.axes()
    arrow_head = 0.02

    # plot obstacles

    if ins.obs.occupancy.max() == 1:
        ax.imshow(1 - ins.obs.occupancy.T, cmap="gray")
    scale = ins.obs.get_size()

    # plot solution
    if solution is not None and len(solution) > 0:
        path_len = (
            int(
                max(
                    [
                        get_cost(solution[i], ins.goals[i], ins.goal_rads[i])
                        for i in range(ins.num_agents)
                    ]
                )
            )
            + 1
        )
        for i, path in enumerate(solution):
            color = COLORS[i % len(COLORS)]
            jit = np.random.rand(2) * 0.02 - 0.01
            rad = ins.rads[i]
            s = path[0].pos + jit
            g = path[-1].pos + jit
            for t in range(path_len):
                u = path[t].pos + jit
                alpha = 1 - (0.4 / path_len) * t - 0.55
                ax.add_patch(
                    plt.Circle(u * scale, rad * scale, fc=color, alpha=alpha, ec=color)
                )
                if t == len(path) - 1:
                    continue
                v = path[t + 1].pos + jit
                ax.arrow(
                    u[0] * scale,
                    u[1] * scale,
                    (v[0] - u[0]) * scale,
                    (v[1] - u[1]) * scale,
                    color=color,
                    head_width=arrow_head,
                    length_includes_head=True,
                )

    # plot start and goal
    for i in range(ins.num_agents):
        color = COLORS[i % len(COLORS)]
        s = ins.starts[i] * scale
        g = ins.goals[i] * scale
        rad = ins.rads[i] * scale
        # start
        if solution is None or len(solution) == 0:
            ax.add_patch(plt.Circle(s, rad, fc=color, alpha=0.45, ec=color))
        ax.text(s[0], s[1], i, size=20)
        ax.scatter([s[0]], [s[1]], marker="o", color=color, s=40)
        # goal
        ax.scatter([g[0]], [g[1]], marker="x", color=color, s=40)

    # set axes
    ax.set_xlim(0, scale)
    ax.set_ylim(0, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    return None


def plot_trms(
    ins: Instance,
    trms: list[TimedRoadmap],
    res: Optional[Result] = None,
    agent: int = 0,
    is_timed: bool = False,
    ax: matplotlib.axes._subplots.AxesSubplot = None,
):
    """
    Plot timed roadmaps for a given agent

    Args:
        ins (Instance): problem instance
        trms (list[TimedRoadmap]): timed roadmap
        res (Optional[Result], optional): planning result. Defaults to None.
        agent (int, optional): agent id. Defaults to 0.
        is_timed (bool, optional): whether or not trms is created with timed_roadmap == True. Defaults to False.
        ax (matplotlib.axes._subplots.AxesSubplot) : subplot ax created by plt.subplots
    """

    if ax is None:
        ax = plt.axes()

    # obstacles
    if ins.obs.occupancy.max() == 1:
        ax.imshow(1 - ins.obs.occupancy.T, cmap="gray")
    scale = ins.obs.get_size()

    # roadmap
    color = "gray"
    trm = trms[agent]
    T = len(trm.V) - 1 if is_timed else 2
    for t in range(0, T):
        locs_t0 = np.array([v.pos for v in trm.V[t]]) * scale
        X_t0, Y_t0 = locs_t0[:, 0], locs_t0[:, 1]
        locs_t1 = np.array([v.pos for v in trm.V[t + 1]]) * scale
        X_t1, Y_t1 = locs_t1[:, 0], locs_t1[:, 1]
        for i, neighbors in enumerate(trm.E[t]):
            for j in neighbors:
                ax.plot(
                    [X_t0[i], X_t1[j]],
                    [Y_t0[i], Y_t1[j]],
                    color=color,
                    linewidth=1,
                    alpha=max(0.1, np.exp(-len(trm.V[t]) / 50.0)),
                )

    # start and goal
    color = "#03A9F4"
    s = ins.starts[agent] * scale
    g = ins.goals[agent] * scale
    rad = ins.rads[agent] * scale
    ax.add_patch(plt.Circle(s, rad, fc=color, alpha=1, ec=color))
    ax.scatter([g[0]], [g[1]], marker="s", color=color, s=40)

    # path
    if res is not None:
        color = "#03A9F4"
        path = res.paths[agent]
        s = path[0].pos
        g = path[-1].pos
        path_numpy = np.array([v.pos for v in path])
        ax.plot(
            path_numpy[:, 0] * scale,
            path_numpy[:, 1] * scale,
            color=color,
            linewidth=3,
            alpha=1,
        )

    ax.set_xlim(0, scale)
    ax.set_ylim(0, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.xaxis.tick_top()


def visualize_evaluation_results(
    dirname: str, is_global: bool = True, figsize=[12, 3]
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Visualize multiple benchmark results

    Args:
        dirname (str): directory containing reults with ***Sampler.pkl
        is_global (bool, optional): If runtime and cost metrics are computed for instances solved by all methods. Defaults to True.
        figsize (list, optional): figure size. Defaults to [12, 3].

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: success_rate, elapsed_trms, elapsed_planner, and cost
    """
    name = []
    solved = []
    success_rate = []
    elapsed_trms = []
    elapsed_planner = []
    cost = []
    for filename in sorted(glob(f"{dirname}/*Sampler.pkl")):
        res = pickle.load(open(filename, "rb"))[1]
        name.append(re.split("_", re.split("/", filename)[-1][:-4])[-1])
        solved.append(res[:, 2].astype(bool))
        success_rate.append(res[:, 2].astype(float).mean())
        elapsed_trms.append(res[:, -2].astype(float))
        elapsed_planner.append(res[:, -1].astype(float))
        cost.append((res[:, -3].astype(float) / res[:, 1].astype(int)))

    if is_global:
        all_solved = np.all(np.vstack(solved), axis=0)
        elapsed_trms = [e[all_solved].mean() for e in elapsed_trms]
        elapsed_planner = [e[all_solved].mean() for e in elapsed_planner]
        cost = [e[all_solved].mean() for e in cost]
    else:
        elapsed_trms = [e[s].mean() for e, s in zip(elapsed_trms, solved)]
        elapsed_planner = [e[s].mean() for e, s in zip(elapsed_planner, solved)]
        cost = [e[s].mean() for e, s in zip(cost, solved)]

    _, axes = plt.subplots(1, 3, figsize=figsize)
    x = range(len(name))
    axes[0].bar(x, success_rate)
    axes[0].set_title("Success rate")
    axes[0].set_xticks(x, name, rotation=30)
    axes[1].bar(x, cost)
    axes[1].set_xticks(x, name, rotation=30)
    axes[1].set_title("Sum-of-costs (normalized)")
    axes[2].bar(x, elapsed_trms, label="Roadmap")
    axes[2].bar(x, elapsed_planner, bottom=elapsed_trms, label="Planning")
    axes[2].set_title("Runtime (sec)")
    axes[2].set_xticks(x, name, rotation=30)
    plt.legend()

    return success_rate, elapsed_trms, elapsed_planner, cost
