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
import seaborn as sn
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
    dirname: str, abbrev: bool = False, figsize: tuple[int, int] = [12, 3]
) -> None:
    """
    Visualize benchmark experiment results

    Args:
        dirname (str): dataset directory that contains *Sampler.pkl
        abbrev (bool, optional): if the sampler name is abbreviated in the xtick. Defaults to False.
        figsize (tuple[int, int], optional): figure size. Defaults to [12, 3].
    """
    methods = []
    num_samples = []
    num_expanded = []
    solved = []
    success_rate = []
    elapsed_trms = []
    elapsed_planner = []
    cost = []
    for filename in sorted(glob(f"{dirname}/*Sampler.pkl")):
        data = pickle.load(open(filename, "rb"))
        config = data[0]
        res = data[1]
        methods.append(re.split("\.", config.sampler._target_)[-1])
        num_samples.append(config.sampler.num_samples)
        cost.append((res[:, 4].astype(float) / res[:, 1].astype(int)))
        num_expanded.append(res[:, 5].astype(float) / res[:, 1].astype(int))
        solved.append(res[:, 2].astype(bool))
        success_rate.append(res[:, 2].astype(float).mean())
        elapsed_trms.append(res[:, 6].astype(float))
        elapsed_planner.append(res[:, 7].astype(float))

    all_solved = np.all(np.vstack(solved), axis=0)
    num_expanded = [n[all_solved] for n in num_expanded]
    cost = [e[all_solved] for e in cost]
    elapsed_trms = np.array([e[all_solved].mean() for e in elapsed_trms])
    elapsed_planner = np.array([e[all_solved].mean() for e in elapsed_planner])

    methods = np.array(methods)
    method_list = np.unique(methods)
    num_samples = np.array(num_samples)
    success_rate = np.array(success_rate)
    cost = np.array(cost)

    _, axes = plt.subplots(1, 3, figsize=figsize)
    counter = 0
    for i, method in enumerate(method_list):
        ns = num_samples[methods == method]
        sr = success_rate[methods == method]
        et = elapsed_trms[methods == method]
        ep = elapsed_planner[methods == method]
        axes[0].plot(ns, sr, "o-", label=method)

        ne = [num_expanded[i] for i in range(len(methods)) if methods[i] == method]
        c = [cost[i] for i in range(len(methods)) if methods[i] == method]
        for ne_, c_ in zip(ne, c):
            sn.kdeplot(
                x=ne_,
                y=c_,
                ax=axes[1],
                alpha=0.5,
                shade=False,
                levels=5,
                thresh=0.3,
                color=COLORS[i],
            )
        axes[1].plot(
            np.array(ne).mean(axis=1),
            np.array(c).mean(axis=1),
            "o-",
            color=COLORS[i],
            label=method,
        )
        axes[2].bar(
            np.arange(len(ns)) + counter,
            et,
            label="Roadmap" if i == 0 else None,
            color="#333366",
        )
        axes[2].bar(
            np.arange(len(ns)) + counter,
            ep,
            bottom=et,
            label="Planner" if i == 0 else None,
            color="#9999bb",
        )
        counter += len(ns)

    axes[0].set_ylim([0, 1.1])
    axes[0].set_title("Success rate")
    axes[0].set_xlabel("num_samples per agent & timestep")
    axes[0].set_ylabel("Success rate")
    axes[0].legend()
    axes[1].set_ylim([0, 30])
    axes[1].set_xscale("log")
    axes[1].set_title("Sum-of-costs")
    axes[1].set_xlabel("expanded vertices / num_agents")
    axes[1].set_ylabel("Sum-of-costs / agents")
    axes[1].legend()
    runtime_labels = []
    for i, method in enumerate(method_list):
        ms = methods[methods == method]
        ns = num_samples[methods == method]
        runtime_labels += [f"{m[0] if abbrev else m}_{n:04d}" for m, n in zip(ms, ns)]
    axes[2].set_xticks(range(len(runtime_labels)), runtime_labels, rotation=90)
    axes[2].set_title("Runtime (sec)")
    axes[2].legend()
