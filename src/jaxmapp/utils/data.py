"""Utilities for data loading

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import os
import pickle
import re
from logging import getLogger

import numpy as np
import scipy.sparse
from CMap2D import CMap2D

from ..env import Instance
from ..env.obstacle import ObstacleMap
from ..planner import Result


def load_instance(filename: str) -> Instance:
    """
    Load problem instance

    Args:
        filename (str): filename

    Returns:
        Instance: problem instance
    """
    ins = pickle.load(open(filename, "rb"))
    occupancy = np.array(ins.obs.occupancy.todense())
    cmap2d = CMap2D()
    cmap2d.from_array(occupancy, (0, 0), 1.0 / occupancy.shape[0])
    sdf = cmap2d.as_sdf()
    obs = ObstacleMap(occupancy, sdf)
    ins.obs = obs
    return ins


def load_result(filename: str) -> Result:
    """
    Load planning result

    Args:
        filename (str): filename

    Returns:
        Result: planning result
    """
    res = pickle.load(open(filename, "rb"))
    return res


def save_instance(ins: Instance, filename: str) -> None:
    """
    Save problem instance

    Args:
        ins (Instance): problem instance
        filename (str): filename
    """
    occupancy = scipy.sparse.coo_matrix(ins.obs.occupancy)
    obs = ObstacleMap(occupancy, None)
    ins.obs = obs
    pickle.dump(ins, open(filename, "wb"))


def get_original_config(dirname: str) -> dict:
    """
    Get original model config used with hydra

    Args:
        dirname (str): model directory

    Returns:
        dict: config dict
    """
    config = pickle.load(open(f"{dirname}/tb/config.pkl", "rb"))
    return config
