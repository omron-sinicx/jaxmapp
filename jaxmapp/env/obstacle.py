"""Static objects

Author: Keisuke Okumura and Ryo Yonetani

Affiliation: Tokyo Institute of Technology, OMRON SINIC X

"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from chex import Array
from skimage.draw import disk


class ObstacleMap(NamedTuple):
    """Obstacle map containing occupancy map and its signed distance function"""

    occupancy: Array
    sdf: Array

    def get_size(self):
        return self.occupancy.shape[0]


class ObstacleSphere(NamedTuple):
    """Static sphere obstacle"""

    pos: np.ndarray  # center position
    rad: float  # radius

    def draw_2d(self, map_size: int) -> np.ndarray:
        """
        Draw 2d image for creating occupancy maps

        Args:
            map_size (int): map size

        Returns:
            np.ndarray: occupancy map for the given circle obstacle
        """

        shape = (map_size, map_size)
        img = np.zeros(shape, dtype=np.float32)
        X = int(map_size * self.pos[0])
        Y = int(map_size * self.pos[1])
        R = int(map_size * self.rad)
        rr, cc = disk((X, Y), R, shape=shape)
        img[rr, cc] = 1.0

        return img
