"""Encoders

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import flax.linen as nn
from chex import Array


class AgentEncoder(nn.Module):

    dim_hidden: int
    dim_attention: int
    dim_message: int

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        x = nn.Dense(self.dim_hidden)(x)
        x = nn.relu(x)
        a = nn.Dense(self.dim_attention)(x)
        m = nn.Dense(self.dim_message)(x)
        m = nn.sigmoid(m)
        return m, a


class FOVEncoder(nn.Module):

    dim_hidden: int
    dim_output: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.dim_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim_output)(x)
        x = nn.sigmoid(x)
        return x


class MLP(nn.Module):
    """MLP network used as a base module in CVAE"""

    dim_hidden: int
    dim_output: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.dim_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim_output)(x)
        return x
