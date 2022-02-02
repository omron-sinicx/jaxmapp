"""CTRM Net

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial
from typing import Callable

import flax.linen as fnn
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from .cvae import CVAE
from .preprocessor import Preprocessor


class CTRMNet(fnn.Module):
    """
    CTRM sampler model

    This model integrates a feature extractor (Preprocessor) and a CVAE model (CVAE) to serve as a sampler for constructing CTRM.
    """

    # common
    dim_hidden: int = 32

    # preprocessor
    fov_size: int = 19
    num_neighbors: int = 15
    dim_attention: int = 10
    dim_message: int = 32
    dim_output_fov: int = 32

    # CVAE
    dim_latent: int = 64
    dim_output: int = 3
    dim_indicator: int = 3
    temp: float = 2.0

    def setup(self):
        self.preprocessor = Preprocessor(
            fov_size=self.fov_size,
            num_neighbors=self.num_neighbors,
            dim_hidden=self.dim_hidden,
            dim_attention=self.dim_attention,
            dim_message=self.dim_message,
            dim_output_fov=self.dim_output_fov,
        )
        self.cvae = CVAE(
            dim_hidden=self.dim_hidden,
            dim_latent=self.dim_latent,
            dim_output=self.dim_output,
            dim_indicator=self.dim_indicator,
            temp=self.temp,
        )

    def __call__(
        self,
        key: Array,
        arr_current_pos: Array,
        arr_prev_pos: Array,
        arr_next_pos: Array,
        goals: Array,
        max_speeds: Array,
        rads: Array,
        occupancy_map: Array,
        cost_to_go_maps: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """
        Forward function used in the training phase

        Args:
            key (Array): jax.random.PRNGKey
            arr_current_pos (Array): current positions
            arr_prev_pos (Array): previous positions
            arr_next_pos (Array): next positions
            goals (Array): goal positions
            max_speeds (Array): list of agent's max speeds
            rads (Array): list of agent's radius
            occupancy_map (Array): occupancy map
            cost_to_go_maps (Array): cost-to-go maps

        Returns:
            tuple[Array, Array, Array, Array, Array, Array, Array]: a tuple of y_pred, y, ind_pred, ind, log_prob_x, log_prob_y, weight
        """

        features = jax.vmap(self.preprocessor)(
            arr_current_pos,
            arr_prev_pos,
            goals,
            max_speeds,
            rads,
            occupancy_map,
            cost_to_go_maps,
        )
        diff = arr_next_pos - arr_current_pos
        vec_mag = jnp.expand_dims(jnp.linalg.norm(diff, axis=-1), -1)
        vec_mag_avoid_zero = jnp.where(vec_mag == 0, 1, vec_mag)
        y = jnp.concatenate((diff / vec_mag_avoid_zero, vec_mag), -1)

        arr_sin = jnp.cross(
            (goals - arr_current_pos)
            / jnp.linalg.norm(goals - arr_current_pos + 1e-10, axis=-1, keepdims=True),
            (arr_next_pos - arr_current_pos)
            / jnp.linalg.norm(
                arr_next_pos - arr_current_pos + 1e-10, axis=-1, keepdims=True
            ),
        )
        # workaround -- indicator will be given as "going straight" when
        # the agent already reached the goal and current_pos = next_pos = goal_pos
        arr_sin = jnp.where(jnp.isnan(arr_sin), 0.0, arr_sin)
        arr_sin = jnp.clip(arr_sin, -1, 1)
        ind = ((arr_sin + 1) / 2 * self.dim_indicator).astype(int)
        ind = jax.nn.one_hot(ind, self.dim_indicator)
        y_pred, ind_pred, log_prob_x, log_prob_y = self.cvae(key, features, y, ind)

        bc = jnp.bincount(jnp.argmax(ind, -1).flatten(), length=self.dim_indicator)
        weight = ind @ jnp.exp(-(bc / bc.sum()))

        return y_pred, y, ind_pred, ind, log_prob_x, log_prob_y, weight

    def __call_inference__(
        self,
        key: PRNGKey,
        arr_current_pos: Array,
        arr_prev_pos: Array,
        goals: Array,
        max_speeds: Array,
        rads: Array,
        occupancy_map: Array,
        cost_to_go_maps: Array,
    ) -> Array:
        """
        Forward function used in the inference phase

        Args:
            key (PRNGKey): jax.random.PRNGKey
            arr_current_pos (Array): current positions
            arr_prev_pos (Array): previous positions
            goals (Array): goal positions
            max_speeds (Array): list of agent's max speeds
            rads (Array): list of agent's radius
            occupancy_map (Array): occupancy map
            cost_to_go_maps (Array): cost-to-go maps

        Returns:
            Array: next-step information
        """

        features = self.preprocessor(
            arr_current_pos,
            arr_prev_pos,
            goals,
            max_speeds,
            rads,
            occupancy_map,
            cost_to_go_maps,
        )
        y_pred = self.cvae.__call_inference__(key, features)

        return y_pred


def get_inference_fn(net: CTRMNet) -> Callable:
    """Instantiate jit-compiled inference function"""
    return jax.jit(partial(net.apply, method=net.__call_inference__))
