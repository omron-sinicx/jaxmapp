"""Feature extractor

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import flax.linen as fnn
import jax
import jax.numpy as jnp
from chex import Array

from .utils.encoder import AgentEncoder, FOVEncoder
from .utils.features import get_arr_others_info, get_fov_features, get_self_info


class Preprocessor(fnn.Module):
    """Preprocessor to extract features from input instances"""

    fov_size: int
    num_neighbors: int
    dim_hidden: int
    dim_attention: int
    dim_message: int
    dim_output_fov: int

    def setup(self):

        self.agent_encoder = AgentEncoder(
            dim_hidden=self.dim_hidden,
            dim_attention=self.dim_attention,
            dim_message=self.dim_message,
        )
        self.fov_encoder = FOVEncoder(
            dim_hidden=self.dim_hidden, dim_output=self.dim_output_fov
        )
        self.fov_encoder_vain = FOVEncoder(
            dim_hidden=self.dim_hidden, dim_output=self.dim_output_fov
        )

    def __call__(
        self,
        arr_current_pos: Array,
        arr_prev_pos: Array,
        goals: Array,
        max_speeds: Array,
        rads: Array,
        occupancy_map: Array,
        cost_to_go_maps: Array,
    ) -> Array:
        """
        Extract one-step features

        Args:
            arr_current_pos (Array): current positions
            arr_prev_pos (Array): previous positions
            goals (Array): goal positions
            max_speeds (Array): a list of max speeds
            rads (Array): a list of agent's radius
            occupancy_map (Array): occupancy map
            cost_to_go_maps (Array): cost-to-go maps

        Returns:
            Array: concatenation of goal-driven, fov, and commucation features
        """

        num_agents = len(arr_current_pos)
        num_neighbors = self.num_neighbors
        assert num_neighbors < num_agents

        # compute fov
        arr_fov = [
            get_fov_features(
                current_pos=arr_current_pos[j],
                occupancy_map=occupancy_map,
                cost_to_go_map=cost_to_go_maps[j],
                fov_size=self.fov_size,
                flatten=True,
            )
            for j in range(num_agents)
        ]
        arr_fov = jnp.array([jnp.array(x).astype(float) for x in arr_fov])
        encoded_fov = self.fov_encoder(arr_fov)
        encoded_fov_vain = self.fov_encoder_vain(arr_fov)

        arr_others_info = get_arr_others_info(
            arr_current_pos, goals, arr_prev_pos, max_speeds, rads
        )
        self_info = get_self_info(arr_others_info, num_agents)
        input_vain = jnp.concatenate(
            (arr_others_info, jnp.concatenate([encoded_fov_vain] * num_agents)), axis=-1
        )

        messages, attentions = self.agent_encoder(input_vain)

        dist_mat = arr_others_info.reshape(num_agents, num_agents, -1)[..., 2]
        messages_mat = messages.reshape(num_agents, num_agents, -1)
        attentions_mat = attentions.reshape(num_agents, num_agents, -1)
        sorted_index = jnp.argsort(dist_mat)
        sort = jax.vmap(lambda x, y: x[y])
        # sorted_dist_mat = sort(dist_mat, sorted_index)
        sorted_messages_mat = sort(messages_mat, sorted_index)[:, 1 : num_neighbors + 1]
        sorted_attentions_mat = sort(attentions_mat, sorted_index)[
            :, : num_neighbors + 1
        ]
        # do not use jnp.linalg.norm here as it will cause nan error
        sim = -jax.vmap(lambda x: ((x - x[0]) ** 2).sum(axis=1))(sorted_attentions_mat)
        weights = fnn.softmax(sim, axis=1)[:, 1:]
        vain_output = (sorted_messages_mat * jnp.expand_dims(weights, -1)).sum(1)

        return jnp.concatenate((self_info, encoded_fov, vain_output), -1)
