"""CVAE

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from tensorflow_probability.substrates.jax.distributions import RelaxedOneHotCategorical

from .utils.encoder import MLP


class CVAE(nn.Module):
    """
    Conditional variational autoencoder with discrete latent variables.

    ref:
    Ivanovic et al., Multimodal Deep Generative Models for Trajectory Prediction: A Conditional Variational Autoencoder Approach
    https://arxiv.org/abs/2008.03880

    """

    dim_hidden: int
    dim_latent: int
    dim_output: int
    dim_indicator: int
    temp: float

    def setup(self):
        self.encoder_input = MLP(self.dim_hidden, self.dim_latent)
        self.encoder_output = MLP(self.dim_hidden, self.dim_latent)
        self.decoder = MLP(self.dim_hidden, self.dim_output)
        self.indicator = MLP(self.dim_hidden, self.dim_indicator)

    def __call__(
        self, key: PRNGKey, x: Array, y: Array, ind: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Forward function used in the training phase

        Args:
            key (PRNGKey): jax.random.PRNGKey
            x (Array): current observation
            y (Array): next step information
            ind (Array): indicator value

        Returns:
            tuple[Array, Array, Array, Array]: [description]
        """

        ind_pred = nn.log_softmax(self.indicator(x))
        augment_x = jnp.concatenate((x, ind), axis=-1)
        log_prob_x = nn.log_softmax(self.encoder_input(augment_x))

        augment_y = jnp.concatenate((x, y), axis=-1)
        log_prob_y = nn.log_softmax(self.encoder_output(augment_y))
        dist_y = RelaxedOneHotCategorical(temperature=self.temp, logits=log_prob_y)
        latent_y = dist_y.sample(seed=key)
        y_pred = self.decoder(jnp.concatenate([latent_y, augment_x], axis=-1))

        return y_pred, ind_pred, log_prob_x, log_prob_y

    def __call_inference__(self, key: PRNGKey, x: Array) -> Array:
        """
        Forward function used in the inference (i.e., roadmap construction) phase

        Args:
            key (PRNGKey): jax.random.PRNGKey
            x (Array): current observation

        Returns:
            Array: next step information
        """

        ind_pred = jnp.eye(self.dim_indicator)[jnp.argmax(self.indicator(x), -1)]
        augment_x = jnp.concatenate((x, ind_pred), axis=-1)
        log_prob_x = nn.log_softmax(self.encoder_input(augment_x))
        dist_x = RelaxedOneHotCategorical(temperature=self.temp, logits=log_prob_x)
        latent_x = dist_x.sample(seed=key)
        y_pred = self.decoder(jnp.concatenate([latent_x, augment_x], axis=-1))

        return y_pred

    @staticmethod
    def get_inference_fn(net: CVAE) -> Callable:
        """Instantiate jit-compiled inference function"""
        return jax.jit(partial(net.apply, method=net.__call_inference__))
