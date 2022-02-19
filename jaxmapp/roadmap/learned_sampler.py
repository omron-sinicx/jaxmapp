"""Sampler using learned CTRMs

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from functools import partial
from logging import getLogger
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, dataclass

from ..env import Instance
from ..model.ctrmnet import CTRMNet, get_inference_fn
from .sampler import DefaultSampler
from .utils import valid_linear_move

logger = getLogger(__name__)


def load_pretrained_ctrm_sampler(
    model_yaml: str,
    model_dir: str,
    sampler_yaml: str,
    model_args: dict = None,
    sampler_args: dict = None,
) -> CTRMSampler:
    """
    Load pretrained CTRM sampler

    Args:
        model_yaml (str): Hydra config file for CTRM model
        model_dir (str): directory storing model checkpoint
        sampler_yaml (str): Hydra config file for CTRMSampler

    Returns:
        CTRMSampler: sampler
    """
    import hydra
    import omegaconf
    from flax.training.checkpoints import restore_checkpoint

    if model_args is not None:
        model = hydra.utils.instantiate(
            omegaconf.OmegaConf.load(model_yaml), **model_args
        )
    else:
        model = hydra.utils.instantiate(omegaconf.OmegaConf.load(model_yaml))
    params = restore_checkpoint(model_dir, None)["params"]
    if sampler_args is not None:
        sampler = hydra.utils.instantiate(
            omegaconf.OmegaConf.load(sampler_yaml), **sampler_args
        )
    else:
        sampler = hydra.utils.instantiate(omegaconf.OmegaConf.load(sampler_yaml))
    sampler.set_model_and_params(model, params)

    return sampler


@dataclass
class CTRMSampler(DefaultSampler):
    """Biased sampler using learned CTRM model"""

    share_roadmap: bool = False
    timed_roadmap: bool = True
    num_samples: int = None

    inference_fn: Callable = None  # inference function of CTRMNet
    params: dict = None  # parameter dicts for CTRMNet
    num_samples: int = 25  # number of trajectories (paths) to sample
    num_rw_samples: int = (
        15  # number of trajectories to sample with high random-walk decay
    )
    max_T: int = 64  # maximum number of timesteps for each traj
    prob_rw_decay_high: float = (
        25.0  # parameter for invoking random walk in learned trajectories
    )
    prob_rw_decay_low: float = (
        5.0  # parameter for invoking random walk in RW-prone trajectories
    )
    prob_rw_after_goal: float = (
        0.9  # probability of invoking random walk after reaching goal
    )
    num_rw_attempts: int = 3  # number of random walk to try
    max_speed_discount: float = (
        0.99  # hyperparameter to make model outputs valid in terms of maximum speed
    )

    def build_sample_vertices(self):
        def sample_vertices(
            key: PRNGKey, num_samples: int, instance: Instance
        ) -> Array:
            vertices = self.sample_trajectories(key, num_samples, instance)

            return vertices

        return sample_vertices

    def set_model_and_params(
        self, model: CTRMNet, params: dict, num_neighbors: Optional[int] = None
    ):
        """
        Set pretrained CTRM

        Args:
            model (CTRMNet): CTRMNet
            params (dict): trained parameters
            num_neighbors (Optional[int], optional): Number of neighbors to overwrite. Defaults to None.
        """
        if num_neighbors is not None:
            model.num_neighbors = num_neighbors
        self.model = model
        self.default_num_neighbors = model.num_neighbors
        self.inference_fn = get_inference_fn(model)
        self.params = params

    def sample_trajectories(
        self, key: PRNGKey, num_samples: int, instance: Instance
    ) -> Array:
        """
        Sample a predefined number of trajectories using the model

        Args:
            key (PRNGKey): jax.random.PRNGKey
            num_samples (int): number of trajectories
            ins (Instance): problem instance

        Returns:
            Array: stack of trajectories
        """

        pos_carry = (
            jnp.ones((num_samples, self.max_T, *instance.starts.shape)) * jnp.inf
        )
        makespan = self.max_T
        cost_map = instance.calc_cost_to_go_maps()
        for trial_id in range(num_samples):
            current_pos = instance.starts
            previous_pos = instance.starts
            has_reached_goals = jnp.zeros_like(instance.max_speeds).astype(bool)
            loop_carry = [
                key,
                current_pos,
                previous_pos,
                instance.goals,
                instance.max_speeds,
                instance.rads,
                instance.obs.occupancy,
                cost_map,
                instance.obs.sdf,
                pos_carry,
                trial_id,
                makespan,
                has_reached_goals,
            ]
            for t in range(self.max_T):
                loop_carry = sample_next(
                    t,
                    loop_carry,
                    self.params,
                    self.inference_fn,
                    prob_rw_decay=self.prob_rw_decay_high
                    if trial_id > self.num_rw_samples
                    else self.prob_rw_decay_low,
                    prob_rw_after_goal=self.prob_rw_after_goal,
                    num_rw_attempts=self.num_rw_attempts,
                    max_speed_discount=self.max_speed_discount,
                )
                has_reached_goals = loop_carry[-1]
                if jnp.all(has_reached_goals):
                    break
            pos_carry = loop_carry[-4]
            key = loop_carry[0]

        # at this point pos_carry has the shape of (num_trajs, max_T, num_agents, 2)
        pos_carry = pos_carry.transpose(
            2, 1, 0, 3
        )  # change it to (name_agents, max_T, num_trajs, 2)
        return pos_carry


@partial(
    jax.jit,
    static_argnames=(
        "inference_fn",
        "num_rw_attempts",
        "max_speed_discount",
    ),
)
def sample_next(
    t: int,
    loop_carry: list[Array],
    params: dict,
    inference_fn: Callable,
    prob_rw_decay: float,
    prob_rw_after_goal: float,
    num_rw_attempts: int,
    max_speed_discount: float,
) -> list[Array]:
    """
    Compiled function for sampling next vertex using the trained model

    Args:
        t (int): current timestep
        loop_carry (list[Array]): loop_carry
        params (dict): model parameters
        inference_fn (Callable): inference function of the model
        prob_rw_decay (float): parameter for decaying the probability of random walk
        prob_rw_after_goal (float): the probability of random walk after reaching the goal
        num_rw_attempts (int): number of resampling with the random walk
        max_speed_discount (float): parameter to make the movement valid within a single time step

    Returns:
        list[Array]: updated loop carry
    """

    # extract elements from loop_carry and update pos_carry
    (
        key,
        current_pos,
        previous_pos,
        goals,
        max_speeds,
        rads,
        occupancy,
        cost_map,
        sdf,
        pos_carry,
        trial_id,
        makespan,
        has_reached_goals,
    ) = loop_carry

    # determine random walk probability
    key0, key1, key = jax.random.split(key, 3)
    prob_random_walk = prob_random_walk = jnp.exp(-prob_rw_decay * t / makespan)
    has_reached_goals = jax.vmap(valid_linear_move, in_axes=(0, 0, 0, 0, None))(
        current_pos,
        goals,
        max_speeds,
        rads,
        sdf,
    )
    prob_random_walk = jax.vmap(
        lambda x: jax.lax.cond(
            x, lambda _: prob_rw_after_goal, lambda _: prob_random_walk, None
        )
    )(has_reached_goals)

    # generate next motion candidates
    next_motion_learned = inference_fn(
        params,
        key0,
        current_pos,
        previous_pos,
        goals,
        max_speeds,
        rads,
        occupancy,
        cost_map,
    )

    # clip next motion to ensure validity
    next_motion_mag = next_motion_learned[:, 2]
    next_motion_mag = jnp.minimum(next_motion_mag, max_speeds * max_speed_discount)
    next_motion_learned = next_motion_learned.at[:, 2].set(next_motion_mag)
    next_motion_dir = next_motion_learned[:, :2]
    vec_mag = jnp.expand_dims(jnp.linalg.norm(next_motion_dir, axis=-1), -1)
    vec_mag_avoid_zero = jnp.where(vec_mag == 0, 1, vec_mag)
    next_motion_dir = next_motion_dir / vec_mag_avoid_zero
    next_motion_learned = next_motion_learned.at[:, :2].set(next_motion_dir)

    def sample_uniform_i(key, max_speed):
        random_vals = jax.random.uniform(key, shape=(2 * num_rw_attempts,))
        mag = max_speed * random_vals[:num_rw_attempts] * max_speed_discount
        theta = jnp.pi * 2 * random_vals[num_rw_attempts:]
        next_motion_ = jnp.vstack((jnp.sin(theta), jnp.cos(theta), mag)).T
        return next_motion_

    next_motion_random = jax.vmap(sample_uniform_i, in_axes=(None, 0), out_axes=1)(
        key, max_speeds
    )
    next_motion_learned = jax.vmap(
        lambda x, nmr, nml: jax.lax.cond(
            jax.random.uniform(key1) < x, lambda _: nmr, lambda _: nml, None
        )
    )(prob_random_walk, next_motion_random[0], next_motion_learned)
    next_motion_learned = jnp.expand_dims(next_motion_learned, 0)

    next_motion = jnp.concatenate(
        (next_motion_learned, next_motion_random, jnp.zeros_like(next_motion_learned)),
        axis=0,
    )
    next_pos_cands = (next_motion[:, :, :2] * next_motion[:, :, 2:3]) + current_pos
    # determine next position
    validity = jax.vmap(
        jax.vmap(valid_linear_move, in_axes=(0, 0, 0, 0, None)),
        in_axes=(None, 0, None, None, None),
    )(
        current_pos,
        next_pos_cands,
        max_speeds,
        rads,
        sdf,
    ).T
    selected_id = jax.vmap(
        lambda x: num_rw_attempts
        + 1
        - jax.lax.fori_loop(
            0,
            num_rw_attempts + 2,
            lambda i, x: jax.lax.cond(x[0][i], lambda _: [x[0], i], lambda _: x, None),
            [x, 0],
        )[1]
    )(jnp.fliplr(validity))
    next_pos = jax.vmap(lambda x, y: x[y], in_axes=(1, 0))(next_pos_cands, selected_id)

    # update positions and pack everything back into loop_carry
    previous_pos = current_pos
    current_pos = next_pos
    pos_carry = pos_carry.at[trial_id, t].set(current_pos)
    loop_carry = [
        key,
        current_pos,
        previous_pos,
        goals,
        max_speeds,
        rads,
        occupancy,
        cost_map,
        sdf,
        pos_carry,
        trial_id,
        makespan,
        has_reached_goals,
    ]

    return loop_carry
