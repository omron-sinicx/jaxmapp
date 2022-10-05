"""Training CTRM Net

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import pickle
from functools import partial
from logging import getLogger

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.checkpoints import save_checkpoint
from flax.training.train_state import TrainState
from numpy2tfrecord import build_dataset_from_tfrecord
from tensorboardX import SummaryWriter
from tqdm import tqdm


@partial(jax.jit, static_argnums=(3, 4, 5))
def step(key, batch, state, kl_weight, ind_weight, is_training=True):
    key0, key = jax.random.split(key)

    def loss_fn(params):
        output = state.apply_fn(
            params,
            key0,
            batch["current_pos"],
            batch["previous_pos"],
            batch["next_pos"],
            batch["goals"],
            batch["max_speeds"],
            batch["rads"],
            batch["occupancy"],
            batch["cost_map"],
        )
        weights = output[6]
        recon_loss = ((output[0] - output[1]) ** 2).sum(axis=-1) * weights
        ind_loss = (
            optax.softmax_cross_entropy(logits=output[2], labels=output[3]) * weights
        )
        kl_loss = (jnp.exp(output[4]) * (output[4] - output[5])).sum(axis=-1) * weights
        loss = (recon_loss + kl_weight * kl_loss + ind_weight * ind_loss).mean()
        aux = {
            "output": output,
            "recon_loss": recon_loss.mean(),
            "ind_loss": ind_loss.mean() * ind_weight,
            "kl_loss": kl_loss.mean() * kl_weight,
        }
        return loss, aux

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
    else:
        loss, aux = loss_fn(state.params)
    return key, loss, state, aux


@hydra.main(config_path="config", config_name="train")
def main(config):
    logger = getLogger(__name__)
    logger.info(f"random seed: {config.seed}")
    map_size = config.dataset.instance.map_size
    logger.info(f"map size: {map_size}")

    datasets = dict()
    for split in ["train", "val"]:
        filename = f"{config.dataset.datadir}/{split}_{map_size:05d}.tfrec"
        dataset = build_dataset_from_tfrecord(filename)
        dataset = (
            dataset.shuffle(config.batch_size * 100, seed=config.seed)
            .batch(config.batch_size)
            .prefetch(1)
        )
        datasets[split] = dataset

    writer = SummaryWriter(log_dir=f"{config.modeldir}/tb")
    pickle.dump(config, open(f"{config.modeldir}/tb/config.pkl", "wb"))

    key = jax.random.PRNGKey(config.seed)
    model = hydra.utils.instantiate(config.model)
    batch = next(datasets["train"].as_numpy_iterator())
    params = model.init(
        key,
        key,
        batch["current_pos"][:, :, 0],
        batch["previous_pos"][:, :, 0],
        batch["next_pos"][:, :, 0],
        batch["goals"],
        batch["max_speeds"],
        batch["rads"],
        batch["occupancy"],
        batch["cost_map"],
    )

    tx = optax.adam(config.learning_rate)
    apply_fn = jax.vmap(
        model.apply, in_axes=(None, None, 2, 2, 2, None, None, None, None, None)
    )
    state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)

    losses = {"train": None, "val": None}
    losses_details = {
        "recon_loss": {"train": None, "val": None},
        "kl_loss": {"train": None, "val": None},
        "ind_loss": {"train": None, "val": None},
    }
    val_loss_best = jnp.inf
    for e in range(config.num_epochs):
        for split in ["train", "val"]:
            for batch in tqdm(datasets[split].as_numpy_iterator()):
                key, loss, state, aux = step(
                    key,
                    batch,
                    state,
                    config.kl_weight,
                    config.ind_weight,
                    is_training=(split == "train"),
                )
                losses[split] = (
                    loss if losses[split] is None else (loss + losses[split]) / 2.0
                )
                for kwg in losses_details.keys():
                    losses_details[kwg][split] = (
                        aux[kwg]
                        if losses_details[kwg][split] is None
                        else (aux[kwg] + losses_details[kwg][split]) / 2.0
                    )

            writer.add_scalar(f"loss/{split}", losses[split], e)
            for kwg in losses_details.keys():
                writer.add_scalar(
                    f"loss_details_{split}/{kwg}", losses_details[kwg][split], e
                )
        logger.info(
            f"epoch: {e} | train loss: {losses['train']:0.4f} | val loss: {losses['val']:0.4f}"
        )
        if losses["val"] < val_loss_best:
            val_loss_best = losses["val"]
            save_checkpoint(config.modeldir, state, e)
            logger.info(
                f"update best model (epoch: {e} | val loss: {val_loss_best:0.4f})"
            )

    writer.close()


if __name__ == "__main__":
    main()
