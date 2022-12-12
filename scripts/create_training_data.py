"""Creating training data

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import os
import pickle
from logging import getLogger

import hydra
import jax
from jaxmapp.utils.data import save_instance
from joblib import Parallel, delayed
from tqdm import tqdm


def process(config, seed, seed_start, savedir):
    success = False
    num_trials = 0
    key = jax.random.PRNGKey(seed + seed_start)
    generator = hydra.utils.instantiate(config.dataset.instance)

    while not success:
        key0, key1, key = jax.random.split(key, 3)
        try:
            ins = generator.generate(key0)
        except:
            print("Exception: could not initialize valid start/goal pairs")
            num_trials += 1
        else:
            sampler = hydra.utils.instantiate(config.sampler)
            trms = sampler.construct_trms(key1, ins)
            planner = hydra.utils.instantiate(config.planner)
            res = planner.solve(ins.to_numpy(), trms)
            success = res.solved
            if not success:
                num_trials += 1

        if num_trials == config.num_max_trials:
            print(f"Failed {num_trials} times -- break")
            return None, None

    save_instance(ins, f"{savedir}/{seed:08d}_ins.pkl")
    pickle.dump(res, open(f"{savedir}/{seed:08d}_res.pkl", "wb"))


@hydra.main(config_path="config", config_name="create_training_data")
def main(config):
    logger = getLogger(__name__)
    logger.info(f"random seed: {config.seed}")
    seed_start = 0
    for label, num_instances in config.dataset.num_instances.items():
        savedir = f"{config.dataset.datadir}/{label}"
        os.makedirs(savedir, exist_ok=True)
        logger.info(f"creating {num_instances} {label} instances")
        if config.n_jobs == 1:
            for i in tqdm(range(num_instances)):
                process(config, i, seed_start, savedir)
        else:
            Parallel(n_jobs=config.n_jobs, verbose=1)(
                [
                    delayed(process)(config, i, seed_start, savedir)
                    for i in range(num_instances)
                ]
            )
        seed_start += num_instances


if __name__ == "__main__":
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "This script must be executed in a CPU environment. Execute 'CUDA_VISIBLE_DEVICES= python scripts/create_training_data.py'"
        )
    main()
