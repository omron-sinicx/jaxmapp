"""Evaluating roadmaps on benchmark data

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import pickle
import re
import time
from glob import glob
from logging import getLogger

import hydra
import jax
import numpy as np
from flax.training.checkpoints import restore_checkpoint
from jaxmapp.utils.data import get_original_config, load_instance
from tqdm import tqdm


@hydra.main(config_path="config", config_name="eval")
def main(config):
    logger = getLogger(__name__)
    logger.info(f"random seed: {config.seed}")
    key = jax.random.PRNGKey(config.seed)

    sampler = hydra.utils.instantiate(config.sampler)
    if hasattr(sampler, "set_model_and_params"):
        model = hydra.utils.instantiate(config.model)
        original_config = get_original_config(config.modeldir)
        model.num_neighbors = original_config.model.num_neighbors
        params = restore_checkpoint(config.modeldir, None)["params"]
        sampler.set_model_and_params(model, params)

    # dryrun
    if config.dryrun:
        logger.info("start jit compile")
        tic = time.time()
        for num_agents in tqdm(
            range(
                config.dataset.instance.num_agents_min,
                config.dataset.instance.num_agents_max + 1,
            )
        ):
            generator = hydra.utils.instantiate(
                config.dataset.instance,
                num_agents_min=num_agents,
                num_agents_max=num_agents,
            )
            ins = generator.generate(key)
            ins.starts = ins.goals
            trms = sampler.construct_trms(key, ins)
        toc_compile = time.time() - tic
        logger.info(f"completed jit compile. {toc_compile=}")

    # main run
    results = []
    num_solved = 0
    for insfile in sorted(glob(f"{config.dataset.datadir}/*_ins.pkl")):
        logger.info(insfile)
        ins = load_instance(insfile)
        ins = ins.to_jnumpy()
        tic = time.time()
        trms = sampler.construct_trms(key, ins)
        toc_trms = time.time() - tic
        planner = hydra.utils.instantiate(config.planner)
        res = planner.solve(ins.to_numpy(), trms)
        num_solved += res.solved
        toc_planner = time.time() - tic - toc_trms
        result_text = f"{res.solved=} ({num_solved=}), {res.maximum_costs=:0.2f}, {res.sum_of_costs/ins.num_agents=:0.2f}, {toc_trms=:0.2f}, {toc_planner=:0.2f}"
        logger.info(result_text)

        results.append(
            [
                insfile,
                int(ins.num_agents),
                int(res.solved),
                res.maximum_costs,
                res.sum_of_costs,
                res.lowlevel_expanded,
                toc_trms,
                toc_planner,
            ]
        )
    logger.info("complete all instances")
    results = np.vstack(results)
    sampler_name = re.split("\.", config.sampler._target_)[-1]
    pickle.dump(
        [config, results],
        open(f"{config.savefile_prefix}_{sampler_name}.pkl", "wb"),
    )


if __name__ == "__main__":
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "This script must be executed in a CPU environment. Execute 'CUDA_VISIBLE_DEVICES= python scripts/eval.py'"
        )

    main()
