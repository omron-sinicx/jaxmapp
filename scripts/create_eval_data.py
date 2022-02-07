"""Creating benchmark dataset

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

import os
from logging import getLogger

import hydra
import jax
from jaxmapp.utils.data import check_rootdir, save_instance
from tqdm import tqdm


@hydra.main(config_path="config", config_name="create_eval_data")
def main(config):
    logger = getLogger(__name__)
    config.rootdir = check_rootdir(config.rootdir)
    logger.info(f"random seed: {config.seed}")
    savedir = f"{config.rootdir}/{config.dataset.datadir}"
    os.makedirs(savedir, exist_ok=True)
    logger.info(f"creating {config.dataset.num_instances} instances")
    key = jax.random.PRNGKey(config.seed)
    generator = hydra.utils.instantiate(config.dataset.instance)

    for i in tqdm(range(config.dataset.num_instances)):
        key0, key = jax.random.split(key)
        ins = generator.generate(key0)
        save_instance(ins, f"{savedir}/{i:08d}_ins.pkl")


if __name__ == "__main__":
    main()
