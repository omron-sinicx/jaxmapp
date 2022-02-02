"""Converting created training data to tfrecord

Author: Ryo Yonetani

Affiliation: OMRON SINIC X

"""

from __future__ import annotations

from glob import glob
from logging import getLogger

import hydra
import numpy as np
from jaxmapp.utils.data import check_rootdir, load_instance, load_result
from numpy2tfrecord import Numpy2TFRecordConverter
from tqdm import tqdm


@hydra.main(config_path="config", config_name="train")
def main(config):
    logger = getLogger(__name__)
    config.rootdir = check_rootdir(config.rootdir)
    logger.info(f"random seed: {config.seed}")
    map_size = config.dataset.instance.map_size
    logger.info(f"map size: {map_size}")

    for split in ["train", "val", "test"]:
        dirname = f"{config.rootdir}/{config.dataset.datadir}/{split}"
        filename = (
            f"{config.rootdir}/{config.dataset.datadir}/{split}_{map_size:05d}.tfrec"
        )
        logger.info(f"dirname: {dirname}")

        with Numpy2TFRecordConverter(filename) as converter:
            for i, r in tqdm(
                zip(
                    sorted(glob(f"{dirname}/*_ins.pkl")),
                    sorted(glob(f"{dirname}/*_res.pkl")),
                )
            ):
                ins = load_instance(i)
                res = load_result(r)
                paths = np.stack([np.array([x.pos for x in y]) for y in res.paths])
                current_pos = paths[:, 0:-1, :]
                previous_pos = np.concatenate(
                    (paths[:, 0:1, :], paths[:, 0:-2, :]), axis=1
                )
                next_pos = paths[:, 1:, :]

                sample = {
                    "current_pos": current_pos.astype(np.float32),
                    "previous_pos": previous_pos.astype(np.float32),
                    "next_pos": next_pos.astype(np.float32),
                    "goals": ins.goals.astype(np.float32),
                    "max_speeds": ins.max_speeds.astype(np.float32),
                    "rads": ins.rads.astype(np.float32),
                    "occupancy": ins.obs.occupancy.astype(np.float32),
                    "cost_map": ins.calc_cost_to_go_maps().astype(np.float32),
                }
                converter.convert_sample(sample)


if __name__ == "__main__":
    main()
