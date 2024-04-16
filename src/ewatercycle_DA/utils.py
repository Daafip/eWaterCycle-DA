import numpy as np

rng = np.random.default_rng()  # Initiate a Random Number Generator
def add_normal_noise(like_sigma) -> float:
    """Normal (zero-mean) noise to be added to a state

    Args:
        like_sigma (float): scale parameter - pseudo variance & thus 'like'-sigma

    Returns:
        sample from normal distribution
    """
    return rng.normal(loc=0, scale=like_sigma)  # log normal so can't go to 0 ?

import datetime
from datetime import timezone
from pathlib import Path

from ewatercycle.util import to_absolute_path
from ewatercycle.config import CFG
def custom_make_cfg_dir(model_name, i) -> Path:
    """adds ensemble member index after make config path implemented by eWatercycle

    based on:
    https://github.com/Daafip/ewatercycle/blob/c7bc51dee4e7ea8b69f79710e5b80ef55067c3c1/src/ewatercycle/base/model.py#L121
    """
    tz = timezone.utc
    timestamp = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")
    folder_prefix = model_name.lower()
    if i < 10:
        i = f'0{i}'
    cfg_path = to_absolute_path(
        f"{folder_prefix}_{i}_{timestamp}", parent=CFG.output_dir
    )

    cfg_path.mkdir(parents=True, exist_ok=True)

    return cfg_path