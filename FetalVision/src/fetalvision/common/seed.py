"""Seed everything that can be seeded. Full determinism on GPU is not guaranteed; we record the seed anyway."""
from __future__ import annotations
import os, random
import numpy as np


def set_seed(seed: int = 42) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        import torch  # noqa
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed
