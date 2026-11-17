"""
Compatibility helpers for different pytorch_lightning versions.
"""
import inspect
from functools import lru_cache

import pytorch_lightning as pl


@lru_cache(maxsize=None)
def checkpoint_load_kwargs() -> dict:
    """
    Extra kwargs for LightningModule.load_from_checkpoint.

    Up to pytorch_lightning 2.5.x checkpoints were loaded with torch.load(weights_only=False).
    Since 2.6 load_from_checkpoint has a `weights_only` argument that falls back to the torch
    default (True since torch 2.6), which fails on our checkpoints with custom objects
    (e.g. label_converter in hyper_parameters). Our models come from trusted modelhub,
    so restore the previous behaviour when the argument is supported.
    """
    params = inspect.signature(pl.LightningModule.load_from_checkpoint).parameters
    if "weights_only" in params:
        return {"weights_only": False}
    return {}
