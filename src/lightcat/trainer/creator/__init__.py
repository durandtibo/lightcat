r"""Contain the trainer creators."""

from __future__ import annotations

__all__ = [
    "BaseTrainerCreator",
    "is_trainer_creator_config",
    "setup_trainer_creator",
    "TrainerCreator",
]

from lightcat.trainer.creator.base import (
    BaseTrainerCreator,
    is_trainer_creator_config,
    setup_trainer_creator,
)
from lightcat.trainer.creator.vanilla import TrainerCreator
