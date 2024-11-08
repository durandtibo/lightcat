r"""Contain ``lightning.Trainer`` and code to manage them."""

from __future__ import annotations

__all__ = ["is_trainer_config", "setup_trainer"]

from lightcat.trainer.factory import is_trainer_config, setup_trainer
