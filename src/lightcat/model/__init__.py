r"""Contain ``lightning.LightningModule`` and code to manage them."""

from __future__ import annotations

__all__ = ["is_model_config", "setup_model"]

from lightcat.model.factory import is_model_config, setup_model
