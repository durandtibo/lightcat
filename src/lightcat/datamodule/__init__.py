r"""Contain ``lightning.LightningDataModule`` and code to manage
them."""

from __future__ import annotations

__all__ = ["is_datamodule_config", "setup_datamodule"]

from lightcat.datamodule.factory import is_datamodule_config, setup_datamodule
