r"""Contain ``lightning.LightningModule`` and code to manage them."""

from __future__ import annotations

__all__ = ["is_lmodule_config", "setup_lmodule"]

from lightcat.lmodule.factory import is_lmodule_config, setup_lmodule
