r"""Contain callbacks and code to manage callbacks."""

from __future__ import annotations

__all__ = ["is_callback_config", "setup_callback", "setup_list_callbacks"]

from lightcat.callback.factory import (
    is_callback_config,
    setup_callback,
    setup_list_callbacks,
)
