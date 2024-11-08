r"""Contain functions to instantiate a
``lightning.pytorch.callbacks.Callback`` object from its
configuration."""

from __future__ import annotations

__all__ = ["is_callback_config", "setup_callback", "setup_list_callbacks"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from lightning import Callback

from lightcat.utils.imports import check_objectory, is_objectory_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()


logger = logging.getLogger(__name__)


def is_callback_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``lightning.pytorch.callbacks.Callback``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``lightning.pytorch.callbacks.Callback`` object,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.callback import is_callback_config
    >>> is_callback_config(
    ...     {"_target_": "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"}
    ... )
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, Callback)


def setup_callback(callback: Callback | dict) -> Callback:
    r"""Set up a ``lightning.pytorch.callbacks.Callback`` object.

    Args:
        callback: The callback or its configuration.

    Returns:
        The instantiated ``lightning.pytorch.callbacks.Callback``
            object.

    Example usage:

    ```pycon

    >>> from lightcat.callback import setup_callback
    >>> callback = setup_callback(
    ...     {"_target_": "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"}
    ... )
    >>> callback
    <lightning.pytorch.callbacks.early_stopping.EarlyStopping object at 0x...>

    ```
    """
    if isinstance(callback, dict):
        logger.info(
            "Initializing a 'lightning.pytorch.callbacks.Callback' from its configuration... "
        )
        check_objectory()
        callback = objectory.factory(**callback)
    if not isinstance(callback, Callback):
        logger.warning(
            "callback is not a 'lightning.pytorch.callbacks.Callback' object"
            f"(received: {type(callback)})"
        )
    return callback


def setup_list_callbacks(callbacks: Sequence[Callback | dict]) -> list[Callback]:
    r"""Set up a list of ``lightning.pytorch.callbacks.Callback``
    objects.

    Args:
        callbacks: The callbacks or their configuration.

    Returns:
        The instantiated list of
            ``lightning.pytorch.callbacks.Callback`` objects.

    Example usage:

    ```pycon

    >>> from lightcat.callback import setup_list_callbacks
    >>> callbacks = setup_list_callbacks(
    ...     [
    ...         {"_target_": "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"},
    ...         {"_target_": "lightning.pytorch.callbacks.ModelSummary"},
    ...     ]
    ... )
    >>> callbacks
    [<lightning.pytorch.callbacks.early_stopping.EarlyStopping ...>,
     <lightning.pytorch.callbacks.model_summary.ModelSummary ...>]

    ```
    """
    return [setup_callback(callback) for callback in callbacks]
