r"""Contain functions to instantiate a
``lightning.pytorch.callbacks.Callback`` object from its
configuration."""

from __future__ import annotations

__all__ = ["is_callback_config", "setup_callback"]

import logging
from unittest.mock import Mock

from lightning import Callback

from lightcat.utils.imports import check_objectory, is_objectory_available

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
    >>> is_callback_config({"_target_": "torch.nn.Identity"})
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
        The instantiated ``lightning.pytorch.callbacks.Callback`` object.

    Example usage:

    ```pycon

    >>> from lightcat.callback import setup_callback
    >>> linear = setup_callback(
    ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
    ... )
    >>> linear
    Linear(in_features=4, out_features=6, bias=True)

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
