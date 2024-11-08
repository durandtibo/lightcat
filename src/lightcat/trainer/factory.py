r"""Contain functions to instantiate a ``lightning.Trainer`` object from
its configuration."""

from __future__ import annotations

__all__ = ["is_trainer_config", "setup_trainer"]

import logging
from unittest.mock import Mock

from lightning import Trainer

from lightcat.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()


logger = logging.getLogger(__name__)


def is_trainer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``lightning.Trainer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``lightning.Trainer`` object,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.trainer import is_trainer_config
    >>> is_trainer_config({"_target_": "lightning.Trainer"})
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, Trainer)


def setup_trainer(module: Trainer | dict) -> Trainer:
    r"""Set up a ``lightning.Trainer`` object.

    Args:
        module: The module or its configuration.

    Returns:
        The instantiated ``lightning.Trainer`` object.

    Example usage:

    ```pycon

    >>> from lightcat.trainer import setup_trainer
    >>> trainer = setup_trainer({"_target_": "lightning.Trainer"})
    >>> trainer
    <lightning.pytorch.trainer.trainer.Trainer ...>

    ```
    """
    if isinstance(module, dict):
        logger.info("Initializing a 'lightning.Trainer' from its configuration... ")
        check_objectory()
        module = objectory.factory(**module)
    if not isinstance(module, Trainer):
        logger.warning(f"module is not a 'lightning.Trainer' object (received: {type(module)})")
    return module
