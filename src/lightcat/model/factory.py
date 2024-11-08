r"""Contain functions to instantiate a ``lightning.LightningModule``
object from its configuration."""

from __future__ import annotations

__all__ = ["is_model_config", "setup_model"]

import logging
from unittest.mock import Mock

from lightning import LightningModule

from lightcat.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()


logger = logging.getLogger(__name__)


def is_model_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``lightning.LightningModule``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``lightning.LightningModule`` object,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.model import is_model_config
    >>> is_model_config({"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"})
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, LightningModule)


def setup_model(module: LightningModule | dict) -> LightningModule:
    r"""Set up a ``lightning.LightningModule`` object.

    Args:
        module: The module or its configuration.

    Returns:
        The instantiated ``lightning.LightningModule`` object.

    Example usage:

    ```pycon

    >>> from lightcat.model import setup_model
    >>> model = setup_model({"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"})
    >>> model
    BoringModel(
      (layer): Linear(in_features=32, out_features=2, bias=True)
    )

    ```
    """
    if isinstance(module, dict):
        logger.info("Initializing a 'lightning.LightningModule' from its configuration... ")
        check_objectory()
        module = objectory.factory(**module)
    if not isinstance(module, LightningModule):
        logger.warning(
            f"module is not a 'lightning.LightningModule' object (received: {type(module)})"
        )
    return module
