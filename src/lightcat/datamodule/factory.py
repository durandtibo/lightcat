r"""Contain functions to instantiate a ``lightning.LightningDataModule``
object from its configuration."""

from __future__ import annotations

__all__ = ["is_datamodule_config", "setup_datamodule"]

import logging
from unittest.mock import Mock

from lightning import LightningDataModule

from lightcat.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()


logger = logging.getLogger(__name__)


def is_datamodule_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``lightning.LightningDataModule``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``lightning.LightningDataModule`` object,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.datamodule import is_datamodule_config
    >>> is_datamodule_config(
    ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"}
    ... )
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, LightningDataModule)


def setup_datamodule(datamodule: LightningDataModule | dict) -> LightningDataModule:
    r"""Set up a ``lightning.LightningDataModule`` object.

    Args:
        datamodule: The datamodule or its configuration.

    Returns:
        The instantiated ``lightning.LightningDataModule`` object.

    Example usage:

    ```pycon

    >>> from lightcat.datamodule import setup_datamodule
    >>> datamodule = setup_datamodule(
    ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"}
    ... )
    >>> datamodule
    <lightning.pytorch.demos.boring_classes.BoringDataModule object ...>

    ```
    """
    if isinstance(datamodule, dict):
        logger.info("Initializing a 'lightning.LightningDataModule' from its configuration... ")
        check_objectory()
        datamodule = objectory.factory(**datamodule)
    if not isinstance(datamodule, LightningDataModule):
        logger.warning(
            f"datamodule is not a 'lightning.LightningDataModule' object (received: {type(datamodule)})"
        )
    return datamodule
