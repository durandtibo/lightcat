r"""Contain the ``lightning.LightningDataModule`` creator base class."""

from __future__ import annotations

__all__ = ["BaseDataModuleCreator", "is_datamodule_creator_config", "setup_datamodule_creator"]

import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from unittest.mock import Mock

from lightcat.utils.factory import str_target_object
from lightcat.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    import objectory
    from objectory import AbstractFactory
else:  # pragma: no cover
    objectory = Mock()
    AbstractFactory = ABCMeta


if TYPE_CHECKING:
    from lightning import LightningDataModule

logger = logging.getLogger(__name__)


class BaseDataModuleCreator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to create a
    ``lightning.LightningDataModule``.

    Note that it is not the unique approach to create a
    ``lightning.LightningDataModule``. Feel free to use other
    approaches if this approach does not fit your needs.

    Example usage:

    ```pycon

    >>> from lightcat.datamodule.creator import DataModuleCreator
    >>> creator = DataModuleCreator(
    ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"}
    ... )
    >>> creator
    DataModuleCreator(
      (_target_): lightning.pytorch.demos.boring_classes.BoringDataModule
    )
    >>> datamodule = creator.create()
    >>> datamodule
    <lightning.pytorch.demos.boring_classes.BoringDataModule object at 0x...>

    ```
    """

    @abstractmethod
    def create(self) -> LightningDataModule:
        r"""Create a ``lightning.LightningDataModule`` object.

        Returns:
            The created ``lightning.LightningDataModule`` object.

        Example usage:

        ```pycon

        >>> from lightcat.datamodule.creator import DataModuleCreator
        >>> creator = DataModuleCreator(
        ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"}
        ... )
        >>> datamodule = creator.create()
        >>> datamodule
        <lightning.pytorch.demos.boring_classes.BoringDataModule object at 0x...>

        ```
        """


def is_datamodule_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataModuleCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseDataModuleCreator`` object.

    Example usage:

    ```pycon

    >>> from lightcat.datamodule.creator import is_datamodule_creator_config
    >>> is_datamodule_creator_config(
    ...     {
    ...         "_target_": "lightcat.datamodule.creator.DataModuleCreator",
    ...         "datamodule": {
    ...             "_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"
    ...         },
    ...     }
    ... )
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, BaseDataModuleCreator)


def setup_datamodule_creator(creator: BaseDataModuleCreator | dict) -> BaseDataModuleCreator:
    r"""Set up the ``lightning.LightningDataModule`` creator.

    The ``lightning.LightningDataModule`` creator is instantiated from
    its configuration by using the ``BaseDataModuleCreator`` factory
    function.

    Args:
        creator: The ``lightning.LightningDataModule`` creator or its
            configuration.

    Returns:
        The instantiated ``lightning.LightningDataModule`` creator.

    Example usage:

    ```pycon

    >>> from lightcat.datamodule.creator import setup_datamodule_creator
    >>> creator = setup_datamodule_creator(
    ...     {
    ...         "_target_": "lightcat.datamodule.creator.DataModuleCreator",
    ...         "datamodule": {
    ...             "_target_": "lightning.pytorch.demos.boring_classes.BoringDataModule"
    ...         },
    ...     }
    ... )
    >>> creator
    DataModuleCreator(
      (_target_): lightning.pytorch.demos.boring_classes.BoringDataModule
    )

    ```
    """
    if isinstance(creator, dict):
        logger.info(
            f"Initializing a 'LightningDataModule' creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        check_objectory()
        creator = BaseDataModuleCreator.factory(**creator)
    if not isinstance(creator, BaseDataModuleCreator):
        logger.warning(f"creator is not a 'BaseDataModuleCreator' (received: {type(creator)})")
    return creator
