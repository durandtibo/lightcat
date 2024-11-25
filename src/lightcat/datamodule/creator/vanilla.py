r"""Contain the ``lightning.LightningDataModule`` creator base class."""

from __future__ import annotations

__all__ = ["DataModuleCreator"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from lightcat.datamodule.creator.base import BaseDataModuleCreator
from lightcat.datamodule.factory import setup_datamodule

if TYPE_CHECKING:
    from lightning import LightningDataModule

logger = logging.getLogger(__name__)


class DataModuleCreator(BaseDataModuleCreator):
    r"""Create a ``lightning.LightningDataModule`` object.

    Args:
        datamodule: The ``lightning.LightningDataModule`` object or
            its configuration.

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
    <lightning.pytorch.demos.boring_classes.BoringDataModule object at ...>

    ```
    """

    def __init__(self, datamodule: LightningDataModule | dict) -> None:
        self._datamodule = datamodule

    def __repr__(self) -> str:
        args = (
            repr_mapping(self._datamodule)
            if isinstance(self._datamodule, dict)
            else self._datamodule
        )
        return f"{self.__class__.__qualname__}(\n  {repr_indent(args)}\n)"

    def __str__(self) -> str:
        args = (
            str_mapping(self._datamodule)
            if isinstance(self._datamodule, dict)
            else self._datamodule
        )
        return f"{self.__class__.__qualname__}(\n  {str_indent(args)}\n)"

    def create(self) -> LightningDataModule:
        logger.info("Creating 'LightningDataModule'...")
        return setup_datamodule(datamodule=self._datamodule)
