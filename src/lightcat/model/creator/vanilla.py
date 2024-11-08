r"""Contain the model creator base class."""

from __future__ import annotations

__all__ = ["ModelCreator"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from lightcat.model.creator.base import BaseModelCreator
from lightcat.model.factory import setup_model

if TYPE_CHECKING:
    from lightning import LightningModule

logger = logging.getLogger(__name__)


class ModelCreator(BaseModelCreator):
    r"""Create a ``lightning.LightningModule`` object.

    Args:
        model: The model or its configuration.

    Example usage:

    ```pycon

    >>> from lightcat.model.creator import ModelCreator
    >>> creator = ModelCreator(
    ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"}
    ... )
    >>> creator
    ModelCreator(
      (_target_): lightning.pytorch.demos.boring_classes.BoringModel
    )
    >>> model = creator.create()
    >>> model
    BoringModel(
      (layer): Linear(in_features=32, out_features=2, bias=True)
    )

    ```
    """

    def __init__(self, model: LightningModule | dict) -> None:
        self._model = model

    def __repr__(self) -> str:
        args = repr_mapping(self._model) if isinstance(self._model, dict) else self._model
        return f"{self.__class__.__qualname__}(\n  {repr_indent(args)}\n)"

    def __str__(self) -> str:
        args = str_mapping(self._model) if isinstance(self._model, dict) else self._model
        return f"{self.__class__.__qualname__}(\n  {str_indent(args)}\n)"

    def create(self) -> LightningModule:
        logger.info("Creating model...")
        return setup_model(model=self._model)
