r"""Contain the ``lightning.Trainer`` creator base class."""

from __future__ import annotations

__all__ = ["TrainerCreator"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from lightcat.trainer.creator.base import BaseTrainerCreator
from lightcat.trainer.factory import setup_trainer

if TYPE_CHECKING:
    from lightning import Trainer

logger = logging.getLogger(__name__)


class TrainerCreator(BaseTrainerCreator):
    r"""Create a ``lightning.Trainer`` object.

    Args:
        trainer: The ``lightning.Trainer`` or its configuration.

    Example usage:

    ```pycon

    >>> from lightcat.trainer.creator import TrainerCreator
    >>> creator = TrainerCreator({"_target_": "lightning.Trainer"})
    >>> creator
    TrainerCreator(
      (_target_): lightning.Trainer
    )
    >>> trainer = creator.create()
    >>> trainer
    <lightning.pytorch.trainer.trainer.Trainer object at ...>

    ```
    """

    def __init__(self, trainer: Trainer | dict) -> None:
        self._trainer = trainer

    def __repr__(self) -> str:
        args = repr_mapping(self._trainer) if isinstance(self._trainer, dict) else self._trainer
        return f"{self.__class__.__qualname__}(\n  {repr_indent(args)}\n)"

    def __str__(self) -> str:
        args = str_mapping(self._trainer) if isinstance(self._trainer, dict) else self._trainer
        return f"{self.__class__.__qualname__}(\n  {str_indent(args)}\n)"

    def create(self) -> Trainer:
        logger.info("Creating 'Trainer'...")
        return setup_trainer(trainer=self._trainer)
