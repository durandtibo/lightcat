r"""Contain the ``lightning.Trainer`` creator base class."""

from __future__ import annotations

__all__ = ["BaseTrainerCreator", "is_trainer_creator_config", "setup_trainer_creator"]

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
    from lightning import Trainer

logger = logging.getLogger(__name__)


class BaseTrainerCreator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to create a ``lightning.Trainer``.

    Note that it is not the unique approach to create a
    ``lightning.Trainer``. Feel free to use other approaches
    if this approach does not fit your needs.

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

    @abstractmethod
    def create(self) -> Trainer:
        r"""Create a ``lightning.Trainer``.

        Returns:
            The created ``lightning.Trainer``.

        Example usage:

        ```pycon

        >>> from lightcat.trainer.creator import TrainerCreator
        >>> creator = TrainerCreator({"_target_": "lightning.Trainer"})
        >>> trainer = creator.create()
        >>> trainer
        <lightning.pytorch.trainer.trainer.Trainer object at ...>

        ```
        """


def is_trainer_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTrainerCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseTrainerCreator`` object.

    Example usage:

    ```pycon

    >>> from lightcat.trainer.creator import is_trainer_creator_config
    >>> is_trainer_creator_config(
    ...     {
    ...         "_target_": "lightcat.trainer.creator.TrainerCreator",
    ...         "trainer": {"_target_": "lightning.Trainer"},
    ...     }
    ... )
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, BaseTrainerCreator)


def setup_trainer_creator(creator: BaseTrainerCreator | dict) -> BaseTrainerCreator:
    r"""Set up the trainer creator.

    The trainer creator is instantiated from its configuration by using
    the ``BaseTrainerCreator`` factory function.

    Args:
        creator: The trainer creator or its configuration.

    Returns:
        The instantiated trainer creator.

    Example usage:

    ```pycon

    >>> from lightcat.trainer.creator import setup_trainer_creator
    >>> creator = setup_trainer_creator(
    ...     {
    ...         "_target_": "lightcat.trainer.creator.TrainerCreator",
    ...         "trainer": {"_target_": "lightning.Trainer"},
    ...     }
    ... )
    >>> creator
    TrainerCreator(
      (_target_): lightning.Trainer
    )

    ```
    """
    if isinstance(creator, dict):
        logger.info(
            f"Initializing a 'Trainer' creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        check_objectory()
        creator = BaseTrainerCreator.factory(**creator)
    if not isinstance(creator, BaseTrainerCreator):
        logger.warning(f"creator is not a 'BaseTrainerCreator' (received: {type(creator)})")
    return creator
