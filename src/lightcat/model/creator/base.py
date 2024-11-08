r"""Contain the model creator base class."""

from __future__ import annotations

__all__ = ["BaseModelCreator", "is_model_creator_config", "setup_model_creator"]

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
    from lightning import LightningModule

logger = logging.getLogger(__name__)


class BaseModelCreator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to create a model.

    Note that it is not the unique approach to create a model. Feel
    free to use other approaches if this approach does not fit your
    needs.

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

    @abstractmethod
    def create(self) -> LightningModule:
        r"""Create a model on the device(s) where it should run.

        This method is responsible to register the event handlers
        associated to the model. This method is also responsible to
        move the model parameters to the device(s).

        Returns:
            The created model.

        Example usage:

        ```pycon

        >>> from lightcat.model.creator import ModelCreator
        >>> creator = ModelCreator(
        ...     {"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"}
        ... )
        >>> model = creator.create()
        >>> model
        BoringModel(
          (layer): Linear(in_features=32, out_features=2, bias=True)
        )

        ```
        """


def is_model_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseModelCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseModelCreator`` object.

    Example usage:

    ```pycon

    >>> from lightcat.model.creator import is_model_creator_config
    >>> is_model_creator_config(
    ...     {
    ...         "_target_": "lightcat.model.creator.ModelCreator",
    ...         "model": {"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"},
    ...     }
    ... )
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, BaseModelCreator)


def setup_model_creator(creator: BaseModelCreator | dict) -> BaseModelCreator:
    r"""Set up the model creator.

    The model creator is instantiated from its configuration by using
    the ``BaseModelCreator`` factory function.

    Args:
        creator: The model creator or its configuration.

    Returns:
        The instantiated model creator.

    Example usage:

    ```pycon

    >>> from lightcat.model.creator import setup_model_creator
    >>> creator = setup_model_creator(
    ...     {
    ...         "_target_": "lightcat.model.creator.ModelCreator",
    ...         "model": {"_target_": "lightning.pytorch.demos.boring_classes.BoringModel"},
    ...     }
    ... )
    >>> creator
    ModelCreator(
      (_target_): lightning.pytorch.demos.boring_classes.BoringModel
    )

    ```
    """
    if isinstance(creator, dict):
        logger.info(
            f"Initializing a model creator from its configuration... {str_target_object(creator)}"
        )
        check_objectory()
        creator = BaseModelCreator.factory(**creator)
    if not isinstance(creator, BaseModelCreator):
        logger.warning(f"creator is not a 'BaseModelCreator' (received: {type(creator)})")
    return creator
