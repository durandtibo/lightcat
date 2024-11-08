r"""Contain object factory utility functions."""

from __future__ import annotations

__all__ = ["setup_object", "str_target_object"]

import logging
from typing import TypeVar
from unittest.mock import Mock

from lightcat.utils.imports import is_objectory_available

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()

logger = logging.getLogger(__name__)

T = TypeVar("T")


def setup_object(obj_or_config: T | dict) -> T:
    r"""Set up an object from its configuration.

    Args:
        obj_or_config: The object or its configuration.

    Returns:
        The instantiated object.

    Example usage:

    ```pycon

    >>> from lightcat.utils.factory import setup_object
    >>> linear = setup_object(
    ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
    ... )
    >>> linear
    Linear(in_features=4, out_features=6, bias=True)
    >>> setup_object(linear)  # Do nothing because the module is already instantiated
    Linear(in_features=4, out_features=6, bias=True)

    ```
    """
    if isinstance(obj_or_config, dict):
        logger.info(
            f"Initializing {str_target_object(obj_or_config)} object from its configuration... "
        )
        return objectory.factory(**obj_or_config)
    return obj_or_config


def str_target_object(config: dict) -> str:
    r"""Get a string that indicates the target object in the config.

    Args:
        config: A config using the ``objectory`` library.
            This dict is expected to have a key ``'_target_'`` to
            indicate the target object.

    Returns:
        str: A string with the target object.

    Example usage:

    ```pycon

    >>> from lightcat.utils.factory import str_target_object
    >>> str_target_object({"_target_": "something.MyClass"})
    something.MyClass
    >>> str_target_object({})
    N/A

    ```
    """
    return config.get(objectory.OBJECT_TARGET, "N/A")
