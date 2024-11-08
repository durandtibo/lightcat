r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_karbonn",
    "check_objectory",
    "check_torchmetrics",
    "is_karbonn_available",
    "is_objectory_available",
    "is_torchmetrics_available",
    "karbonn_available",
    "objectory_available",
    "torchmetrics_available",
]

from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available, package_available

if TYPE_CHECKING:
    from collections.abc import Callable


###################
#     karbonn     #
###################


def is_karbonn_available() -> bool:
    r"""Indicate if the ``karbonn`` package is installed or not.

    Returns:
        ``True`` if ``karbonn`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import is_karbonn_available
    >>> is_karbonn_available()

    ```
    """
    return package_available("karbonn")


def check_karbonn() -> None:
    r"""Check if the ``karbonn`` package is installed.

    Raises:
        RuntimeError: if the ``karbonn`` package is not installed.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import check_karbonn
    >>> check_karbonn()

    ```
    """
    if not is_karbonn_available():
        msg = (
            "'karbonn' package is required but not installed. "
            "You can install 'karbonn' package with the command:\n\n"
            "pip install karbonn\n"
        )
        raise RuntimeError(msg)


def karbonn_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``karbonn``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``karbonn`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import karbonn_available
    >>> @karbonn_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_karbonn_available)


#####################
#     objectory     #
#####################


def is_objectory_available() -> bool:
    r"""Indicate if the ``objectory`` package is installed or not.

    Returns:
        ``True`` if ``objectory`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import is_objectory_available
    >>> is_objectory_available()

    ```
    """
    return package_available("objectory")


def check_objectory() -> None:
    r"""Check if the ``objectory`` package is installed.

    Raises:
        RuntimeError: if the ``objectory`` package is not installed.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import check_objectory
    >>> check_objectory()

    ```
    """
    if not is_objectory_available():
        msg = (
            "'objectory' package is required but not installed. "
            "You can install 'objectory' package with the command:\n\n"
            "pip install objectory\n"
        )
        raise RuntimeError(msg)


def objectory_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``objectory``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``objectory`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import objectory_available
    >>> @objectory_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_objectory_available)


########################
#     torchmetrics     #
########################


def is_torchmetrics_available() -> bool:
    r"""Indicate if the ``torchmetrics`` package is installed or not.

    Returns:
        ``True`` if ``torchmetrics`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import is_torchmetrics_available
    >>> is_torchmetrics_available()

    ```
    """
    return package_available("torchmetrics")


def check_torchmetrics() -> None:
    r"""Check if the ``torchmetrics`` package is installed.

    Raises:
        RuntimeError: if the ``torchmetrics`` package is not installed.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import check_torchmetrics
    >>> check_torchmetrics()

    ```
    """
    if not is_torchmetrics_available():
        msg = (
            "'torchmetrics' package is required but not installed. "
            "You can install 'torchmetrics' package with the command:\n\n"
            "pip install scikit-learn\n"
        )
        raise RuntimeError(msg)


def torchmetrics_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if
    ``torchmetrics`` package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torchmetrics`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from lightcat.utils.imports import torchmetrics_available
    >>> @torchmetrics_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_torchmetrics_available)
