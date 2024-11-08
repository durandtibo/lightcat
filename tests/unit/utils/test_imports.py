from __future__ import annotations

from unittest.mock import patch

import pytest

from lightcat.utils.imports import (
    check_karbonn,
    check_objectory,
    check_torchmetrics,
    is_karbonn_available,
    is_objectory_available,
    is_torchmetrics_available,
    karbonn_available,
    objectory_available,
    torchmetrics_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


###################
#     karbonn     #
###################


def test_check_karbonn_with_package() -> None:
    with patch("lightcat.utils.imports.is_karbonn_available", lambda: True):
        check_karbonn()


def test_check_karbonn_without_package() -> None:
    with (
        patch("lightcat.utils.imports.is_karbonn_available", lambda: False),
        pytest.raises(RuntimeError, match="'karbonn' package is required but not installed."),
    ):
        check_karbonn()


def test_is_karbonn_available() -> None:
    assert isinstance(is_karbonn_available(), bool)


def test_karbonn_available_with_package() -> None:
    with patch("lightcat.utils.imports.is_karbonn_available", lambda: True):
        fn = karbonn_available(my_function)
        assert fn(2) == 44


def test_karbonn_available_without_package() -> None:
    with patch("lightcat.utils.imports.is_karbonn_available", lambda: False):
        fn = karbonn_available(my_function)
        assert fn(2) is None


def test_karbonn_available_decorator_with_package() -> None:
    with patch("lightcat.utils.imports.is_karbonn_available", lambda: True):

        @karbonn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_karbonn_available_decorator_without_package() -> None:
    with patch("lightcat.utils.imports.is_karbonn_available", lambda: False):

        @karbonn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("lightcat.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("lightcat.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("lightcat.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("lightcat.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("lightcat.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


########################
#     torchmetrics     #
########################


def test_check_torchmetrics_with_package() -> None:
    with patch("lightcat.utils.imports.is_torchmetrics_available", lambda: True):
        check_torchmetrics()


def test_check_torchmetrics_without_package() -> None:
    with (
        patch("lightcat.utils.imports.is_torchmetrics_available", lambda: False),
        pytest.raises(RuntimeError, match="'torchmetrics' package is required but not installed."),
    ):
        check_torchmetrics()


def test_is_torchmetrics_available() -> None:
    assert isinstance(is_torchmetrics_available(), bool)


def test_torchmetrics_available_with_package() -> None:
    with patch("lightcat.utils.imports.is_torchmetrics_available", lambda: True):
        fn = torchmetrics_available(my_function)
        assert fn(2) == 44


def test_torchmetrics_available_without_package() -> None:
    with patch("lightcat.utils.imports.is_torchmetrics_available", lambda: False):
        fn = torchmetrics_available(my_function)
        assert fn(2) is None


def test_torchmetrics_available_decorator_with_package() -> None:
    with patch("lightcat.utils.imports.is_torchmetrics_available", lambda: True):

        @torchmetrics_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torchmetrics_available_decorator_without_package() -> None:
    with patch("lightcat.utils.imports.is_torchmetrics_available", lambda: False):

        @torchmetrics_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
