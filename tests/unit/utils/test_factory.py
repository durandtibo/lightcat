from __future__ import annotations

import pytest
from torch.nn import Module, ReLU

from lightcat.testing import objectory_available
from lightcat.utils.factory import setup_object, str_target_object
from lightcat.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

##################################
#     Tests for setup_object     #
##################################


@objectory_available
@pytest.mark.parametrize("module", [ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_object(module: Module | dict) -> None:
    assert isinstance(setup_object(module), ReLU)


@objectory_available
def test_setup_object_object() -> None:
    module = ReLU()
    assert setup_object(module) is module


#######################################
#     Tests for str_target_object     #
#######################################


@objectory_available
def test_str_target_object_with_target() -> None:
    assert str_target_object({OBJECT_TARGET: "something.MyClass"}) == "something.MyClass"


@objectory_available
def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "N/A"
