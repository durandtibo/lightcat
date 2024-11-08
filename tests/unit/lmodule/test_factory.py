from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel

from lightcat.lmodule import is_lmodule_config, setup_lmodule
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


#######################################
#     Tests for is_lmodule_config     #
#######################################


@objectory_available
def test_is_lmodule_config_true() -> None:
    assert is_lmodule_config({OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringModel"})


@objectory_available
def test_is_lmodule_config_false() -> None:
    assert not is_lmodule_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for setup_lmodule     #
###################################


@objectory_available
@pytest.mark.parametrize(
    "module",
    [BoringModel(), {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringModel"}],
)
def test_setup_lmodule(module: LightningModule | dict) -> None:
    assert isinstance(setup_lmodule(module), BoringModel)


@objectory_available
def test_setup_lmodule_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_lmodule({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity)
        assert caplog.messages


def test_setup_lmodule_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_lmodule({OBJECT_TARGET: "torch.nn.ReLU"})
