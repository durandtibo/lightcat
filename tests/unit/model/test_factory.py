from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel

from lightcat.model import is_model_config, setup_model
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


#######################################
#     Tests for is_model_config     #
#######################################


@objectory_available
def test_is_model_config_true() -> None:
    assert is_model_config({OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringModel"})


@objectory_available
def test_is_model_config_false() -> None:
    assert not is_model_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for setup_model     #
###################################


@objectory_available
@pytest.mark.parametrize(
    "module",
    [BoringModel(), {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringModel"}],
)
def test_setup_model(module: LightningModule | dict) -> None:
    assert isinstance(setup_model(module), BoringModel)


@objectory_available
def test_setup_model_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_model({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity)
        assert caplog.messages


def test_setup_model_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_model({OBJECT_TARGET: "torch.nn.ReLU"})
