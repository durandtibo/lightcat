from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch
from lightning import Trainer

from lightcat.testing import objectory_available
from lightcat.trainer import is_trainer_config, setup_trainer
from lightcat.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


#######################################
#     Tests for is_trainer_config     #
#######################################


@objectory_available
def test_is_trainer_config_true() -> None:
    assert is_trainer_config({OBJECT_TARGET: "lightning.Trainer"})


@objectory_available
def test_is_trainer_config_false() -> None:
    assert not is_trainer_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for setup_trainer     #
###################################


@objectory_available
@pytest.mark.parametrize(
    "trainer",
    [Trainer(), {OBJECT_TARGET: "lightning.Trainer"}],
)
def test_setup_trainer(trainer: Trainer | dict) -> None:
    assert isinstance(setup_trainer(trainer), Trainer)


@objectory_available
def test_setup_trainer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_trainer({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity)
        assert caplog.messages


def test_setup_trainer_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_trainer({OBJECT_TARGET: "torch.nn.ReLU"})
