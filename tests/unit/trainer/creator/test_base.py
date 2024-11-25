from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from lightning import Trainer

from lightcat.testing import objectory_available
from lightcat.trainer.creator import (
    TrainerCreator,
    is_trainer_creator_config,
    setup_trainer_creator,
)
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


###############################################
#     Tests for is_trainer_creator_config     #
###############################################


@objectory_available
def test_is_trainer_creator_config_true() -> None:
    assert is_trainer_creator_config(
        {
            OBJECT_TARGET: "lightcat.trainer.creator.TrainerCreator",
            "trainer": {OBJECT_TARGET: "lightning.Trainer"},
        }
    )


@objectory_available
def test_is_trainer_creator_config_false() -> None:
    assert not is_trainer_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


###########################################
#     Tests for setup_trainer_creator     #
###########################################


@objectory_available
@pytest.mark.parametrize(
    "trainer",
    [
        TrainerCreator(Trainer()),
        {
            OBJECT_TARGET: "lightcat.trainer.creator.TrainerCreator",
            "trainer": {OBJECT_TARGET: "lightning.Trainer"},
        },
    ],
)
def test_setup_trainer_creator(trainer: LightningModule | dict) -> None:
    assert isinstance(setup_trainer_creator(trainer), TrainerCreator)


@objectory_available
def test_setup_trainer_creator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_trainer_creator({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity
        )
        assert caplog.messages


def test_setup_trainer_creator_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_trainer_creator({OBJECT_TARGET: "torch.nn.ReLU"})
