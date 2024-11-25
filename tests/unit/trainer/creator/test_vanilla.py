from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from lightning import Trainer

from lightcat.testing import objectory_available
from lightcat.trainer.creator.vanilla import TrainerCreator
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


MODEL_OR_CONFIG = [Trainer(), {OBJECT_TARGET: "lightning.Trainer"}]

####################################
#     Tests for TrainerCreator     #
####################################


@pytest.mark.parametrize("trainer", MODEL_OR_CONFIG)
def test_trainer_creator_repr(trainer: LightningModule | dict) -> None:
    assert repr(TrainerCreator(trainer)).startswith("TrainerCreator(")


@pytest.mark.parametrize("trainer", MODEL_OR_CONFIG)
def test_trainer_creator_str(trainer: LightningModule | dict) -> None:
    assert str(TrainerCreator(trainer)).startswith("TrainerCreator(")


@objectory_available
@pytest.mark.parametrize("trainer", MODEL_OR_CONFIG)
def test_trainer_creator_create(trainer: LightningModule | dict) -> None:
    assert isinstance(TrainerCreator(trainer).create(), Trainer)


def test_trainer_creator_create_object() -> None:
    trainer = Trainer()
    assert TrainerCreator(trainer).create() is trainer
