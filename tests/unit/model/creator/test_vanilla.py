from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from lightning.pytorch.demos.boring_classes import BoringModel

from lightcat.model.creator.vanilla import ModelCreator
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


MODEL_OR_CONFIG = [
    BoringModel(),
    {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringModel"},
]

##################################
#     Tests for ModelCreator     #
##################################


@pytest.mark.parametrize("model", MODEL_OR_CONFIG)
def test_model_creator_repr(model: LightningModule | dict) -> None:
    assert repr(ModelCreator(model)).startswith("ModelCreator(")


@pytest.mark.parametrize("model", MODEL_OR_CONFIG)
def test_model_creator_str(model: LightningModule | dict) -> None:
    assert str(ModelCreator(model)).startswith("ModelCreator(")


@objectory_available
@pytest.mark.parametrize("model", MODEL_OR_CONFIG)
def test_model_creator_create(model: LightningModule | dict) -> None:
    assert isinstance(ModelCreator(model).create(), BoringModel)


def test_model_creator_create_object() -> None:
    model = BoringModel()
    assert ModelCreator(model).create() is model
