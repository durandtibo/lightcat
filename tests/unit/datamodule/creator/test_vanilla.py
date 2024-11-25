from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from lightning.pytorch.demos.boring_classes import BoringDataModule

from lightcat.datamodule.creator.vanilla import DataModuleCreator
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


MODEL_OR_CONFIG = [
    BoringDataModule(),
    {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringDataModule"},
]

#######################################
#     Tests for DataModuleCreator     #
#######################################


@pytest.mark.parametrize("datamodule", MODEL_OR_CONFIG)
def test_datamodule_creator_repr(datamodule: LightningModule | dict) -> None:
    assert repr(DataModuleCreator(datamodule)).startswith("DataModuleCreator(")


@pytest.mark.parametrize("datamodule", MODEL_OR_CONFIG)
def test_datamodule_creator_str(datamodule: LightningModule | dict) -> None:
    assert str(DataModuleCreator(datamodule)).startswith("DataModuleCreator(")


@objectory_available
@pytest.mark.parametrize("datamodule", MODEL_OR_CONFIG)
def test_datamodule_creator_create(datamodule: LightningModule | dict) -> None:
    assert isinstance(DataModuleCreator(datamodule).create(), BoringDataModule)


def test_datamodule_creator_create_object() -> None:
    datamodule = BoringDataModule()
    assert DataModuleCreator(datamodule).create() is datamodule
