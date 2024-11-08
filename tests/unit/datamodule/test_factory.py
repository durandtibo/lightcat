from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringDataModule

from lightcat.datamodule import is_datamodule_config, setup_datamodule
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import LightningDataModule

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


##########################################
#     Tests for is_datamodule_config     #
##########################################


@objectory_available
def test_is_datamodule_config_true() -> None:
    assert is_datamodule_config(
        {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringDataModule"}
    )


@objectory_available
def test_is_datamodule_config_false() -> None:
    assert not is_datamodule_config({OBJECT_TARGET: "torch.nn.Identity"})


######################################
#     Tests for setup_datamodule     #
######################################


@objectory_available
@pytest.mark.parametrize(
    "datamodule",
    [
        BoringDataModule(),
        {OBJECT_TARGET: "lightning.pytorch.demos.boring_classes.BoringDataModule"},
    ],
)
def test_setup_datamodule(datamodule: LightningDataModule | dict) -> None:
    assert isinstance(setup_datamodule(datamodule), BoringDataModule)


@objectory_available
def test_setup_datamodule_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_datamodule({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity)
        assert caplog.messages


def test_setup_datamodule_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_datamodule({OBJECT_TARGET: "torch.nn.ReLU"})
