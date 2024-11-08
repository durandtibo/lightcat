from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary

from lightcat.callback import is_callback_config, setup_callback, setup_list_callbacks
from lightcat.testing import objectory_available
from lightcat.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from lightning import Callback

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


########################################
#     Tests for is_callback_config     #
########################################


@objectory_available
def test_is_callback_config_true() -> None:
    assert is_callback_config(
        {OBJECT_TARGET: "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"}
    )


@objectory_available
def test_is_callback_config_false() -> None:
    assert not is_callback_config({OBJECT_TARGET: "torch.nn.Identity"})


####################################
#     Tests for setup_callback     #
####################################


@objectory_available
@pytest.mark.parametrize(
    "callback",
    [
        EarlyStopping(monitor="loss"),
        {OBJECT_TARGET: "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"},
    ],
)
def test_setup_callback(callback: Callback | dict) -> None:
    assert isinstance(setup_callback(callback), EarlyStopping)


@objectory_available
def test_setup_callback_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_callback({OBJECT_TARGET: "torch.nn.Identity"}), torch.nn.Identity)
        assert caplog.messages


def test_setup_callback_object_no_objectory() -> None:
    with (
        patch("lightcat.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_callback({OBJECT_TARGET: "torch.nn.ReLU"})


##########################################
#     Tests for setup_list_callbacks     #
##########################################


@objectory_available
def test_setup_list_callbacks() -> None:
    callbacks = setup_list_callbacks(
        [
            {OBJECT_TARGET: "lightning.pytorch.callbacks.EarlyStopping", "monitor": "loss"},
            {OBJECT_TARGET: "lightning.pytorch.callbacks.ModelSummary"},
        ]
    )
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], EarlyStopping)
    assert isinstance(callbacks[1], ModelSummary)


def test_setup_list_callbacks_empty() -> None:
    assert setup_list_callbacks([]) == []
