r"""Contain the model creators."""

from __future__ import annotations

__all__ = ["BaseModelCreator", "ModelCreator", "is_model_creator_config", "setup_model_creator"]

from lightcat.model.creator.base import (
    BaseModelCreator,
    is_model_creator_config,
    setup_model_creator,
)
from lightcat.model.creator.vanilla import ModelCreator
