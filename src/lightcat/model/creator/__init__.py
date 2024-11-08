r"""Contain the model creators."""

from __future__ import annotations

__all__ = ["BaseModelCreator", "is_model_creator_config", "setup_model_creator", "ModelCreator"]

from lightcat.model.creator.base import (
    BaseModelCreator,
    is_model_creator_config,
    setup_model_creator,
)
from lightcat.model.creator.vanilla import ModelCreator
