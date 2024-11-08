r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "cuda_available",
    "distributed_available",
    "gloo_available",
    "karbonn_available",
    "nccl_available",
    "objectory_available",
    "torchmetrics_available",
    "two_gpus_available",
]

from lightcat.testing.fixtures import (
    cuda_available,
    distributed_available,
    gloo_available,
    karbonn_available,
    nccl_available,
    objectory_available,
    torchmetrics_available,
    two_gpus_available,
)
