"""
Simple space and spec utilities for GenesisLab.

These helpers are intentionally minimal and independent of any particular RL
library. They provide enough structure to describe batched observation and
action tensors while keeping dependencies light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class TensorSpec:
    """Specification for a batched tensor space.

    The `shape` field does not include the leading environment dimension. All
    GenesisLab tensors are expected to have shape `(num_envs, *shape)` at
    runtime.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    low: Optional[torch.Tensor] = None
    high: Optional[torch.Tensor] = None

    def make_batch(self, num_envs: int) -> torch.Tensor:
        """Allocate an empty batch tensor matching this spec."""
        return torch.empty((num_envs, *self.shape), dtype=self.dtype, device=self.device)


@dataclass
class DictSpec:
    """Dictionary of named tensor specs."""

    specs: dict[str, TensorSpec]

    def keys(self) -> Sequence[str]:
        return list(self.specs.keys())


__all__ = ["TensorSpec", "DictSpec"]

