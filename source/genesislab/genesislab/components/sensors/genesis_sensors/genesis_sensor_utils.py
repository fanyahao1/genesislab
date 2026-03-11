"""Shared utilities for Genesis-native sensor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    try:
        import genesis as gs  # type: ignore
    except Exception:  # pragma: no cover - optional at type-check time
        gs = Any  # type: ignore


def to_tensor(x: Any, device: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Best-effort conversion of Genesis sensor outputs to torch.Tensor."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    if str(t.device) != device:
        t = t.to(device)
    return t

