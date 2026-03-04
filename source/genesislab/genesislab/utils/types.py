"""Common data types for GenesisLab."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

@dataclass
class ArticulationActions:
    """Data container to store articulation joint actions.

    This mirrors IsaacLab's ``ArticulationActions`` but is defined in the
    GenesisLab namespace so that actuator code does not depend on IsaacLab.
    """

    joint_positions: torch.Tensor | None = None
    joint_velocities: torch.Tensor | None = None
    joint_efforts: torch.Tensor | None = None
    joint_indices: torch.Tensor | Sequence[int] | slice | None = None


__all__ = ["ArticulationActions"]

