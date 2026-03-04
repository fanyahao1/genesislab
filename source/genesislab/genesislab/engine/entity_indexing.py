"""
Entity and DOF indexing utilities for the Genesis engine binding.

This module is responsible for mapping human-readable joint and entity names
to integer indices and compact tensors that higher layers can use for control
and observation. All Genesis-specific calls are contained here or in
`scene_builder.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import genesis as gs
import torch


@dataclass
class RobotIndexInfo:
    """Indexing information for a single robot entity.

    This structure is intentionally small and only exposes the pieces needed
    by the binding and managers: joint DOF indices and base link identity.
    """

    entity: gs.Entity
    """Underlying Genesis entity representing the robot."""

    motor_dof_idx: torch.Tensor
    """Tensor of DOF indices (on gs.device) for the actuated joints."""

    joint_name_to_dof_start: Dict[str, int]
    """Mapping from joint name to starting DOF index in the global DOF vector."""


def build_robot_index_info(
    robot_entity: gs.Entity,
    joint_names: Sequence[str],
) -> RobotIndexInfo:
    """Construct indexing information for a named robot entity.

    Parameters
    ----------
    robot_entity:
        The Genesis entity returned from `scene.add_entity` for the robot.
    joint_names:
        Names of joints that are actuated and should be part of the control
        DOF set.
    """
    device = gs.device
    tc_int = gs.tc_int

    joint_name_to_dof_start: Dict[str, int] = {}
    dof_starts = []
    for name in joint_names:
        joint = robot_entity.get_joint(name)
        dof_start = int(joint.dof_start)
        joint_name_to_dof_start[name] = dof_start
        dof_starts.append(dof_start)

    motor_dof_idx = torch.tensor(dof_starts, dtype=tc_int, device=device)
    return RobotIndexInfo(
        entity=robot_entity,
        motor_dof_idx=motor_dof_idx,
        joint_name_to_dof_start=joint_name_to_dof_start,
    )


__all__ = ["RobotIndexInfo", "build_robot_index_info"]

