"""
High-level state accessors built on top of the Genesis engine binding.

These helpers provide typed, batched views of common robot state such as base
pose and joint states. They accept the binding object rather than Genesis
objects directly, so that higher layers stay decoupled from the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from genesislab.engine.entity_indexing import RobotIndexInfo


@dataclass
class BaseState:
    """Batched base state for a robot."""

    pos: torch.Tensor  # (num_envs, 3)
    quat: torch.Tensor  # (num_envs, 4)
    lin_vel: torch.Tensor  # (num_envs, 3)
    ang_vel: torch.Tensor  # (num_envs, 3)


def get_base_state(index_info: RobotIndexInfo) -> BaseState:
    """Query the batched base state of a robot entity."""
    robot = index_info.entity

    pos = robot.get_pos()
    quat = robot.get_quat()
    vel = robot.get_vel()
    ang = robot.get_ang()

    assert pos.ndim == 2 and pos.shape[1] == 3
    assert quat.ndim == 2 and quat.shape[1] == 4
    assert vel.ndim == 2 and vel.shape[1] == 3
    assert ang.ndim == 2 and ang.shape[1] == 3

    return BaseState(pos=pos, quat=quat, lin_vel=vel, ang_vel=ang)


def get_joint_state(index_info: RobotIndexInfo) -> Tuple[torch.Tensor, torch.Tensor]:
    """Query joint positions and velocities for the actuated DOFs."""
    robot = index_info.entity
    idx = index_info.motor_dof_idx

    q = robot.get_dofs_position(idx)
    qd = robot.get_dofs_velocity(idx)

    return q, qd


__all__ = ["BaseState", "get_base_state", "get_joint_state"]

