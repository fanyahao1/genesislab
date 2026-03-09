"""Observation functions for simple Go2 task.

These functions align with genesis-forge's observation functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def angle_velocity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """The angular velocity of the entity's base link, in the entity's local frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs, 3) containing angular velocity [x, y, z].
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    return entity.data.root_ang_vel_b


def linear_velocity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """The linear velocity of the entity's base link, in the entity's local frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs, 3) containing linear velocity [x, y, z].
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    return entity.data.root_lin_vel_b


def projected_gravity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """The projected gravity of the entity's base link, in the entity's local frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs, 3) containing projected gravity [x, y, z].
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    return entity.data.projected_gravity_b


def dof_position(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """The position of the entity's DOFs.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs, num_dofs) containing DOF positions.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    return entity.data.joint_pos


def dof_velocity(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """The velocity of the entity's DOFs.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs, num_dofs) containing DOF velocities.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    return entity.data.joint_vel


def actions(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """The most current step actions.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs, action_dim) containing current actions.
    """
    return env.action_manager.action
