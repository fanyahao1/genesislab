"""Reward functions for simple Go2 task.

These functions align with genesis-forge's reward functions.
Each of these should return a float tensor with the reward value for each environment, in the shape (num_envs,).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def base_height(
    env: "ManagerBasedRlEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize base height away from target, using the L2 squared kernel.

    Args:
        env: The environment instance.
        target_height: The target height to penalize the base height away from.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    base_pos = entity.data.root_pos_w
    return torch.square(base_pos[:, 2] - target_height)


def command_tracking_lin_vel(
    env: "ManagerBasedRlEnv",
    command: torch.Tensor = None,
    command_name: str = None,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for tracking commanded linear velocity (xy axes).

    Args:
        env: The environment instance.
        command: The commanded XY linear velocity in the shape (num_envs, 2). If None, will get from command_manager.
        command_name: Name of the command term to get command from. Used if command is None.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    if command is None:
        if command_name is None:
            command_name = "base_velocity"
        command = env.command_manager.get_command(command_name)
    
    entity = env.entities[asset_cfg.entity_name]
    linear_vel = entity.data.root_lin_vel_w
    
    lin_vel_error = torch.sum(torch.square(command[:, :2] - linear_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / 0.25)


def command_tracking_ang_vel(
    env: "ManagerBasedRlEnv",
    commanded_ang_vel: torch.Tensor = None,
    command_name: str = None,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw).

    Args:
        env: The environment instance.
        commanded_ang_vel: The commanded angular velocity in the shape (num_envs, 1) or (num_envs,). If None, will get from command_manager.
        command_name: Name of the command term to get command from. Used if commanded_ang_vel is None.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    if commanded_ang_vel is None:
        if command_name is None:
            command_name = "base_velocity"
        command = env.command_manager.get_command(command_name)
        commanded_ang_vel = command[:, 2]
    
    entity = env.entities[asset_cfg.entity_name]
    angular_vel = entity.data.root_ang_vel_w
    
    if commanded_ang_vel.dim() == 1:
        commanded_ang_vel = commanded_ang_vel.unsqueeze(-1)
    
    ang_vel_error = torch.square(commanded_ang_vel.squeeze(-1) - angular_vel[:, 2])
    return torch.exp(-ang_vel_error / 0.25)


def lin_vel_z_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize z axis base linear velocity.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    linear_vel = entity.data.root_lin_vel_w
    return torch.square(linear_vel[:, 2])


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    actions = env.action_manager.action
    last_actions = env.action_manager.prev_action
    if last_actions is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(last_actions - actions), dim=1)


def dof_similar_to_default(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize joint poses far away from default pose.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").

    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    entity = env.entities[asset_cfg.entity_name]
    dof_pos = entity.data.joint_pos
    default_pos = entity.data.default_joint_pos
    
    return torch.sum(torch.abs(dof_pos - default_pos), dim=1)
