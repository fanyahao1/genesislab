"""Termination functions for simple Go2 task.

These functions align with genesis-forge's termination functions.
Each of these should return a boolean tensor indicating which environments should terminate, in the tensor shape (num_envs,).
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from genesislab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def timeout(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Terminate the environment if the episode length exceeds the maximum episode length.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    if env.cfg.episode_length_s is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    max_episode_length = int(env.cfg.episode_length_s / env.cfg.scene.dt)
    return env.episode_length_buf > max_episode_length


def bad_orientation(
    env: "ManagerBasedRlEnv",
    limit_angle: float = 40.0,
    asset_cfg: SceneEntityCfg = None,
    grace_steps: int = 0,
) -> torch.Tensor:
    """Terminate the environment if the robot is tipping over too much.

    This function uses projected gravity to detect when the robot has tilted
    beyond a safe threshold. When the robot is perfectly upright, projected
    gravity should be [0, 0, -1] in the body frame. As the robot tilts,
    the x,y components increase, indicating roll and pitch angles.

    Args:
        env: The environment instance.
        limit_angle: Maximum allowed tilt angle in degrees (default: 40 degrees).
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").
        grace_steps: Number of steps at episode start to ignore tilt detection (default: 0)
                     This gives the robot a chance to stabilize before tilt detection is active.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    
    in_grace_period = env.episode_length_buf <= grace_steps
    
    entity = env.entities[asset_cfg.entity_name]
    projected_gravity = entity.data.projected_gravity_b
    
    projected_gravity_xy = projected_gravity[:, :2]
    tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)
    
    tilt_angle = torch.asin(torch.clamp(tilt_magnitude, max=0.99))
    
    return (~in_grace_period) & (tilt_angle > math.radians(limit_angle))
