"""Event functions for simple Go2 task.

These functions align with genesis-forge's event functions and can be used
with the EventManager for reset, startup, and interval events.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv
    from genesislab.managers import SceneEntityCfg


def position(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | list[int],
    position: tuple[float, float, float],
    quat: tuple[float, float, float, float] = None,
    zero_velocity: bool = True,
    asset_cfg: SceneEntityCfg = None,
) -> None:
    """Reset the entity to a fixed position and (optional) rotation.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        position: The position to set the entity to.
        quat: The quaternion to set the entity to.
        zero_velocity: Whether to zero the velocity of all the entity's dofs.
                      Defaults to True. This is a safety measure after a sudden change in entity pose.
        asset_cfg: Configuration for the asset entity. Defaults to None (uses "robot").
    """
    if asset_cfg is None:
        from genesislab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    
    entity_obj = env._binding.entities[asset_cfg.entity_name]
    
    # Handle env_ids: convert to tensor or handle slice(None)
    if env_ids is None or env_ids == slice(None):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, list):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    elif isinstance(env_ids, slice):
        # Handle slice by converting to tensor
        if env_ids == slice(None):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            start = env_ids.start if env_ids.start is not None else 0
            stop = env_ids.stop if env_ids.stop is not None else env.num_envs
            step = env_ids.step if env_ids.step is not None else 1
            env_ids = torch.arange(start, stop, step, device=env.device, dtype=torch.long)
    
    # Convert position to tensor
    pos_tensor = torch.tensor(position, device=env.device, dtype=torch.float32)
    pos_tensor = pos_tensor.unsqueeze(0).expand(len(env_ids), -1)
    
    # Set position
    entity_obj.set_pos(pos_tensor, envs_idx=env_ids, zero_velocity=zero_velocity)
    
    # Set quaternion if provided
    if quat is not None:
        quat_tensor = torch.tensor(quat, device=env.device, dtype=torch.float32)
        quat_tensor = quat_tensor.unsqueeze(0).expand(len(env_ids), -1)
        entity_obj.set_quat(quat_tensor, envs_idx=env_ids, zero_velocity=zero_velocity)
    
    # Zero DOF velocities if requested
    if zero_velocity:
        entity_obj.zero_all_dofs_velocity(envs_idx=env_ids)
