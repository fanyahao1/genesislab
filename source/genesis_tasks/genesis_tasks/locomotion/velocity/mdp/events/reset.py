"""Reset-related event functions for velocity locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesislab.managers import SceneEntityCfg

from .utils import (
    resolve_env_ids,
    sample_range,
    sample_range_dict,
    euler_xyz_to_quat,
    quat_mul,
)

if TYPE_CHECKING:
    from genesislab.envs import ManagerBasedRlEnv


def reset_root_state_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the asset root state to a random pose and velocity within given ranges."""
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    entity_name = asset_cfg.entity_name
    pos_w, quat_w, lin_vel_w, ang_vel_w = env.scene.querier.get_root_state(entity_name)
    pos_w = pos_w[env_ids].clone()
    quat_w = quat_w[env_ids].clone()
    lin_vel_w = lin_vel_w[env_ids].clone()
    ang_vel_w = ang_vel_w[env_ids].clone()

    device = env.device
    num_envs = env_ids.numel()

    pose_offsets = sample_range_dict(
        pose_range,
        keys=("x", "y", "z", "roll", "pitch", "yaw"),
        num_envs=num_envs,
        device=device,
    )
    pos_offsets = pose_offsets[:, :3]
    rot_offsets = pose_offsets[:, 3:]

    pos_w = pos_w + pos_offsets

    if not torch.allclose(rot_offsets, torch.zeros_like(rot_offsets)):
        roll, pitch, yaw = rot_offsets.unbind(dim=-1)
        delta_quat = euler_xyz_to_quat(roll, pitch, yaw)
        quat_w = quat_w / torch.norm(quat_w, dim=-1, keepdim=True)
        delta_quat = delta_quat / torch.norm(delta_quat, dim=-1, keepdim=True)
        quat_w = quat_mul(quat_w, delta_quat)

    vel_offsets = sample_range_dict(
        velocity_range,
        keys=("x", "y", "z", "roll", "pitch", "yaw"),
        num_envs=num_envs,
        device=device,
    )
    lin_offsets = vel_offsets[:, :3]
    ang_offsets = vel_offsets[:, 3:]

    lin_vel_w = lin_vel_w + lin_offsets
    ang_vel_w = ang_vel_w + ang_offsets

    env.scene.controller.set_root_state(
        entity_name=entity_name,
        position=pos_w,
        quaternion=quat_w,
        linear_velocity=lin_vel_w,
        angular_velocity=ang_vel_w,
        env_ids=env_ids,
    )


def reset_joints_by_scale(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joints by scaling default joint positions/velocities."""
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    entity = env.entities[asset_cfg.entity_name]
    data = entity.data

    default_pos = data.default_joint_pos[env_ids].clone()
    default_vel = data.default_joint_vel[env_ids].clone()

    device = env.device

    pos_scale = sample_range(position_range[0], position_range[1], default_pos.shape, device=device)
    vel_scale = sample_range(velocity_range[0], velocity_range[1], default_vel.shape, device=device)

    joint_pos = default_pos * pos_scale
    joint_vel = default_vel * vel_scale

    env.scene.controller.set_joint_positions(asset_cfg.entity_name, joint_pos, env_ids=env_ids)
    env.scene.controller.set_joint_velocities(asset_cfg.entity_name, joint_vel, env_ids=env_ids)


__all__ = [
    "reset_root_state_uniform",
    "reset_joints_by_scale",
]

