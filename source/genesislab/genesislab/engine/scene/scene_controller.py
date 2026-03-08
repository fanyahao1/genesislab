"""Scene controller for control and state setting.

This module provides the SceneController class for controlling entities
and setting their states (reset, joint positions, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .lab_scene import LabScene


class SceneController:
    """Helper class for controlling entities and setting states."""
    
    def __init__(self, scene: "LabScene"):
        """Initialize the scene controller.
        
        Args:
            scene: Reference to the LabScene instance.
        """
        self._scene = scene
    
    def reset(self, env_ids: torch.Tensor = None) -> None:
        """Reset specified environments to initial state.
        
        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            self._scene.gs_scene.reset()
        else:
            # Convert to list if tensor
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
            self._scene.gs_scene.reset(envs_idx=env_ids)
    
    def step(self) -> None:
        """Step the physics simulation by one timestep."""
        self._scene.gs_scene.step()
    
    def set_joint_targets(
        self, entity_name: str, targets: torch.Tensor, control_type: str = "position"
    ) -> None:
        """Set joint control targets for an entity.
        
        Args:
            entity_name: Name of the entity.
            targets: Target values of shape (num_envs, num_dofs).
            control_type: Type of control ('position', 'velocity', or 'torque').
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = self._scene._dof_indices.get(entity_name)
        
        if control_type == "position":
            if dof_indices is not None:
                entity.control_dofs_position(targets, dof_indices)
            else:
                entity.control_dofs_position(targets)
        elif control_type == "velocity":
            if dof_indices is not None:
                entity.control_dofs_velocity(targets, dof_indices)
            else:
                entity.control_dofs_velocity(targets)
        elif control_type == "torque":
            # Torque control: apply forces directly
            # Genesis uses control_dofs_force() for torque/force control
            if dof_indices is not None:
                entity.control_dofs_force(targets, dof_indices)
            else:
                entity.control_dofs_force(targets)
        else:
            raise ValueError(f"Unknown control type: {control_type}")
    
    def set_joint_positions(self, entity_name: str, positions: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Set joint positions for an entity (for reset/initialization).
        
        Args:
            entity_name: Name of the entity.
            positions: Joint positions of shape (num_envs, num_dofs) or (num_dofs,).
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = self._scene._dof_indices.get(entity_name)
        
        if env_ids is not None:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
            # Set positions for specific environments
            if dof_indices is not None:
                entity.set_dofs_position(positions, dof_indices, envs_idx=env_ids)
            else:
                entity.set_dofs_position(positions, envs_idx=env_ids)
        else:
            # Set positions for all environments
            if dof_indices is not None:
                entity.set_dofs_position(positions, dof_indices)
            else:
                entity.set_dofs_position(positions)
    
    def set_joint_velocities(self, entity_name: str, velocities: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Set joint velocities for an entity (for reset/initialization).
        
        Args:
            entity_name: Name of the entity.
            velocities: Joint velocities of shape (num_envs, num_dofs) or (num_dofs,).
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = self._scene._dof_indices.get(entity_name)
        
        if env_ids is not None:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
            # Set velocities for specific environments
            if dof_indices is not None:
                entity.set_dofs_velocity(velocities, dof_indices, envs_idx=env_ids)
            else:
                entity.set_dofs_velocity(velocities, envs_idx=env_ids)
        else:
            # Set velocities for all environments
            if dof_indices is not None:
                entity.set_dofs_velocity(velocities, dof_indices)
            else:
                entity.set_dofs_velocity(velocities)
    
    def set_root_state(
        self,
        entity_name: str,
        position: torch.Tensor = None,
        quaternion: torch.Tensor = None,
        linear_velocity: torch.Tensor = None,
        angular_velocity: torch.Tensor = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """Set root state for an entity (for reset/initialization).
        
        Args:
            entity_name: Name of the entity.
            position: Root position of shape (num_envs, 3) or (3,). If None, not set.
            quaternion: Root quaternion of shape (num_envs, 4) or (4,). If None, not set.
            linear_velocity: Linear velocity of shape (num_envs, 3) or (3,). If None, not set.
            angular_velocity: Angular velocity of shape (num_envs, 3) or (3,). If None, not set.
            env_ids: Environment indices to set. If None, sets all environments.
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        
        if env_ids is not None:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()
        
        # Set position
        if position is not None:
            if env_ids is not None:
                entity.set_pos(position, envs_idx=env_ids)
            else:
                entity.set_pos(position)
        
        # Set quaternion
        if quaternion is not None:
            if env_ids is not None:
                entity.set_quat(quaternion, envs_idx=env_ids)
            else:
                entity.set_quat(quaternion)
        
        # Set linear velocity
        if linear_velocity is not None:
            if env_ids is not None:
                entity.set_vel(linear_velocity, envs_idx=env_ids)
            else:
                entity.set_vel(linear_velocity)
        
        # Set angular velocity
        if angular_velocity is not None:
            if env_ids is not None:
                entity.set_ang(angular_velocity, envs_idx=env_ids)
            else:
                entity.set_ang(angular_velocity)
