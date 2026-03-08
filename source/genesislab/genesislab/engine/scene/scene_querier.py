"""Scene querier for state queries.

This module provides the SceneQuerier class for querying entity states
from the scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .lab_scene import LabScene


class SceneQuerier:
    """Helper class for querying entity state from the scene."""
    
    def __init__(self, scene: "LabScene"):
        """Initialize the scene querier.
        
        Args:
            scene: Reference to the LabScene instance.
        """
        self._scene = scene
    
    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity.
        
        Args:
            entity_name: Name of the entity.
        
        Returns:
            Tuple of (positions, velocities) tensors of shape (num_envs, num_dofs).
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = lab_entity.dof_indices
        
        if dof_indices is not None:
            positions = entity.get_dofs_position(dof_indices)
            velocities = entity.get_dofs_velocity(dof_indices)
        else:
            positions = entity.get_dofs_position()
            velocities = entity.get_dofs_velocity()
        
        return positions, velocities
    
    def get_root_state(
        self, entity_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocity for an entity.
        
        Args:
            entity_name: Name of the entity.
        
        Returns:
            Tuple of (position, quaternion, linear_velocity, angular_velocity) tensors.
            Positions and velocities have shape (num_envs, 3).
            Quaternions have shape (num_envs, 4).
        """
        lab_entity = self._scene.entities[entity_name]
        entity = lab_entity.raw_entity
        
        # Get root pose
        position = entity.get_pos()
        quaternion = entity.get_quat()
        
        # Get root velocity
        linear_velocity = entity.get_vel()
        angular_velocity = entity.get_ang()
        
        return position, quaternion, linear_velocity, angular_velocity
    
    def get_body_positions(self, entity_name: str) -> torch.Tensor:
        """Get positions of all bodies/links for an entity.
        
        Args:
            entity_name: Name of the entity.
        
        Returns:
            Tensor of shape ``(num_envs, num_links, 3)`` containing the world-frame
            positions of all links/bodies.
        """
        lab_entity = self._scene.entities[entity_name]
        return lab_entity.raw_entity.get_links_pos()
