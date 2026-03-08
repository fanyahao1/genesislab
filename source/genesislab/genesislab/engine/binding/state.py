"""State query methods for GenesisBinding."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .binding import GenesisBinding

import torch


class StateQuerier:
    """Helper class for querying entity state."""

    def __init__(self, binding: "GenesisBinding"):
        self._binding = binding

    def get_joint_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint positions and velocities for an entity.
        Returns:
            Tuple of (positions, velocities) tensors of shape (num_envs, num_dofs).
        """
        lab_entity = self._binding.entities[entity_name]
        entity = lab_entity.raw_entity
        dof_indices = self._binding._dof_indices.get(entity_name)
        positions = entity.get_dofs_position(dof_indices)
        velocities = entity.get_dofs_velocity(dof_indices)
        return positions, velocities

    def get_root_state(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root pose and velocity for an entity.
        Returns:
            Tuple of (position, quaternion, linear_velocity, angular_velocity) tensors.
            Positions and velocities have shape (num_envs, 3).
            Quaternions have shape (num_envs, 4).
        """
        lab_entity = self._binding.entities[entity_name]
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
        Returns:
            Tensor of shape ``(num_envs, num_links, 3)`` containing the world-frame
            positions of all links/bodies.
        """
        lab_entity = self._binding.entities[entity_name]
        return lab_entity.raw_entity.get_links_pos()
