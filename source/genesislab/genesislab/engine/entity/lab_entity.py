"""Entity data view for GenesisLab environments.

This module provides a data view abstraction similar to IsaacLab's ArticulationData,
allowing MDP code to access entity state through a clean, typed interface like
`env.entities["go2"].data.joint_pos` instead of calling `env.get_joint_state("go2")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv
    from genesislab.engine.gstype import KinematicEntity
    from genesislab.engine.assets.robot import Robot
    from genesislab.components.actuators import ActuatorBase

from .lab_entity_data import LabEntityData

class LabEntity:
    """Wrapper for a Genesis entity with data view access.

    This class wraps the underlying Genesis entity and provides a clean
    interface for accessing entity state through the `data` property,
    similar to IsaacLab's Articulation class.
    """

    _raw_entity: "KinematicEntity"
    _robot_asset: "Robot"
    _data: "LabEntityData" = None
    _actuators: Dict[str, "ActuatorBase"] = None

    def __init__(self, env: "ManagerBasedGenesisEnv", entity_name: str, raw_entity: "KinematicEntity", robot_asset: "Robot" = None):
        """Initialize the entity wrapper.

        Args:
            env: The environment instance. Can be None if EntityData is not needed immediately.
            entity_name: Name of the entity.
            raw_entity: The underlying Genesis entity object.
            robot_asset: Optional Robot asset wrapper for name resolution (for articulated robots).
        """
        self._env = env
        self._entity_name = entity_name
        self._raw_entity = raw_entity
        self._robot_asset = robot_asset
        self._actuators = {}

    @property
    def name(self) -> str:
        """Name of the entity."""
        return self._entity_name

    @property
    def data(self) -> LabEntityData:
        """Data view for accessing entity state."""
        if self._data is None:
            # Pass self so that LabEntityData can access the raw entity directly
            self._data = LabEntityData(self._env, self)
        return self._data

    @property
    def raw_entity(self) -> KinematicEntity:
        """Access to the underlying Genesis entity (for advanced use cases)."""
        return self._raw_entity

    @property
    def robot_asset(self) -> Robot:
        """Access to the Robot asset wrapper (for name resolution, if available)."""
        return self._robot_asset

    @property
    def n_links(self):
        return self._raw_entity.n_links
    
    @property
    def raw_link_names(self):
        return self._robot_asset.body_normalizer.raw_names
    
    @property
    def link_names(self):
        return self._robot_asset.body_normalizer.normalized_names
    
    @property
    def n_joints(self):
        return self._raw_entity.n_joints
    
    @property
    def raw_joint_names(self):
        return self._robot_asset.joint_normalizer.raw_names
    
    @property
    def joint_names(self):
        return self._robot_asset.joint_normalizer.normalized_names
    
    @property
    def actuators(self) -> Dict[str, "ActuatorBase"]:
        """Dictionary of actuators for this entity, keyed by actuator name."""
        return self._actuators
    
    @property
    def dof_indices(self) -> torch.Tensor | None:
        """DOF indices for controlled joints.
        
        Returns:
            Tensor of DOF indices from the robot asset, or None if not available.
        """
        return self._robot_asset.dof_indices

    # ------------------------------------------------------------------
    # IsaacLab-style state write APIs used by imitation/tracking MDP
    # ------------------------------------------------------------------

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Write joint positions and velocities into the underlying Genesis entity.

        Args:
            joint_pos: Tensor of shape (N, num_joints) where N == len(env_ids).
            joint_vel: Tensor of shape (N, num_joints) where N == len(env_ids).
            env_ids: Indices of environments to update. If None, updates all envs.
        """
        import torch as _torch  # local import to avoid circulars when TYPE_CHECKING

        # Infer DOF layout from entity
        dofs_pos_full = self._raw_entity.get_dofs_position()
        _, num_dofs_full = dofs_pos_full.shape
        base_offset = 6 if num_dofs_full > 6 else 0
        joint_slice = slice(base_offset, num_dofs_full)

        # Normalize env_ids to a 1D LongTensor
        if env_ids is None:
            env_ids_tensor = _torch.arange(self._env.num_envs, device=self._env.device, dtype=_torch.long)
        elif isinstance(env_ids, _torch.Tensor):
            env_ids_tensor = env_ids.to(device=self._env.device, dtype=_torch.long)
        else:
            env_ids_tensor = _torch.tensor(env_ids, device=self._env.device, dtype=_torch.long)

        # Ensure tensors live on the correct device
        joint_pos = joint_pos.to(device=self._env.device)
        joint_vel = joint_vel.to(device=self._env.device)

        # Set joint DOFs only (exclude base)
        self._raw_entity.set_dofs_position(
            joint_pos,
            dofs_idx_local=joint_slice,
            envs_idx=env_ids_tensor,
            zero_velocity=False,
        )
        self._raw_entity.set_dofs_velocity(
            joint_vel,
            dofs_idx_local=joint_slice,
            envs_idx=env_ids_tensor,
        )

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Write root (base) pose and velocities into the underlying Genesis entity.

        Args:
            root_state: Tensor of shape (N, 13) with layout
                [pos(3), quat(4 in [x,y,z,w]), lin_vel(3), ang_vel(3)].
            env_ids: Indices of environments to update. If None, updates all envs.
        """
        import torch as _torch

        # Normalize env_ids to a 1D LongTensor
        if env_ids is None:
            env_ids_tensor = _torch.arange(self._env.num_envs, device=self._env.device, dtype=_torch.long)
        elif isinstance(env_ids, _torch.Tensor):
            env_ids_tensor = env_ids.to(device=self._env.device, dtype=_torch.long)
        else:
            env_ids_tensor = _torch.tensor(env_ids, device=self._env.device, dtype=_torch.long)

        root_state = root_state.to(device=self._env.device)

        # Split components: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        pos = root_state[:, 0:3]
        quat = root_state[:, 3:7]
        lin_vel = root_state[:, 7:10]
        ang_vel = root_state[:, 10:13]

        # Base generalized coordinates: first 7 DOFs (pos 3 + quat 4)
        base_q = _torch.cat([pos, quat], dim=-1)
        self._raw_entity.set_dofs_position(
            base_q,
            dofs_idx_local=slice(0, 7),
            envs_idx=env_ids_tensor,
            zero_velocity=False,
        )

        # Base velocities: first 6 DOFs (lin_vel 3 + ang_vel 3)
        base_vel = _torch.cat([lin_vel, ang_vel], dim=-1)
        self._raw_entity.set_dofs_velocity(
            base_vel,
            dofs_idx_local=slice(0, 6),
            envs_idx=env_ids_tensor,
        )
