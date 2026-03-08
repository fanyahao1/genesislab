"""Actuator and PD gains management for GenesisBinding."""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .binding import GenesisBinding

from genesislab.components.actuators import ActuatorBase
import logging
logger = logging.getLogger(__name__)

class ActuatorManager:
    """Helper class for managing actuators and PD gains."""

    def __init__(self, binding: "GenesisBinding"):
        """Initialize the actuator manager.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def process_actuators_cfg(self) -> None:
        """Process and apply actuator configurations for robots (IsaacLab-style).

        This method processes actuator configurations from RobotCfg.actuators and:
        1. Creates actuator instances for each actuator group
        2. Sets engine kp/kv to 0 for all actuators (all actuators compute torques explicitly)
        3. All actuators compute torques and apply them via control_dofs_force()
        """

        for entity_name, robot_cfg in self._binding.cfg.robots.items():
            actuators_cfg = getattr(robot_cfg, "actuators", None)
            if actuators_cfg is None:
                continue

            lab_entity = self._binding._entities[entity_name]
            # Get raw entity for direct access
            entity = lab_entity.raw_entity
            self._binding._actuators[entity_name] = {}

            # Get Robot asset from LabEntity (required)
            robot_asset = lab_entity.robot_asset
            if robot_asset is None:
                raise RuntimeError(
                    f"Robot asset for entity '{entity_name}' is None. "
                    f"This indicates that the robot was not properly initialized with GenesisArticulationRobot."
                )
            # Get actuated joint names (filtered by Robot asset)
            raw_joint_names = robot_asset.get_actuated_joint_names(normalized=False)
            
            if not raw_joint_names:
                raise RuntimeError(f"Robot '{entity_name}': No actuated joints found. Skipping actuator processing.")

            normalized_joint_names = robot_asset.get_normalized_joint_names()

            # Get joint state to infer number of DOFs
            dof_pos, _ = self._binding.get_joint_state(entity_name)
            num_dofs = dof_pos.shape[-1]
            num_envs = dof_pos.shape[0]

            # Build joint name to DOF index mapping (using raw names for indexing)
            joint_name_to_dof_indices = {}
            for raw_joint_name in raw_joint_names:
                joint = entity.get_joint(raw_joint_name)
                if joint is not None and hasattr(joint, "dof_start") and joint.dof_start is not None:
                    dof_start = joint.dof_start
                    dof_count = getattr(joint, "dof_count", 1) if hasattr(joint, "dof_count") else 1
                    joint_name_to_dof_indices[raw_joint_name] = list(range(dof_start, dof_start + dof_count))

            # Process each actuator group
            for actuator_name, actuator_cfg in actuators_cfg.items():
                # Find matching joints using normalized names (for consistent pattern matching)
                try:
                    matched_indices, matched_normalized_names = robot_asset.match_joints(
                        actuator_cfg.joint_names_expr
                    )
                    # Convert normalized names back to raw names for indexing
                    matched_raw_names = [raw_joint_names[idx] for idx in matched_indices]
                except ValueError as e:
                    raise ValueError(
                        f"Robot '{entity_name}': Actuator '{actuator_name}': {e}\n"
                        f"Available normalized joint names: {normalized_joint_names}"
                    )

                if not matched_raw_names:
                    raise ValueError(
                        f"Robot '{entity_name}': Actuator '{actuator_name}': "
                        f"No joints matched expression {actuator_cfg.joint_names_expr}. "
                        f"Available normalized joint names: {normalized_joint_names}"
                    )

                # Resolve DOF indices for matched joints (using raw names)
                matched_dof_indices = []
                for raw_joint_name in matched_raw_names:
                    if raw_joint_name in joint_name_to_dof_indices:
                        matched_dof_indices.extend(joint_name_to_dof_indices[raw_joint_name])
                num_actuator_joints = len(matched_dof_indices)

                # Convert to tensor or slice for efficiency
                if len(matched_raw_names) == len(raw_joint_names):
                    joint_ids_tensor = slice(None)
                else:
                    joint_ids_tensor = torch.tensor(matched_indices, dtype=torch.long, device=self._binding.device)

                # Get default joint properties from entity (for now, use zeros as defaults)
                # In a full implementation, these would be read from the USD/URDF file
                default_stiffness = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_damping = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_armature = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_dynamic_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_viscous_friction = torch.zeros(num_envs, num_actuator_joints, device=self._binding.device)
                default_effort_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._binding.device)
                default_velocity_limit = torch.full((num_envs, num_actuator_joints), float('inf'), device=self._binding.device)

                # Create actuator instance (use normalized names for actuator's internal use)
                actuator: ActuatorBase = actuator_cfg.class_type(
                    cfg=actuator_cfg,
                    joint_names=matched_normalized_names,  # Use normalized names for consistency
                    joint_ids=joint_ids_tensor,
                    num_envs=num_envs,
                    device=self._binding.device,
                    stiffness=default_stiffness,
                    damping=default_damping,
                    armature=default_armature,
                    friction=default_friction,
                    dynamic_friction=default_dynamic_friction,
                    viscous_friction=default_viscous_friction,
                    effort_limit=default_effort_limit,
                    velocity_limit=default_velocity_limit,
                )

                # Store actuator instance
                self._binding._actuators[entity_name][actuator_name] = actuator

                # Set engine kp/kv to 0 for all actuators
                # All actuators compute torques explicitly and apply them via control_dofs_force()
                dof_indices_tensor = torch.tensor(matched_dof_indices, dtype=torch.long, device=self._binding.device)
                zero_kp = torch.zeros(len(matched_dof_indices), device=self._binding.device)
                zero_kd = torch.zeros(len(matched_dof_indices), device=self._binding.device)
                entity.set_dofs_kp(zero_kp, dof_indices_tensor)
                entity.set_dofs_kv(zero_kd, dof_indices_tensor)
                
                logger.info(
                    f"Robot '{entity_name}': Actuator '{actuator_name}': "
                    f"Set engine kp/kv to 0 for joints {matched_normalized_names}. "
                    f"Actuator will compute torques explicitly."
                )

