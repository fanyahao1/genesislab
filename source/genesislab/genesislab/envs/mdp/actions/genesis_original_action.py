"""Genesis original action implementation.

This action follows the genesis-forge PositionActionManager pattern, providing
affine transformations (scale and offset) for joint position actions with
support for per-joint configuration via regex patterns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from genesislab.components.actuators import ActuatorBase
from genesislab.managers.action_manager import ActionTerm, ActionTermCfg
from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

def _ensure_dof_pattern(value: float | dict[str, float] | None) -> dict[str, float] | None:
    """Convert a value to a DOF pattern dict if needed.
    
    Args:
        value: A scalar value or dict mapping joint name patterns to values.
        
    Returns:
        A dict mapping patterns to values, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {".*": float(value)}
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items()}
    raise TypeError(f"Expected float or dict, got {type(value)}")


class GenesisOriginalAction(ActionTerm):
    """Action term that maps normalized actions to joint position targets.
    
    This follows the genesis-forge PositionActionManager pattern, providing:
    - Affine transformations: position = offset + scale * action
    - Per-joint configuration via regex patterns
    - Automatic clipping to joint limits
    - Support for default joint positions as offset
    
    Uses Genesis PD control directly via control_dofs_position.
    """

    cfg: "GenesisOriginalActionCfg"

    def __init__(self, cfg: "GenesisOriginalActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        self._entity_name = cfg.entity_name
        entity_obj = env.scene.entities[self._entity_name]

        self._actuators: dict[str, ActuatorBase] = entity_obj.actuators
        if self._actuators:
            self._action_dim = sum(actuator.num_joints for actuator in self._actuators.values())
        else:
            num_joints = 0
            for joint in entity_obj.joints:
                if hasattr(joint, "name") and joint.name.lower() == "base":
                    continue
                if hasattr(joint, "dof_start") and joint.dof_start is not None:
                    num_joints += 1
            self._action_dim = num_joints

        # Buffers.
        self._raw_action = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._targets = torch.zeros_like(self._raw_action)

        # Convert config values to pattern dicts
        self._offset_cfg = _ensure_dof_pattern(cfg.offset)
        self._scale_cfg = _ensure_dof_pattern(cfg.scale)
        # For GenesisOriginalAction, clip must be a tuple (uniform clipping for entire action)
        # Validate clip type
        if cfg.clip is not None and not isinstance(cfg.clip, tuple):
            raise TypeError(
                f"GenesisOriginalAction clip must be tuple[float, float] or None, got {type(cfg.clip)}"
            )
        self._use_default_offset = cfg.use_default_offset

        # Validate: cannot use both use_default_offset and explicit offset
        if self._use_default_offset and self._offset_cfg is not None and self._offset_cfg.get(".*") != 0.0:
            raise ValueError("Cannot set both use_default_offset=True and offset != 0.0")

        # Initialize scale and offset tensors (will be set in _build_values)
        self._scale_values: torch.Tensor = None
        self._offset_values: torch.Tensor = None

        # Get joint names for pattern matching
        self._joint_names: list[str] = []
        self._joint_name_to_index: dict[str, int] = {}
        self._dofs_idx: list[int] = []
        self._build_joint_mapping()

        # Build scale and offset values
        self._build_values()
        
        # Build clip bounds (cached in base class, tuple clip for uniform clipping)
        self._build_clip_bounds(self._action_dim, cfg.clip, joint_names=None)

    def _build_joint_mapping(self) -> None:
        """Build mapping from joint names to action indices and DOF indices."""
        entity_obj = self._env.scene.entities[self._entity_name]
        
        if self._actuators:
            idx = 0
            for actuator in self._actuators.values():
                for joint_name in actuator.joint_names:
                    joint = entity_obj.get_joint(joint_name)
                    if joint is not None and hasattr(joint, "dof_start") and joint.dof_start is not None:
                        self._joint_names.append(joint_name)
                        self._joint_name_to_index[joint_name] = idx
                        dof_start = joint.dof_start
                        dof_count = getattr(joint, "dof_count", 1)
                        self._dofs_idx.extend(range(dof_start, dof_start + dof_count))
                        idx += 1
        else:
            idx = 0
            for joint in entity_obj.joints:
                if hasattr(joint, "name") and joint.name.lower() == "base":
                    continue
                if hasattr(joint, "dof_start") and joint.dof_start is not None:
                    joint_name = joint.name
                    self._joint_names.append(joint_name)
                    self._joint_name_to_index[joint_name] = idx
                    dof_start = joint.dof_start
                    dof_count = getattr(joint, "dof_count", 1)
                    self._dofs_idx.extend(range(dof_start, dof_start + dof_count))
                    idx += 1

    def _build_values(self) -> None:
        """Build scale, offset, and clip value tensors from configs."""
        num_actions = self._action_dim
        
        # Initialize with defaults
        self._scale_values = torch.ones(num_actions, device=self.device)
        self._offset_values = torch.zeros(num_actions, device=self.device)
        
        # Get default joint positions for offset if needed
        if self._use_default_offset:
            entity = self._env.entities[self._entity_name]
            default_joint_pos = entity.data.default_joint_pos
            
            # Map default positions to action indices
            if default_joint_pos.shape[-1] >= num_actions:
                self._offset_values[:] = default_joint_pos[0, :num_actions].clone()
            else:
                self._offset_values[:default_joint_pos.shape[-1]] = default_joint_pos[0].clone()
        
        # Apply scale config
        if self._scale_cfg is not None:
            self._apply_pattern_to_tensor(self._scale_cfg, self._scale_values, default_value=1.0)
        
        # Apply offset config (if not using default offset)
        if not self._use_default_offset and self._offset_cfg is not None:
            self._apply_pattern_to_tensor(self._offset_cfg, self._offset_values, default_value=0.0)


    def _apply_pattern_to_tensor(
        self,
        pattern_dict: dict[str, float],
        output: torch.Tensor,
        default_value: float = 0.0,
    ) -> None:
        """Apply pattern dict values to output tensor.
        
        Args:
            pattern_dict: Dict mapping joint name patterns to values.
            output: Tensor to fill with values.
            default_value: Default value for unmatched joints.
        """
        is_set = [False] * len(self._joint_names)
        
        for pattern, value in pattern_dict.items():
            found = False
            for i, joint_name in enumerate(self._joint_names):
                if not is_set[i] and re.match(f"^{pattern}$", joint_name):
                    idx = self._joint_name_to_index[joint_name]
                    output[idx] = float(value)
                    is_set[i] = True
                    found = True
            if not found and pattern != ".*":
                # Only warn for non-wildcard patterns
                import warnings
                warnings.warn(
                    f"GenesisOriginalAction: No joints matched pattern '{pattern}'. "
                    f"Available joints: {self._joint_names}"
                )


    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store and process raw actions into joint position targets.
        
        Applies affine transformation: target = offset + scale * action
        Then clips to joint limits.
        """
        if actions.shape != self._raw_action.shape:
            if actions.shape[-1] == self._action_dim and actions.shape[0] == 1:
                actions = actions.expand_as(self._raw_action)
            else:
                raise ValueError(
                    f"Invalid action shape for GenesisOriginalAction: expected "
                    f"{self._raw_action.shape}, got {actions.shape}."
                )

        self._raw_action[:] = actions
        
        # Validate actions
        if torch.isnan(actions).any():
            import warnings
            warnings.warn("GenesisOriginalAction: NaN actions received!")
        if torch.isinf(actions).any():
            import warnings
            warnings.warn("GenesisOriginalAction: Infinite actions received!")
        
        # Apply affine transformation: target = offset + scale * action
        self._targets[:] = self._offset_values.unsqueeze(0) + self._scale_values.unsqueeze(0) * actions
        
        # Apply clipping using cached bounds from base class
        self._targets[:] = self._apply_clip(self._targets)

    def apply_actions(self) -> None:
        entity = self._env.scene.entities[self._entity_name]
        entity._raw_entity.control_dofs_position(self._targets, self._dofs_idx if self._dofs_idx else None)


@configclass
class GenesisOriginalActionCfg(ActionTermCfg):
    class_type: type = GenesisOriginalAction  # Will be set to GenesisOriginalAction below
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    use_default_offset: bool = True
    clip: tuple[float, float] = None
    """Clip range for the entire action tensor (uniform clipping).
    
    Only tuple is allowed for GenesisOriginalAction. For per-joint clipping,
    use JointPositionAction with dict clip.
    """
