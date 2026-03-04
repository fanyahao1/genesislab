"""
Thin engine binding over Genesis `Scene` for GenesisLab.

This module owns the Genesis scene and provides a minimal, RL-centric API for:

- Building a batched scene from configuration
- Resetting robot and environment state deterministically
- Stepping physics, optionally with decimation handled at a higher level
- Reading and writing batched robot state

All direct interaction with the Genesis engine should go through this layer or
the `genesislab.engine.*` helpers it uses internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import genesis as gs
import torch

from genesislab.components.entities.binding_cfg import SceneBindingCfg
from genesislab.engine.entity_indexing import RobotIndexInfo
from genesislab.engine.scene_builder import SceneBuildError, build_scene_from_cfg


@dataclass
class GenesisBinding:
    """Binding between GenesisLab and a Genesis Scene.

    This object should be constructed once per environment instance and then
    reused for the lifetime of that instance.
    """

    cfg: SceneBindingCfg
    scene: gs.Scene
    robots: Dict[str, RobotIndexInfo]

    def __post_init__(self) -> None:
        if self.cfg.n_envs <= 0:
            raise ValueError("SceneBindingCfg.n_envs must be positive.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cfg(cls, cfg: SceneBindingCfg) -> "GenesisBinding":
        """Build a Genesis Scene and binding from a binding configuration."""
        scene, robots = build_scene_from_cfg(cfg)
        scene.build(n_envs=cfg.n_envs)
        return cls(cfg=cfg, scene=scene, robots=robots)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        """Number of parallel environments managed by this binding."""
        return int(self.cfg.n_envs)

    @property
    def device(self) -> torch.device:
        """Device used by Genesis tensors."""
        return gs.device

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def set_pd_gains(self, robot_name: str, kp: float, kd: float) -> None:
        """Set PD gains for the actuated DOFs of a given robot."""
        info = self.robots[robot_name]
        num_dofs = info.motor_dof_idx.numel()
        info.entity.set_dofs_kp([kp] * num_dofs, info.motor_dof_idx)
        info.entity.set_dofs_kv([kd] * num_dofs, info.motor_dof_idx)

    def apply_joint_targets(
        self,
        robot_name: str,
        position_targets: torch.Tensor,
    ) -> None:
        """Apply joint position targets for a robot across all environments.

        Parameters
        ----------
        robot_name:
            Logical robot name from the binding configuration.
        position_targets:
            Tensor of shape (num_envs, num_actuated_dofs). The ordering of
            columns must match the order of joint names in the corresponding
            RobotBindingCfg.
        """
        info = self.robots[robot_name]
        if position_targets.shape[0] != self.num_envs:
            raise ValueError(
                f"position_targets has invalid batch size {position_targets.shape[0]}, "
                f"expected {self.num_envs}."
            )
        if position_targets.shape[1] != info.motor_dof_idx.numel():
            raise ValueError(
                f"position_targets has invalid dof dimension {position_targets.shape[1]}, "
                f"expected {info.motor_dof_idx.numel()}."
            )

        # The slice interface is currently hard-coded to start at 0; we rely on
        # `motor_dof_idx` to select the correct DOFs for control.
        info.entity.control_dofs_position(position_targets[:, :], info.motor_dof_idx)

    # ------------------------------------------------------------------
    # Stepping and reset
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        """Advance the scene by a number of physics steps."""
        if n_steps <= 0:
            return
        for _ in range(n_steps):
            self.scene.step()

    def reset_envs(self, robot_name: str, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset specified environments for a given robot.

        This method only resets the robot pose and velocity; higher-level
        logic (e.g., reward buffers, commands) is handled by the environment
        core and managers.
        """
        info = self.robots[robot_name]
        entity = info.entity

        # For simplicity, we reset to the initial configuration stored by Genesis.
        # If custom initial states are desired, they should be injected via a
        # more elaborate mechanism in future revisions.
        entity.set_qpos(entity.qpos0, envs_idx=env_ids, zero_velocity=True, skip_forward=True)

    # ------------------------------------------------------------------
    # Factory for a single-robot locomotion binding
    # ------------------------------------------------------------------

    @classmethod
    def build_simple_locomotion(
        cls,
        morph_file_robot: str,
        joint_names: Sequence[str],
        morph_file_plane: str,
        n_envs: int,
        dt: float,
        substeps: int,
        kp: float,
        kd: float,
    ) -> "GenesisBinding":
        """Convenience constructor for simple quadruped locomotion experiments.

        This is intended for early development and smoke tests; more advanced
        tasks should use fully-configured SceneBindingCfg instances.
        """
        from genesislab.components.entities.binding_cfg import RobotBindingCfg, SceneBindingCfg, TerrainBindingCfg

        robot_cfg = RobotBindingCfg(
            name="agent",
            morph_file=morph_file_robot,
            morph_type="urdf",
            fixed_base=False,
            joint_names=list(joint_names),
            kp=kp,
            kd=kd,
        )
        terrain_cfg = TerrainBindingCfg(
            morph_file=morph_file_plane,
            morph_type="urdf",
            fixed=True,
        )
        scene_cfg = SceneBindingCfg(
            robots=[robot_cfg],
            terrain=terrain_cfg,
            n_envs=n_envs,
            dt=dt,
            substeps=substeps,
            viewer=False,
        )
        binding = cls.from_cfg(scene_cfg)
        binding.set_pd_gains(robot_name="agent", kp=kp, kd=kd)
        return binding


__all__ = ["GenesisBinding", "SceneBuildError"]

