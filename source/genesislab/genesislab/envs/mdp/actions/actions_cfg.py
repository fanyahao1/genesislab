"""IsaacLab-style action term configuration classes for GenesisLab.

For now we only expose the joint position action config used by locomotion
tasks, but the pattern matches IsaacLab's ``actions_cfg.py`` so additional
action types can be added later.
"""

from __future__ import annotations

from dataclasses import MISSING

from genesislab.managers.action_manager import ActionTermCfg
from genesislab.utils.configclass import configclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

@configclass
class JointActionCfg(ActionTermCfg):
    """Base configuration for joint-space action terms.

    This mirrors IsaacLab's :class:`JointActionCfg` and defines the common
    affine transform parameters applied to raw actions:

    .. math::

        action = offset + scale * input
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float = 1.0
    """Scalar scale factor for the action. Defaults to 1.0."""

    offset: float = 0.0
    """Scalar offset for the action. Defaults to 0.0."""

    preserve_order: bool = False
    """Whether to preserve the order of joint names in the action output."""


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for joint position action term."""

    asset_name: str = "robot"
    """Name of the asset entity to control."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offset."""

    def build(self, env: "ManagerBasedRlEnv"):
        """Build the joint position action term from this config.

        This mirrors the build pattern from the velocity task's previous
        JointPositionActionCfg while reusing the generic joint action
        implementation under ``genesislab.envs.mdp.actions``.
        """
        # Ensure base ActionTermCfg field is set for ActionManager.
        self.entity_name = self.asset_name
        from .joint_actions import JointPositionAction

        return JointPositionAction(cfg=self, env=env)
