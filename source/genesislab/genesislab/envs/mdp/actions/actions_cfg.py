"""IsaacLab-style action term configuration classes for GenesisLab.

For now we only expose the joint position action config used by locomotion
tasks, but the pattern matches IsaacLab's ``actions_cfg.py`` so additional
action types can be added later.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from genesislab.managers.action_manager import ActionTermCfg
from genesislab.utils.configclass import configclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv


from .joint_actions import JointPositionAction
from .genesis_original_action import GenesisOriginalAction

@configclass
class JointActionCfg(ActionTermCfg):
    joint_names: list[str] = MISSING

    scale: float = 1.0

    offset: float = 0.0

    preserve_order: bool = False


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for joint position action term."""

    class_type: type = JointPositionAction  # Will be set to JointPositionAction below
    """The action term class type. Set automatically."""

    use_default_offset: bool = True



@configclass
class GenesisOriginalActionCfg(ActionTermCfg):
    class_type: type = GenesisOriginalAction  # Will be set to GenesisOriginalAction below
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    use_default_offset: bool = True
    clip: tuple[float, float] | dict[str, tuple[float, float]] | None = None

