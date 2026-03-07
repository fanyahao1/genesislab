"""Simple Go2 locomotion task.

This module provides a simple Go2 locomotion task configuration that aligns
with Genesis Forge's simple Go2 example, adapted to GenesisLab's manager-based
framework.
"""

from __future__ import annotations

import gymnasium as gym

from . import agents
from .simple_env_cfg import SimpleGo2EnvCfg, SimpleGo2EnvCfg_PLAY

__all__ = [
    "SimpleGo2EnvCfg",
    "SimpleGo2EnvCfg_PLAY",
]


# Register training task
gym.register(
    id="Genesis-Forge-Simple-Go2-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_env_cfg:SimpleGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleGo2PPORunnerCfg",
    },
)

# Register play task
gym.register(
    id="Genesis-Forge-Simple-Go2-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_env_cfg:SimpleGo2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleGo2PPORunnerCfg",
    },
)
