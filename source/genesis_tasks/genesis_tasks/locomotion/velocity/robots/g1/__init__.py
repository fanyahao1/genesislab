"""G1 BeyondMimic velocity tracking task registration for GenesisLab."""

from __future__ import annotations
from math import exp

import gymnasium as gym

from genesis_tasks.locomotion.velocity import agents as velocity_agents
from .flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY
from .rough_env_cfg import G1RoughEnvCfg, G1RoughEnvCfg_PLAY

__all__ = [
    "G1FlatEnvCfg",
    "G1FlatEnvCfg_PLAY",
    "G1RoughEnvCfg",
    "G1RoughEnvCfg_PLAY",
]


# Register flat terrain tasks
gym.register(
    id="Genesis-Velocity-Flat-G1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": velocity_agents.VelocityFlatPPORunnerCfg(experiment_name="g1_flat"),
    },
)

gym.register(
    id="Genesis-Velocity-Flat-G1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
    },
)

# Register rough terrain tasks
gym.register(
    id="Genesis-Velocity-Rough-G1-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": velocity_agents.VelocityRoughPPORunnerCfg(experiment_name="g1_rough"),
    },
)

gym.register(
    id="Genesis-Velocity-Rough-G1-Play-v0",
    entry_point="genesislab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
    },
)
