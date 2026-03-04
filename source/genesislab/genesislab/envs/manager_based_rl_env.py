"""Manager-based reinforcement learning environment for GenesisLab.

This module provides a thin, IsaacLab-style wrapper around
``ManagerBasedGenesisEnv`` that serves as the canonical RL environment base
class for manager-based workflows.

- ``ManagerBasedRlEnvCfg`` is a small specialization of
  :class:`genesislab.components.entities.env_cfg.ManagerBasedGenesisEnvCfg` with RL-centric
  documentation.
- ``ManagerBasedRlEnv`` subclasses :class:`ManagerBasedGenesisEnv` and keeps
  the same reset/step semantics as used throughout GenesisLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv, ManagerBasedGenesisEnvCfg
from genesislab.envs.common import VecEnvObs, VecEnvStepReturn
from genesislab.utils.configclass import configclass


class ManagerBasedRlEnv(ManagerBasedGenesisEnv):
    """Manager-based RL environment for Genesis.

    This class is a thin alias over :class:`ManagerBasedGenesisEnv` that
    emphasizes RL usage (rewards/terminations) and exposes the same
    vectorized API:

    - ``reset(seed, env_ids, options) -> (VecEnvObs, info)``
    - ``step(action) -> VecEnvStepReturn``
    """

    cfg: ManagerBasedRlEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, device: str = "cuda", **_: Any) -> None:
        """Initialize the manager-based RL environment.

        Args:
            cfg: RL environment configuration.
            device: Device to use for tensors ("cuda" or "cpu").
            **_: Additional keyword arguments reserved for future use.
        """
        super().__init__(cfg=cfg, device=device)
        # Set a default render fps in metadata for viewers/wrappers.
        self.metadata["render_fps"] = 1.0 / self.step_dt

    # The ``reset`` and ``step`` methods are inherited directly from
    # :class:`ManagerBasedGenesisEnv` and already use VecEnvObs / VecEnvStepReturn.

@configclass
class ManagerBasedRlEnvCfg(ManagerBasedGenesisEnvCfg):
    """Configuration for a manager-based RL environment on Genesis.

    This class directly reuses :class:`ManagerBasedGenesisEnvCfg` but clarifies
    RL-specific semantics such as ``episode_length_s`` and ``is_finite_horizon``.

    Key fields inherited from :class:`ManagerBasedGenesisEnvCfg`:

    - ``scene``: :class:`genesislab.components.entities.scene_cfg.SceneCfg` describing the
      Genesis scene (robots, terrain, sensors, simulation options).
    - ``decimation``: number of physics steps per environment step.
    - ``rewards`` / ``terminations`` / ``commands`` / ``observations`` /
      ``actions``: manager configurations.
    - ``episode_length_s``: episode duration in seconds (optional).
    - ``is_finite_horizon``: whether timeouts are treated as true terminals.
    """