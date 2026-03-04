"""
GenesisLab: A manager-based training framework built on top of the Genesis engine.

This package provides:

- A thin engine binding over `gs.Scene` (genesislab.core.binding)
- A manager-based vectorized RL environment core (genesislab.core.env_core)
- Engine-agnostic managers for actions, observations, rewards, and terminations

High-level entry points for external code will be added incrementally as the
framework matures. For now, users are expected to construct configurations and
environments explicitly via the core and configs subpackages.
"""

from .core.env_core import ManagerBasedGenesisEnv

__all__ = ["ManagerBasedGenesisEnv"]

