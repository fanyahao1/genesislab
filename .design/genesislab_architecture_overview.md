---
alwaysApply: false
---

## GenesisLab Architecture Overview

This document summarizes the current GenesisLab runtime architecture and how it aligns
with the IsaacLab / mjlab manager-based design, using the actual layout under
`source/genesislab/genesislab/` as reference.

### Layered Structure

From bottom to top the layers are:

- **Genesis Engine (external)**
  - `genesis.Scene`, `Simulator`, solvers, entities, sensors.
- **Engine Binding Layer** (`engine/`)
  - `engine/scene_builder.py`: build a `genesis.Scene` and entities from config.
  - `engine/entity_indexing.py`: joint/link/DOF indexing utilities.
  - `engine/state_api.py`: high-level batched state queries (root pose, joint states, etc.).
  - `engine/genesis_binding.py`: thin binding that owns a `Scene`, entities and indexing and
    provides an RL-centric API (`step_physics`, `reset_envs`, apply joint targets, etc.).
- **Components Layer** (`components/`)
  - `components/scene/*`, `components/terrains/*`, `components/assets/*`, `components/sensors/*`,
    `components/actuators/*`: reusable Genesis-side building blocks for constructing scenes and
    tasks (robots, terrains, sensors, actuators).
- **Environment Core Layer** (`envs/`)
  - `envs/base_env.py`: `ManagerBasedGenesisEnv`, the Genesis counterpart to
    IsaacLab's `ManagerBasedRLEnv` / mjlab's `ManagerBasedRlEnv`.
  - Responsibilities:
    - Own a `GenesisBinding` instance.
    - Construct and own all managers (action, observation, reward, termination, command, etc.).
    - Maintain batched episode buffers (`obs`, `rew`, `terminated`, `truncated`, `episode_length`).
    - Implement the step pipeline with decimation:
      - action manager → physics (binding) → termination/reward managers → command/events →
        observation manager.
- **Manager Layer** (`managers/`)
  - `manager_base.py`: `ManagerTermBaseCfg`, `ManagerTermBase`, `ManagerBase`.
  - `action_manager.py`, `observation_manager.py`, `reward_manager.py`,
    `termination_manager.py`, `command_manager.py`,
    plus optional `event_manager.py`, `curriculum_manager.py`, `metrics_manager.py`.
  - Responsibilities:
    - Follow IsaacLab's manager pattern (term configs with `func` + `params`, `reset` and
      `compute` APIs).
    - Depend only on the environment interface (`ManagerBasedGenesisEnv`) and the binding/state
      APIs, never directly on Genesis solvers.
- **Task / Config Layer** (separate package: `source/genesis_tasks/genesis_tasks/`)
  - Task-specific env configs and MDP functions (e.g. `go2/cfg.py`, `go2/env.py`, `go2/mdp/*`).
  - Map high-level task configuration (episode length, decimation, rewards, terminations, etc.)
    into:
    - Engine binding configuration (robots, terrains, dt, substeps).
    - Manager configs (observation groups, reward terms, termination terms, command terms, etc.).

### Key Design Principles

- **Engine Isolation**
  - All direct interaction with Genesis `Scene` and solvers is contained inside the
    engine binding and state API modules under `engine/`.
  - Managers and task code see only `ManagerBasedGenesisEnv` and high-level binding/state
    methods, not solver internals.

- **IsaacLab-Compatible Manager Pattern**
  - Manager types and term configs are structurally aligned with IsaacLab/mjlab:
    - A base manager, base term config, and specific managers for observations, rewards,
      terminations, actions, commands, curriculum and events.
  - Term configs support function-based and class-based implementations, with optional
    `reset(env_ids)` methods for persistent state.

- **Vectorized Environments**
  - A single `GenesisBinding` + `Scene.build(n_envs)` holds `num_envs` batched environments.
  - All managers and the environment core treat the leading tensor dimension as
    the environment batch dimension.

- **Config-Driven Construction**
  - Task packages (e.g. `genesis_tasks`) define configs using a `configclass` system and
    map them into:
    - Binding/scene construction parameters.
    - Manager term configurations.

For background analysis and the rationale behind this layering, see:

- `mjlab_architecture.md`
- `genesis_engine_analysis.md`
- `comparative_analysis.md`
- `genesislab_preliminary_design.md`
- `implementation_roadmap.md`

