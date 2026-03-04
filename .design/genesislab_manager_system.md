---
alwaysApply: false
---

## GenesisLab Manager System Design

This document describes the GenesisLab manager system and how it aligns with the
IsaacLab / mjlab design, while using Genesis as the underlying engine.

### Core Types

- **`ManagerTermBaseCfg`** (`managers/manager_base.py`)
  - Base configuration for all manager terms (observation, reward, termination,
    action, command, curriculum, event).
  - Fields:
    - `func`: callable or class implementing the term.
    - `params`: dict of keyword arguments passed to the term.

- **`ManagerTermBase`** (`managers/manager_base.py`)
  - Base class for term implementations that need access to the environment.
  - Holds a reference to `ManagerBasedGenesisEnv` and exposes `num_envs`, `device` and `name`.

- **`ManagerBase`** (`managers/manager_base.py`)
  - Abstract base class for all managers.
  - Responsibilities:
    - Store the environment reference.
    - Call `_prepare_terms()` at construction time to build all term instances.
    - Provide `reset(env_ids)` and `get_active_iterable_terms(env_idx)` hooks.
    - Resolve common term config details via `_resolve_common_term_cfg`, including:
      - Resolving `SceneEntityCfg` references to Genesis entities via the binding.
      - Instantiating class-based term implementations once with `(cfg, env)`.

- **`SceneEntityCfg`** (`managers/scene_entity_config.py`)
  - Genesis-specific entity reference used inside term configs.
  - Fields:
    - `entity_name`: logical entity name.
    - `resolved`: engine entity handle set by `resolve(container)`.
  - Resolution:
    - Prefer `env._binding.entities` (mapping of names to entities).
    - Fallback to `env.scene` for attribute-based resolution.

### Manager Types

The following manager types mirror IsaacLab/mjlab concepts but operate on Genesis:

- **ActionManager** (`managers/action_manager.py`)
  - Aggregates multiple action terms.
  - Responsibilities:
    - Maintain batched action buffers: `action`, `prev_action`, `prev_prev_action`.
    - Split flat policy actions across terms and forward slices to each term.
    - Apply processed actions to the engine each physics substep (via the binding).

- **ObservationManager** (`managers/observation_manager.py`)
  - Organizes observation terms into groups (`ObservationGroupCfg`).
  - Responsibilities:
    - Compute observations from term functions or classes using the environment
      and binding/state APIs.
    - Optionally apply noise, scaling, clipping, delay and history.
    - Return grouped observations, typically concatenated tensors per group.
  - Short-term goal:
    - Replace residual `mjlab.utils.*` dependencies with GenesisLab local utilities
      under `utils/` while preserving the high-level API.

- **RewardManager** (`managers/reward_manager.py`)
  - Aggregates weighted reward terms (`RewardTermCfg`).
  - Responsibilities:
    - Compute per-term rewards and aggregate to a single `reward_buf`.
    - Optionally scale rewards by step duration `dt`.
    - Maintain per-term episodic sums and expose episode statistics in `reset`.

- **TerminationManager** (`managers/termination_manager.py`)
  - Evaluates termination and truncation terms (`TerminationTermCfg`).
  - Responsibilities:
    - Maintain per-term done buffers and global `terminated` / `time_outs` masks.
    - Return combined `dones` mask used by the environment to trigger resets.

- **CommandManager** (`managers/command_manager.py`)
  - Manages command signals (e.g. target velocities, poses).
  - Responsibilities:
    - Hold one or more `CommandTerm` instances.
    - Call `compute(dt)` each step to update commands according to their internal
      timers and logic.
    - Provide access to commands via `get_command(name)`.
  - Example:
    - `genesis_tasks/go2/mdp/commands.py` implements a `VelocityCommand` term
      that samples forward velocity targets over time.

- **CurriculumManager / EventManager / MetricsManager**
  - Currently largely IsaacLab/mjlab-style scaffolding.
  - Next steps:
    - Replace all references to `ManagerBasedRlEnv` and MuJoCo-specific APIs with
      `ManagerBasedGenesisEnv` and Genesis binding/state APIs.
    - Redesign event-based domain randomization hooks around Genesis solver and
      options rather than MuJoCo model fields.

### Interaction with the Environment Core

`ManagerBasedGenesisEnv` (`envs/base_env.py`) orchestrates the managers as follows:

- **Reset pipeline**
  - Determine `env_ids` to reset.
  - Use the engine binding to reset Genesis states for selected environments.
  - Reset per-env buffers in the environment (`episode_length_buf`, etc.).
  - Call `reset(env_ids)` on all managers to clear their internal state and collect
    episodic statistics.
  - Compute initial observations via `ObservationManager.compute(update_history=True)`.

- **Step pipeline**
  - `ActionManager.process_action(action)`.
  - For `decimation` substeps:
    - `ActionManager.apply_action()`.
    - `binding.step_physics(1)` to advance Genesis by one physics step.
  - Update episode counters.
  - `TerminationManager.compute()` → `terminated` / `time_outs` / overall `dones`.
  - `RewardManager.compute(dt=step_dt)`.
  - Reset any environments flagged in `dones`.
  - `CommandManager.compute(dt=step_dt)` (if active).
  - `ObservationManager.compute(update_history=True)` to produce new observations.

Managers thus remain engine-agnostic at the conceptual level and rely on the
environment plus binding/state APIs for all engine-specific operations, preserving
the IsaacLab design while targeting Genesis.

