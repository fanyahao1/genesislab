## mjlab architecture overview

This document reverse engineers the `mjlab` codebase as it exists in this repository, focusing on how it emulates IsaacLab’s manager-based environments on top of MuJoCo and `mujoco_warp`. All descriptions below are based on actual code inspection, not on IsaacLab documentation alone.

---

## 1. Folder‑level structural map

From `mjlab/src/mjlab`:

```text
mjlab/
  actuator/      # Actuation utilities and transmission logic (not exhaustively inspected yet)
  asset_zoo/     # Robot and asset definitions (e.g., Unitree robots)
  entity/        # Scene entities that wrap MuJoCo bodies / joints / actuators
  envs/          # Environment-level types and manager-based RL env implementation
  managers/      # Manager system (observation, reward, action, termination, etc.)
  rl/            # RL integration utilities (e.g., vecenv wrapper)
  scene/         # Scene configuration and MuJoCo model construction
  scripts/       # CLI utilities (training, evaluation, visualization)
  sensor/        # Sensor abstractions and sensor context
  sim/           # Simulation wrapper around MuJoCo + mujoco_warp
  tasks/         # Task-specific MDP definitions and configs (velocity, tracking, manipulation)
  terrains/      # Terrain generators and terrain entities
  utils/         # General utilities (spaces, random, buffers, noise, mujoco helpers)
  viewer/        # Online and offline visualization, including MuJoCo-based viewers
```

At a high level:

- **Engine boundary** is concentrated in `scene/` and `sim/`, which construct and step MuJoCo / mujoco_warp models, plus `entity/` and `sensor/` for per‑object behavior.
- **IsaacLab‑style environment and manager system** lives in `envs/` and `managers/`, with task‑specific term implementations in `tasks/*/mdp/`.
- **RL and vectorization layer** is provided by `envs/manager_based_rl_env.py` (vectorized manager‑based env) and `rl/vecenv_wrapper.py` (wrapper to RSL‑RL style vecenv).

---

## 2. Critical base classes and their responsibilities

### 2.1 Manager‑based RL environment

- **File**: `mjlab/envs/manager_based_rl_env.py`
- **Classes**:
  - `ManagerBasedRlEnvCfg` (dataclass)
  - `ManagerBasedRlEnv`

**ManagerBasedRlEnvCfg**

- Plain `@dataclass(kw_only=True)` (no `@configclass` here) that describes the full RL environment configuration:
  - `decimation`: number of physics substeps per environment step.
  - `scene: SceneCfg`: scene description (terrain, entities, sensors, num_envs).
  - `sim: SimulationCfg`: physics, solver, and NaN‑guard options.
  - `viewer: ViewerConfig`: rendering settings.
  - `episode_length_s`, `is_finite_horizon`, `scale_rewards_by_dt`.
  - Dictionaries of term configs:
    - `observations: dict[str, ObservationGroupCfg]`
    - `actions: dict[str, ActionTermCfg]`
    - `rewards: dict[str, RewardTermCfg]`
    - `terminations: dict[str, TerminationTermCfg]`
    - `commands: dict[str, CommandTermCfg]`
    - `curriculum: dict[str, CurriculumTermCfg]`
    - `metrics: dict[str, MetricsTermCfg]`
    - `events: dict[str, EventTermCfg]` (defaults to `reset_scene_to_default`).
- **Responsibility**: a single structured config object that mirrors IsaacLab’s manager‑based env configs, but implemented as a Python dataclass instead of IsaacLab’s `@configclass` hierarchy. It is the authoritative source of task configuration that managers and the env read from.

**ManagerBasedRlEnv**

- **Construction**:
  - Stores `cfg`.
  - Seeds RNG if `cfg.seed` is specified via `self.seed()`.
  - Builds the scene: `self.scene = Scene(self.cfg.scene, device=device)`.
  - Builds the simulation:
    - `self.sim = Simulation(num_envs=self.scene.num_envs, cfg=self.cfg.sim, model=self.scene.compile(), device=device)`
  - Wires scene to simulation:
    - `self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)`
  - If `scene.sensor_context` exists, attaches it to `sim` via `self.sim.set_sensor_context(self.scene.sensor_context)`.
  - Allocates episode counters and sets up an optional offscreen renderer if `render_mode == "rgb_array"`.
  - Calls `load_managers()` followed by `setup_manager_visualizers()`.

- **Manager loading (`load_managers`)**:
  - Creates an **EventManager**, prints info.
  - Calls `self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)` so the simulation model supports domain randomization required by events.
  - Constructs **CommandManager** (or `NullCommandManager` when no commands).
  - Constructs **ActionManager** and **ObservationManager** from the respective cfg dicts.
  - Constructs **TerminationManager**, **RewardManager** (with `scale_by_dt`), **CurriculumManager** (or `NullCurriculumManager`), and **MetricsManager** (or `NullMetricsManager`).
  - Configures Gym‑style observation and action spaces with `_configure_gym_env_spaces()`.
  - If there is a "startup" event mode, calls `self.event_manager.apply(mode="startup")`.

- **Reset logic (`reset`)**:
  - Arguments: `seed`, `env_ids`, `options` (matching gymnasium style).
  - If `env_ids` is None, resets all envs (range over `num_envs`).
  - Re‑seeds if a new `seed` is provided.
  - Calls a private `_reset_idx(env_ids)` (not shown in the snippet but implied), which:
    - Resets scene entities and internal state buffers.
    - Calls each manager’s `reset` to clear per‑episode state and produce logging extras.
  - Writes data from `scene` into `sim`: `self.scene.write_data_to_sim()`.
  - Calls `self.sim.forward()` to recompute derived MuJoCo quantities.
  - Calls `self.sim.sense()` to run sensor pipelines.
  - Computes initial observations with `self.observation_manager.compute(update_history=True)`.
  - Returns `(obs_buf, extras)` in a `dict`‑of‑tensors format (`types.VecEnvObs`).

- **Step logic (`step`)**:
  - See section **3. Step pipeline** below for a full call graph.

- **Dependencies**:
  - On engine: `Scene`, `SceneCfg`, `Simulation`, `SimulationCfg`, and indirectly `mujoco` / `mujoco_warp` via `sim`.
  - On managers: action, observation, reward, termination, command, curriculum, metrics, event.
  - On tasks: task‑specific MDP functions (e.g., reward and observation terms) referenced inside manager term configs.

**Observation**: `ManagerBasedRlEnv` is the mjlab counterpart of IsaacLab’s `ManagerBasedRLEnv`, explicitly vectorized over `cfg.scene.num_envs` and fully dependent on MuJoCo via the `Simulation` wrapper.

---

### 2.2 Manager base classes

- **File**: `mjlab/managers/manager_base.py`
- **Classes**:
  - `ManagerTermBaseCfg`
  - `ManagerTermBase`
  - `ManagerBase`

**ManagerTermBaseCfg**

- A dataclass that matches IsaacLab’s `ManagerTermBaseCfg` semantics:
  - Field `func: Any`: either a function or a class used to compute a term.
  - Field `params: dict[str, Any]`: keyword arguments passed when calling `func`.
- There is explicit support for **function‑based** and **class‑based** term implementations:
  - Functions are called directly as `func(env, **params)` for each step.
  - Classes are instantiated once at startup as `func(cfg=term_cfg, env=env)` and then called like a callable object.
  - Class terms may implement `reset(env_ids)` to maintain state across episodes.

**ManagerTermBase**

- Holds a reference to the environment (`ManagerBasedRlEnv`).
- Provides utility properties:
  - `num_envs`, `device`, and `name`.
- Defines `reset(env_ids)` and `__call__` as the expected interface for term objects.
- This is the base class for term types that need direct access to `env` and possibly engine data.

**ManagerBase**

- Abstract base class for all managers.
- Holds a reference to the environment and calls `_prepare_terms()` in its constructor.
- Requires subclasses to expose `active_terms` and implement `_prepare_terms`.
- Provides:
  - `reset(env_ids)` returning a dict of extras (default: empty).
  - `get_active_iterable_terms(env_idx)` for introspection / logging.
  - `_resolve_common_term_cfg(term_name, term_cfg: ManagerTermBaseCfg)` that:
    - Walks through `term_cfg.params` and resolves any `SceneEntityCfg` objects against `env.scene`.
    - If `term_cfg.func` is a class, instantiates it with `(cfg=term_cfg, env=self._env)` in place.

**Observation**: this is a faithful re‑implementation of IsaacLab’s manager/term base infrastructure, with explicit references to `SceneEntityCfg`, `ManagerBasedRlEnv`, and MuJoCo scene entities.

---

### 2.3 Scene loader and simulation wrapper

#### Scene

- **File**: `mjlab/scene/scene.py`
- **Classes**:
  - `SceneCfg` (dataclass)
  - `Scene`

**SceneCfg**

- Fields:
  - `num_envs: int`
  - `env_spacing: float`
  - `terrain: TerrainEntityCfg | None`
  - `entities: dict[str, EntityCfg]`
  - `sensors: tuple[SensorCfg, ...]`
  - `extent: float | None`
  - `spec_fn: Callable[[mujoco.MjSpec], None] | None`
- This config describes the overall scene layout and what entities / sensors to include.

**Scene**

- During construction:
  - Loads a base MuJoCo `MjSpec` from a template XML file: `_SCENE_XML`.
  - Optionally overrides `spec.stat.extent`.
  - Calls `_add_terrain()`, `_add_entities()`, `_add_sensors()`, and optional `spec_fn` callback to populate the spec.
  - `_add_entities()`:
    - Builds each `Entity` from its `EntityCfg`.
    - Extracts and merges keyframes to create a combined `init_state` keyframe for the model (aggregating `qpos` and `ctrl` from each entity).
    - Attaches each entity’s spec to the main `MjSpec` under its own frame prefix.
  - `_add_terrain()`:
    - Constructs a `TerrainEntity` and attaches it to the spec, with replicated terrain across `num_envs` and `env_spacing`.
  - `_add_sensors()`:
    - Builds each configured sensor, edits the spec to include its MuJoCo sensor definition, and registers built‑in sensors that exist in the compiled `MjModel`.

- After compilation:
  - `compile()` returns `mujoco.MjModel`.
  - `initialize(mj_model, model, data)`:
    - Creates per‑env origins buffer.
    - Calls `ent.initialize(mj_model, model, data, device)` for all entities.
    - Calls `sensor.initialize(...)` for sensors.
    - Creates a shared `SensorContext` when any sensors require camera or raycast support; this context sets up `mujoco_warp.SensorContext` for batched rendering/raycasting.
  - `reset(env_ids)` and `update(dt)` forward to entities and sensors.
  - `write_data_to_sim()` calls `ent.write_data_to_sim()` for all entities so that MuJoCo / mujoco_warp data structures match the high‑level entity state.

**Observation**: `Scene` is a MuJoCo‑specific scene builder with no engine abstraction; it directly manipulates `mujoco.MjSpec` and is tightly coupled to MuJoCo’s concept of specs, keyframes, and sensors.

#### Simulation

- **File**: `mjlab/sim/sim.py`
- **Class**:
  - `Simulation`

Only the relevant portion is needed for the mjlab‑Genesis boundary:

- `Simulation` wraps:
  - `mujoco.MjModel` and `mujoco.MjData` for CPU representations.
  - `mjwarp.Model` and `mjwarp.Data` for GPU‑accelerated stepping.
  - Warp device / CUDA graph management, NaN guard, and sensor context.
- Key methods:
  - `expand_model_fields(fields: set[str])`:
    - Uses `mjwarp.expand_model_fields` to add batched fields to the GPU model to support domain randomization and per‑env parameter variation.
  - `recompute_constants(level: RecomputeLevel)`:
    - Uses `mjwarp.<level>` to recompute MuJoCo constants after randomization (e.g., body masses).
  - `forward()`:
    - Calls `mjwarp.forward(self.wp_model, self.wp_data)` (with optional CUDA graph capture).
  - `step()`:
    - Inside a `wp.ScopedDevice` context and `nan_guard.watch(self.data)`:
      - Calls `mjwarp.step(self.wp_model, self.wp_data)`, which internally runs MuJoCo’s `mj_step` over all vectorized environments.
  - `reset(env_ids)`:
    - Builds a boolean reset mask and calls `mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)`.
  - `sense()`:
    - Uses `mjwarp.refit_bvh`, `mjwarp.render`, and custom raycast kernels via the `SensorContext` to generate camera images and raycasts in a single CUDA graph.

**Observation**: `Simulation` is the central adapter between mjlab and `mujoco_warp`. All physics stepping and sensing happens here, and the rest of mjlab interacts with physics through this wrapper.

---

### 2.4 Task definitions and MDP components

- **Files (examples)**:
  - `mjlab/tasks/velocity/mdp/observations.py`
  - `mjlab/tasks/velocity/mdp/rewards.py`
  - `mjlab/tasks/velocity/mdp/terminations.py`
  - Similar MDP modules exist under `tasks/tracking/mdp` and `tasks/manipulation/mdp`.

These modules define **pure functions** (and occasionally classes) that implement:

- Observation terms: mapping `(env)` → tensors, typically using entity and sensor state.
- Reward terms: mapping `(env)` → reward tensors.
- Termination conditions: mapping `(env)` → boolean masks.
- Curriculums and commands: mapping `(env)` and `dt` → updated targets or difficulty schedules.

They are referenced from the manager term configs (`ObservationGroupCfg`, `RewardTermCfg`, `TerminationTermCfg`, etc.) and do not hold their own persistent engine state; they rely on the environment and managers to access `scene`, `sim`, and cached buffers.

---

## 3. Step pipeline (`env.step(action)`) call graph

The main step implementation is in `ManagerBasedRlEnv.step`:

1. **Entry point**:
   - Signature: `def step(self, action: torch.Tensor) -> types.VecEnvStepReturn`.
   - Assumes `action` shape `(num_envs, total_action_dim)`.

2. **Action processing**:
   - `self.action_manager.process_action(action.to(self.device))`
     - Stores action into manager’s `_action`, `_prev_action`, `_prev_prev_action`.
     - Splits along the action dimension and calls each `ActionTerm.process_actions(slice)`.
     - Each `ActionTerm` typically prepares control targets for a scene entity (e.g., joint position or torque targets) but does not yet write to MuJoCo data.

3. **Physics stepping loop**:
   - For `self.cfg.decimation` substeps:
     - `self._sim_step_counter += 1`
     - `self.action_manager.apply_action()`
       - Each action term writes its current targets to its controlled entity.
     - `self.scene.write_data_to_sim()`
       - Entities propagate their internal state to `mjwarp.Data` via the `Simulation` bridge.
     - `self.sim.step()`
       - Calls `mjwarp.step(self.wp_model, self.wp_data)` inside a CUDA graph, which in turn performs MuJoCo’s `mj_step` for all environments.
     - `self.scene.update(dt=self.physics_dt)`
       - Entities and sensors update any internal buffers that depend on time or on recently applied actions.

   **Where physics stepping happens**:
   - In `Simulation.step()` calling `mjwarp.step`, which is the MuJoCo‑Warp wrapper for MuJoCo’s integration and constraint solving. This is the only place where `mjwarp.step` is invoked, and all environments are stepped in a batched fashion on the GPU.

4. **Episode bookkeeping**:
   - `self.episode_length_buf += 1`
   - `self.common_step_counter += 1`

5. **Termination and reward computation**:
   - `self.reset_buf = self.termination_manager.compute()`
     - For each termination term:
       - Calls `term_cfg.func(self._env, **term_cfg.params)` to get boolean tensors.
       - Accumulates into `_truncated_buf` or `_terminated_buf`.
     - Returns `dones = truncated | terminated`.
   - `self.reset_terminated = self.termination_manager.terminated`
   - `self.reset_time_outs = self.termination_manager.time_outs`
   - `self.reward_buf = self.reward_manager.compute(dt=self.step_dt)`
     - For each reward term:
       - Calls `term_cfg.func(self._env, **term_cfg.params)`.
       - Multiplies by `term_cfg.weight` and optionally by `dt`.
       - NaN/Inf sanitized, accumulated into `_reward_buf` and `_episode_sums`.
   - `self.metrics_manager.compute()`:
     - Computes per‑step metrics from configured functions.

6. **Resetting terminated / timed‑out environments**:
   - `reset_env_ids = self.reset_buf.nonzero().squeeze(-1)`
   - If any envs need reset:
     - Calls `self._reset_idx(reset_env_ids)`:
       - Resets entities, sensors, and manager state for those envs (mirroring `reset`).
     - Writes state back to MuJoCo via `self.scene.write_data_to_sim()`.

7. **Forward and commands**:
   - `self.sim.forward()`
     - Calls `mjwarp.forward(self.wp_model, self.wp_data)` to recompute derived quantities (`xpos`, `xquat`, etc.) after state updates from resets and physics.
   - `self.command_manager.compute(dt=self.step_dt)`
     - Updates command signals (e.g., target velocities) possibly used by observation and reward terms.
   - If `"interval"` is in `event_manager.available_modes`:
     - `self.event_manager.apply(mode="interval", dt=self.step_dt)`
       - Applies domain randomization and event logic at fixed intervals, potentially expanding model fields and recomputing constants.

8. **Sensing and observations**:
   - `self.sim.sense()`
     - Uses `SensorContext` and `mujoco_warp` operations (`refit_bvh`, `render`, raycast kernels) to update GPU sensor data.
   - `self.obs_buf = self.observation_manager.compute(update_history=True)`
     - For each observation group:
       - Calls term functions or instances to compute raw features from `env` state.
       - Applies noise, history buffers, delays, concatenation, and NaN/Inf handling.

9. **Return**:
   - Returns a Gym‑like tuple:
     - `(obs_buf, reward_buf, reset_terminated, reset_time_outs, extras)`.
     - `reset_terminated` and `reset_time_outs` correspond to Gymnasium’s terminated / truncated flags.

**State buffer handling**:

- Physics state lives in `Simulation`’s MuJoCo / mujoco_warp data structures, with per‑env vectors.
- The environment maintains:
  - `episode_length_buf`, `common_step_counter`, `reset_buf`, `reset_terminated`, `reset_time_outs`.
  - `obs_buf`, `reward_buf`.
- Managers maintain their own buffers:
  - Reward: `_reward_buf`, `_step_reward`, `_episode_sums`.
  - Termination: `_truncated_buf`, `_terminated_buf`, per‑term `_term_dones`.
  - Observation: per‑group `_group_obs_dim`, caches (`_obs_buffer`), term history and delay buffers.
  - Action: `_action`, `_prev_action`, `_prev_prev_action`.
  - Command / curriculum / metrics: per‑term internal states depending on implementations.

**Whether mjlab wraps mujoco‑warp or calls it directly**:

- All *core* physics stepping and sensing is done through the `Simulation` wrapper. Managers and tasks never call `mujoco_warp` directly; they obtain state through `env.scene`, `env.sim`, and entity accessors.
- Visualization modules (`viewer/*`) and some utilities (`utils/mujoco.py`) do import `mujoco` directly, but not `mujoco_warp`—the latter is contained in `sim` and some scene‑level sensing code.

---

## 4. Manager system analysis

### 4.1 Manager types and construction

The main managers are:

- `ObservationManager` (`mjlab/managers/observation_manager.py`)
- `RewardManager` (`mjlab/managers/reward_manager.py`)
- `ActionManager` (`mjlab/managers/action_manager.py`)
- `TerminationManager` (`mjlab/managers/termination_manager.py`)
- `CommandManager`, `EventManager`, `CurriculumManager`, `MetricsManager` (not fully detailed here but structurally similar).

All managers:

- Inherit from `ManagerBase`.
- Accept a `cfg` dictionary mapping term names to term configs, plus a reference to the `ManagerBasedRlEnv`.
- In their `_prepare_terms()` implementation, they:
  - Iterate over `cfg` items.
  - Skip `None` configs (disabled terms).
  - Call `_resolve_common_term_cfg(term_name, term_cfg)`:
    - Resolves `SceneEntityCfg` references to concrete scene entities.
    - Instantiates class‑based term functions when `func` is a class.
  - Register term names and term configs, and allocate buffers as necessary.

**ObservationManager**

- Config type: `ObservationGroupCfg` (per‑group configuration) containing `ObservationTermCfg` entries (not shown here but present in the file).
- Organizes terms in *groups* (e.g., "actor", "critic"):
  - Each group can:
    - Concatenate terms into a single tensor or keep them separate.
    - Maintain history and delay for each term through internal buffers.
    - Apply noise models to terms.
    - Use NaN/Inf policies per group (`nan_policy`, `nan_check_per_term`).
- Maintains:
  - `_group_obs_term_names`, `_group_obs_term_dim`, `_group_obs_term_cfgs`.
  - `_group_obs_concatenate`, `_group_obs_concatenate_dim`, `_group_obs_dim`.
  - Delay and history buffers per term.
- `compute(update_history: bool)`:
  - Returns a cached `_obs_buffer` when possible to avoid double‑pushing data to delay/history buffers.
  - Otherwise iterates over groups and computes each group’s observations, stores them in `_obs_buffer`, and returns.
- `reset(env_ids)`:
  - Clears the observation cache and resets:
    - Class‑based term instances (`term_cfg.func.reset(env_ids)`).
    - Delay and history buffers for terms that have them.
  - Returns an empty extras dict (no scalar episode stats here).
- **Ownership of state**:
  - Owns all observation buffers, noise models, and history/delay buffers, but does not own the underlying physics state.
  - Reads state from `env` and `env.scene` / `env.sim` via term functions and entity APIs.

**RewardManager**

- Config type: `RewardTermCfg(ManagerTermBaseCfg)` with additional `weight: float`.
- In `_prepare_terms()`:
  - Resolves the term config via `_resolve_common_term_cfg`.
  - Gathers term names and configs, and class‑based term configs (for later `reset` calls).
- `compute(dt: float)`:
  - Iterates through active terms:
    - Calls `term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale`, where `scale` is `dt` or `1.0` depending on `scale_by_dt`.
    - Sanitizes NaN/Inf to 0.0.
    - Accumulates into `_reward_buf` and `_episode_sums[name]`.
    - Records per‑term step reward (unscaled by `dt`) into `_step_reward`.
- `reset(env_ids)`:
  - For each term’s accumulated episode sum:
    - Computes episodic average reward for the selected envs and logs as `Episode_Reward/<term>`.
    - Zeros the episode sums for those envs.
  - Calls `reset(env_ids)` on any class‑based term instances.
- **Ownership of state**:
  - Owns reward buffers and per‑term episode accumulators.
  - Does not own physics state; all physics‑dependent values are computed via term functions on `env`.

**ActionManager**

- Config type: `ActionTermCfg` (abstract base) with:
  - `entity_name: str`
  - `clip: dict[str, tuple] | None`
  - Abstract `build(self, env) -> ActionTerm`.
- `ActionTerm`:
  - Inherits from `ManagerTermBase`.
  - Constructor:
    - Stores config and a reference to the controlled entity via `self._env.scene[self.cfg.entity_name]`.
  - Requires:
    - `action_dim` (number of action dimensions).
    - `process_actions(actions: torch.Tensor)` to handle the per‑term slice of the action tensor.
    - `apply_actions()` to write targets into the entity / actuators.
    - `raw_action` property to expose unprocessed actions if needed.
- `ActionManager`:
  - `_prepare_terms()`:
    - Builds each `ActionTerm` via `cfg.build(self._env)`.
    - Registers term names and instances.
  - Maintains:
    - `_action`, `_prev_action`, `_prev_prev_action` tensors over `(num_envs, total_action_dim)`.
  - `process_action(action)`:
    - Validates shape.
    - Shifts history and copies action into `_action`.
    - Splits the flat action into slices per term and calls `term.process_actions(slice)`.
  - `apply_action()`:
    - Iterates over terms and calls `term.apply_actions()` to actually write control targets to the simulation each physics substep.
- **Ownership of state**:
  - Owns action history buffers.
  - Term instances own the mapping from actions to entity targets (e.g., PD targets or torque commands).
  - No direct MuJoCo calls; entities and the simulation handle engine‑side details.

**TerminationManager**

- Config type: `TerminationTermCfg(ManagerTermBaseCfg)` with `time_out: bool`.
- Maintains:
  - `_term_names`, `_term_cfgs`, `_class_term_cfgs`.
  - Per‑term done buffers: `_term_dones[name]`.
  - Overall `_truncated_buf` and `_terminated_buf`.
- `compute()`:
  - Zeros `_truncated_buf` and `_terminated_buf`.
  - For each term:
    - Calls `term_cfg.func(self._env, **term_cfg.params)` to get a boolean mask.
    - ORs it into either `_truncated_buf` or `_terminated_buf` depending on `time_out`.
    - Stores the per‑term mask into `_term_dones[name]`.
  - Returns `dones = truncated | terminated`.
- `reset(env_ids)`:
  - Logs per‑term termination counts as `Episode_Termination/<term>`.
  - Calls `reset(env_ids)` on class‑based term instances.

**Command / Event / Curriculum / Metrics Managers (high‑level description)**

- Follow the same pattern:
  - Term configs extend `ManagerTermBaseCfg`.
  - Managers hold dictionaries of term configs and resolve `SceneEntityCfg` references.
  - `compute()` methods compute commands, schedule difficulty, or aggregate metrics.
  - `reset()` methods log episode‑level statistics and clear term‑internal state.

### 4.2 Engine coupling and abstraction

- Managers are **not engine‑agnostic** in practice:
  - Term configs frequently include `SceneEntityCfg` objects that refer to specific MuJoCo entities (bodies, joints, sensors) by name, and they are resolved through `scene`.
  - Term functions (in `tasks/*/mdp`) read MuJoCo‑specific quantities (joint positions/velocities, contacts, etc.) exposed by entity and sensor APIs that are thin MuJoCo wrappers.
  - `ManagerBasedRlEnv` also exposes MuJoCo‑specific metadata (`metadata["mujoco_version"]`, `metadata["warp_version"]`).
- However, the **manager *pattern*** is conceptually engine‑agnostic:
  - All managers operate in terms of an abstract `env` interface, plus term configs and optional `SceneEntityCfg` references.
  - The strict dependence on MuJoCo arises through `Scene`/`Entity`/`Simulation` and task‑specific MDP functions, not through the manager base classes themselves.

---

## 5. Config usage pattern and relation to IsaacLab’s `configclass`

Within `mjlab/src/mjlab`, there is **no direct use** of an `@configclass` decorator:

- Managers and `ManagerBasedRlEnv` use standard Python dataclasses or plain dicts for configuration.
- The higher‑level configclass system appears in **RenforceRL** and IsaacLab integration code (outside `mjlab/src/mjlab`), which maps IsaacLab‑style `@configclass` trees into mjlab’s dataclass and dict configurations.

In effect:

- mjlab defines **engine‑ and task‑level configs** as:
  - Dataclasses: `ManagerBasedRlEnvCfg`, `SceneCfg`, `SimulationCfg`, term cfg dataclasses (`RewardTermCfg`, `TerminationTermCfg`, etc.).
  - Typed dicts mapping string names to term config instances.
- Task modules in `mjlab/tasks/*/config/*` (e.g., `env_cfgs.py`, `rl_cfg.py`) then construct these configs for particular robots/tasks (e.g., `go1`, `g1`), typically in a style inspired by IsaacLab but implemented without the decorator.
- IsaacLab’s `@configclass` system serves as an **outer configuration shell** in this repository (via RenforceRL and `demo_tasks`), feeding mjlab’s dataclasses. That integration is outside the core `mjlab` package we analyzed here, but it is crucial for understanding how GenesisLab should be wired to an existing configclass system.

In summary:

- **Inside mjlab**: config is dataclass‑based and manager‑local (no global configclass).
- **Outside mjlab**: `@configclass` is used to define user‑facing tasks and agent configs, which are then converted into mjlab’s internal dataclass configs.

---

## 6. Mujoco‑warp boundary and MuJoCo integration

### 6.1 Where MuJoCo and mujoco_warp are imported

- Core simulation and scene:
  - `mjlab/scene/scene.py`:
    - `import mujoco`
    - `import mujoco_warp as mjwarp` (for types and sensor context).
  - `mjlab/sim/sim.py`:
    - `import mujoco`
    - `import mujoco_warp as mjwarp`
- Visualization and utilities:
  - `mjlab/viewer/viser/scene.py`, `offscreen_renderer.py`, `native/visualizer.py`, etc. import `mujoco` for rendering and debugging but do not control physics stepping.
  - `mjlab/utils/spec.py` imports `mujoco` to inspect and manipulate `MjModel` fields.

### 6.2 Adapter layer vs direct calls

- `Simulation` is the explicit **adapter layer** between mjlab and `mujoco_warp`:
  - All calls to `mjwarp.forward`, `mjwarp.step`, `mjwarp.reset_data`, `mjwarp.expand_model_fields`, `mjwarp.refit_bvh`, `mjwarp.render` are encapsulated inside this class.
  - `Scene` constructs `mujoco.MjSpec` / `MjModel` and uses `mjwarp.Model`, `mjwarp.Data` only indirectly through `Simulation` and `SensorContext`.
- Managers and tasks **never call `mujoco_warp` directly**. They rely entirely on:
  - `env.sim` methods (`forward`, `step`, `reset`, `sense`, etc.).
  - `env.scene` and entity APIs to access state and write control targets.

### 6.3 State access and DOF indexing

- State access patterns:
  - Entities expose high‑level getters like `get_dofs_position`, `get_dofs_velocity`, `get_pos`, `get_quat`, etc., which read from the `mujoco`/`mjwarp` data structures inside `Simulation`.
  - Tasks use entity and sensor APIs instead of directly indexing into `MjData` arrays, encapsulating indexing logic inside the entity layer.
- DOF indexing:
  - Centralized in entity implementations and utility functions (`utils/mujoco.py`, `asset_zoo/*` constants).
  - DOF indices, joint name mappings, and contact site indices are typically precomputed once at initialization in class‑based term functions or entities.

### 6.4 Contact handling and reset

- Contacts:
  - Observations and rewards that depend on contact data read it through entity and sensor interfaces that wrap MuJoCo’s `cfrc_ext`, `contact` structures, etc.
  - `SensorContext` and raycast sensors use `mujoco_warp` raycasting and BVH refitting to obtain geometry‑based signals.
- Reset:
  - `Simulation.reset(env_ids)` uses `mjwarp.reset_data` to restore MuJoCo data to a default state for selected envs.
  - `Scene.reset(env_ids)` resets entities and sensors to their initial states (often using MuJoCo keyframes and cached defaults).
  - `_reset_idx(env_ids)` in `ManagerBasedRlEnv` coordinates:
    - `scene.reset`.
    - `sim.reset`.
    - Manager `reset` calls and logging of episodic statistics.

### 6.5 Engine abstraction cleanliness

- There is **no generic engine abstraction** in mjlab:
  - The entire stack is explicitly MuJoCo‑centric (`mujoco.MjSpec`, `MjModel`, `MjData`, `mujoco_warp.Model/Data`).
  - Scene and simulation types are not parameterized over an engine interface.
  - Manager configs and term functions depend on `SceneEntityCfg` and entity classes whose semantics are tied to MuJoCo.
- Nonetheless, mjlab demonstrates that:
  - IsaacLab’s manager‑based logic (observation, reward, action, termination, events, curriculum, metrics) can be reimplemented cleanly on top of another engine, given an appropriate scene/sim wrapper.
  - The main work is encapsulating engine specifics into a small number of layers: `Scene`, `Entity`, `Sensor`, and `Simulation`.

---

## 7. Summary of mjlab’s emulation of IsaacLab

- **Environment abstraction**:
  - `ManagerBasedRlEnv` mirrors IsaacLab’s `ManagerBasedEnv` / `ManagerBasedRLEnv`, including decimation, seed handling, Gym‑like API, and manager orchestration.
- **Manager system**:
  - `ManagerBase`, `ManagerTermBaseCfg`, and manager subclasses form a nearly one‑to‑one conceptual mapping to IsaacLab’s manager framework, with similar responsibilities and reset/compute patterns.
- **Engine boundary**:
  - MuJoCo and mujoco_warp are confined to `Scene`, `Simulation`, and entity/sensor types, which together act as the “engine backend” for the manager‑based environment.
- **Vectorization**:
  - `SceneCfg.num_envs` and `Simulation`’s batched `mujoco_warp` integration provide native vectorized stepping over many environments in a single MuJoCo model, analogous to IsaacLab’s vectorized envs.
- **Config handling**:
  - Internally, mjlab uses Python dataclasses and dicts rather than IsaacLab’s `@configclass`; the higher‑level `configclass` usage lives in RenforceRL and external task configs which feed into these dataclasses.

This concludes the reverse‑engineered view of mjlab’s architecture, focusing on environment, manager, simulation, and MuJoCo‑Warp integration. Further documents will build on this to compare with Genesis and to plan the GenesisLab design.

