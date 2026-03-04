## Comparative analysis: IsaacLab, mjlab, and Genesis

This document compares key architectural components across IsaacLab, mjlab, and Genesis, based on code inspection of mjlab and Genesis in this repository and on IsaacLab’s published design. The goal is to identify which IsaacLab concepts are portable to Genesis, which break, and what must be redesigned for GenesisLab.

---

## 1. High‑level comparison table

| Component          | IsaacLab                                                | mjlab                                                            | Genesis                                                          |
| ------------------ | ------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| Engine Abstraction | Managers and tasks sit on top of Isaac Sim engine; engine details mostly wrapped in `SceneEntityCfg` and internal scene/sim wrappers | Manager‑based envs atop MuJoCo + `mujoco_warp`; engine details wrapped by `Scene`, `Simulation`, `Entity`, but MuJoCo concepts leak into managers and tasks | Native physics engine with `Scene`/`Simulator`/`Entity` hierarchy; no RL layer or manager system, but clear step, state, and asset APIs |
| Manager System     | Full manager stack: Observation, Reward, Action, Termination, Command, Curriculum, Randomization, Metrics, Events | Re‑implements IsaacLab managers (`ManagerBase`, `ManagerTermBaseCfg`, specific managers) almost one‑to‑one, with term configs and `SceneEntityCfg` resolution | No manager system; solvers and sensors only. Any manager‑like abstraction must be introduced externally (e.g., GenesisLab) |
| Vectorization      | Vectorized environments via Isaac Sim’s batched scenes and physics; managers operate over batched tensors | Native vectorization via `mujoco_warp` batched models and `num_envs` in `SceneCfg` / `Simulation`; managers assume vectorized tensors | Vectorized environments via `Scene.build(n_envs)` and batched solvers; all state and control APIs expose batched tensors |
| Asset Loading      | Isaac Sim asset pipeline (USD, URDF, etc.), wrapped by IsaacLab scene config and `SceneEntityCfg` | MuJoCo scene built from `MjSpec`, XML assets; robots from MJCF/XML; DOF indexing and naming handled via entity layer and constants | Asset loading via `gs.morphs` (URDF, MJCF, USD, primitives, meshes) and `Scene.add_entity`; DOF indexing and naming exposed via entity/joint APIs |
| Reset Handling     | Environment reset orchestrated by env + managers; randomization via Event/Randomization managers; engine state reset via scene interface | `ManagerBasedRlEnv._reset_idx` coordinates `Scene.reset`, `Simulation.reset`, and manager resets; events and curriculum may randomize `Simulation` model fields via `expand_model_fields` and `recompute_constants` | `Scene._reset` and `Simulator.reset` reset solver states and sensors given a `SimState`; no RL‑specific reset policy or episodic semantics in engine |

---

## 2. Engine abstraction

### 2.1 IsaacLab

- Encapsulates engine details (Isaac Sim) behind:
  - Scene and robot configs.
  - `SceneEntityCfg` which resolves to engine‑specific objects (articulations, bodies, sensors).
  - Manager term configs that reference scene entities and objects indirectly via config.
- Engine details still leak through:
  - Field names and semantics in `SceneEntityCfg` (e.g., link frames, joint dof types).
  - Specific observation and reward terms that rely on Isaac Sim’s state layout and APIs.

### 2.2 mjlab

- Uses MuJoCo as the underlying engine, with GPU acceleration via `mujoco_warp`.
- Engine abstraction is **concentrated** in:
  - `Scene`: builds `mujoco.MjSpec` from entities and terrain; attaches sensors.
  - `Simulation`: wraps `mujoco` and `mujoco_warp` for stepping, forward kinematics, resetting, sensing, and domain randomization.
  - `Entity` classes: wrap MuJoCo joint, body, and sensor accessors (positions, velocities, contacts).
- At the manager layer:
  - `ManagerBase` and `ManagerTermBaseCfg` themselves are engine‑agnostic.
  - However, term configs frequently include `SceneEntityCfg`, which is MuJoCo‑specific, and term functions operate on MuJoCo semantics.
  - Metadata like `metadata["mujoco_version"]`, `metadata["warp_version"]` are referenced in the env.

### 2.3 Genesis

- Engine abstraction is defined entirely by Genesis itself:
  - `Scene`: owns `Simulator`, entities, visualization, and recording; exposes `step()` and asset loading.
  - `Simulator`: manages solvers, coupling, sensor manager, gradient bookkeeping.
  - `Entity`: base class for solver‑specific entities; provides high‑level state and control APIs.
- There is **no existing manager‑based RL layer** or `SceneEntityCfg`‑style indirection for RL tasks.
- Instead, user code (e.g., `Go2Env` in `examples/locomotion/go2_env.py`) interacts directly with:
  - `Scene.add_entity`, `Scene.build`, and `Scene.step`.
  - Entity APIs for robot state and control (e.g., `get_pos`, `get_quat`, `get_dofs_*`, `set_dofs_kp`, `control_dofs_position`).

**Portability**:

- The idea of **encapsulating engine details inside a small set of wrappers** (`Scene`, `Simulation`, `Entity`) clearly transfers from mjlab to Genesis:
  - Genesis already has `Scene` and `Simulator` primitives that play similar roles.
  - A `SceneEntityCfg`‑like layer could be added atop Genesis entities without modifying the engine.
- However:
  - IsaacLab/mjlab’s reliance on MuJoCo‑specific fields and semantics (e.g., `qpos`, `qvel`, site positions) does not port 1:1.
  - Genesis supports multiple solvers and materials beyond rigid bodies; some IsaacLab assumptions (single rigid solver, specific contact model) will not hold universally.

---

## 3. Manager system

### 3.1 IsaacLab

- Manager‑based workflow:
  - `ManagerBase`, `ManagerTermBase`, and `ManagerTermBaseCfg` define the pattern for terms and managers.
  - Built‑in managers: Observation, Reward, Action, Termination, Command, Curriculum, Randomization, Metrics.
  - Environments load managers from a configuration (`ManagerBasedRLEnvCfg`) that uses `@configclass` to define nested configs.
  - Reset and step orchestration:
    - Env calls each manager’s `reset` on episode resets, typically aggregating logging metrics.
    - Env’s `step` delegates to the managers in a fixed order (action → simulation → reward/termination → randomization → observation).

### 3.2 mjlab

- Re‑implements IsaacLab’s manager pattern almost directly:
  - `ManagerTermBaseCfg`, `ManagerTermBase`, `ManagerBase` in `mjlab/managers/manager_base.py`.
  - Observation, Reward, Termination, Action, Command, Event, Curriculum, Metrics managers in `mjlab/managers/*`.
  - `ManagerBasedRlEnvCfg` and `ManagerBasedRlEnv` orchestrate manager creation and step/reset calls.
- Differences from IsaacLab:
  - Uses plain dataclasses instead of `@configclass` internally.
  - Environment config and managers are tuned for MuJoCo and MuJoCo‑Warp semantics.
  - Integration with an outer `configclass` system (in RenforceRL) occurs outside `mjlab`.

### 3.3 Genesis

- No manager system in core engine:
  - No `ManagerBase` concept.
  - No per‑component observation/reward/termination decomposition; users typically write monolithic env classes (e.g., `Go2Env`).
- Environment responsibilities currently live in **user code**:
  - `Go2Env` directly:
    - Manages buffers (`obs_buf`, `rew_buf`, `reset_buf`, etc.).
    - Implements step logic: writes control to robot, calls `scene.step()`, reads state, computes rewards, and handles resets.
  - There is no built‑in event, command, curriculum, or metrics abstraction.

**Portability**:

- The **manager pattern** is portable as a conceptual layer above Genesis’s `Scene`:
  - Managers would operate on a Genesis env object that wraps a `Scene` and a set of entities, analogous to `ManagerBasedRlEnv`.
  - Manager term configs could reference Genesis entities through a `SceneEntityCfg`‑like abstraction adapted to Genesis’s entity model.
- However, none of this exists yet:
  - GenesisLab must design its own manager system, reusing IsaacLab’s patterns but constrained by Genesis’s multi‑solver, multi‑material nature.
  - Direct copying from mjlab would reintroduce MuJoCo concepts (e.g., `mujoco` field names) which are not appropriate in Genesis.

---

## 4. Vectorization

### 4.1 IsaacLab

- Relies on Isaac Sim’s batched simulation to support vectorized environments.
- Env classes and managers assume batched tensors and implement `num_envs` semantics.
- Vectorization is transparent to RL algorithms: envs behave like a vecenv with a single batched step call.

### 4.2 mjlab

- Uses `SceneCfg.num_envs` and `Simulation(num_envs=...)` to create batched MuJoCo/MuJoCo‑Warp models.
- `ManagerBasedRlEnv` exposes `num_envs` and stores:
  - `episode_length_buf`, `reset_buf`, `obs_buf`, `reward_buf` as batched tensors.
  - Manager configs assume per‑env vectorization when defining term functions.
- `rl/vecenv_wrapper.py` adapts `ManagerBasedRlEnv` into the vectorized environment API required by RSL‑RL (TensorDict, etc.).

### 4.3 Genesis

- Native vectorization:
  - `Scene.build(n_envs)` sets `scene.n_envs = n_envs` and configures batch size `_B` and parallelization level `_para_level`.
  - Solvers and entities shape their internal data and kernels based on `n_envs`.
- User‑level envs, such as `Go2Env`, manually:
  - Allocate batched buffers (`obs_buf`, `rew_buf`, `reset_buf`, etc.) of shape `(num_envs, ...)`.
  - Call `scene.step()` once per global step; stepping is inherently batched.
  - Use entity APIs that return batched state (e.g., `get_dofs_position` returns `(num_envs, dofs)`).

**Portability**:

- The notion of treating the first tensor dimension as `num_envs` is directly shared between mjlab and Genesis.
- Differences:
  - Genesis’s vectorization is generalized across multiple solvers and materials; not all solvers support batching yet (e.g., SF solver).
  - In GenesisLab, we must:
    - Detect which solvers/entities are used in a task and respect solver‑specific batching constraints.
    - Avoid assuming rigid‑only or MuJoCo‑style semantics where they do not apply.

---

## 5. Asset loading

### 5.1 IsaacLab

- Uses Isaac Sim USD/URDF asset infrastructure with wrappers:
  - Robots and environments are described via config objects that eventually reference engine‑level assets.
  - `SceneEntityCfg` encapsulates link/joint names, DOFs, and sensor mount points.

### 5.2 mjlab

- Uses MuJoCo XML assets to build robots and environments:
  - `Scene` loads a base XML spec and attaches entities derived from `EntityCfg` and terrain configs.
  - DOF indices and joint names are resolved once at initialization (via `utils/mujoco.py` and entity logic) and then used by MDP terms.
  - Sensors are defined via MuJoCo sensor specifications and wrapped in sensor classes.

### 5.3 Genesis

- Uses `gs.morphs` to define shapes and robots:
  - `URDF`, `MJCF`, `USD`, primitive shapes, and meshes.
  - Entities are created via `Scene.add_entity`, which validates morph/material combinations and sets visualization and contact options.
- DOF indexing and joint naming:
  - Handled via entity/joint APIs (`get_joint(name).dof_start`) and solver‑side state layout.
  - Users (or a higher‑level framework) must query these and construct their own indexing schemes for controllers and observation terms.

**Portability**:

- The pattern “load robot from URDF/MJCF → query DOF indices → build observation and reward terms around those indices” is portable from mjlab to Genesis.
- Differences:
  - Genesis supports a broader range of asset types and solvers; asset loading cannot assume rigid‑only or MuJoCo’s joint model.
  - In GenesisLab, asset loading must:
    - Use `Scene.add_entity` and morph options consistently.
    - Enforce or detect rigid‑only assumptions for IsaacLab‑style locomotion/manipulation tasks, while leaving room for soft bodies or fluids in future tasks.

---

## 6. Reset handling

### 6.1 IsaacLab

- Env reset typically:
  - Resets scene and articulation states.
  - Applies domain randomization via randomization or event managers.
  - Clears manager buffers and posts episodic statistics (reward sums, termination counts).
  - Recomputes initial observations and returns them.

### 6.2 mjlab

- `ManagerBasedRlEnv.reset`:
  - Determines `env_ids` to reset, optionally re‑seeds RNG.
  - Calls `_reset_idx(env_ids)` which:
    - Resets entities and sensors via `Scene.reset`.
    - Resets the `Simulation` (`Simulation.reset`) to default state for those envs.
    - Calls managers’ `reset` methods to clear internal buffers and log episode metrics.
  - Writes scene state into the simulation (`scene.write_data_to_sim`).
  - Calls `sim.forward()` and `sim.sense()` to refresh derived state and sensors.
  - Computes initial observations via `observation_manager.compute(update_history=True)`.

### 6.3 Genesis

- Engine‑level reset:
  - `Simulator.reset(state: SimState, envs_idx=None)` sets solver states to a given `SimState` and resets sensors, removing gradient history.
  - `Scene._reset` (internal) resets scene‑level state and step counters.
  - From an RL perspective, there is no built‑in concept of episode boundaries or episodic metrics; those are handled in user envs.
- `Go2Env` exemplifies RL‑level reset:
  - Maintains `reset_buf` and `episode_length_buf`.
  - Uses `robot.set_qpos(...)` and buffer operations to reset robots and episode statistics.
  - Calls its own `_reset_idx(envs_idx)` and then recomputes observations and `extras`.

**Portability**:

- The logic “env reset orchestrates engine reset + manager resets + randomization + initial observations” is portable but must be built from scratch in GenesisLab.
- Genesis does not expose episodic logic; we must design:
  - A reset pipeline at the RL layer that:
    - Uses Genesis entity and scene APIs to reset physical state deterministically.
    - Invokes manager resets and randomization terms.
    - Maintains per‑env buffer state consistent with vectorization.

---

## 7. Gap analysis between mjlab and Genesis

### 7.1 Portable IsaacLab/mjlab concepts

The following concepts from IsaacLab and mjlab appear **directly portable** to Genesis with minimal redesign:

- **Manager pattern**:
  - Decomposing env logic into Observation, Reward, Action, Termination, Command, Curriculum, Event, and Metrics managers, each built from configurable terms.
- **Term configuration**:
  - Using a `ManagerTermBaseCfg`‑style pattern where each term specifies a `func` (function or class) and `params` dictionary.
  - Supporting both function‑style and class‑style terms, with optional `reset(env_ids)` methods for persistent state.
- **Decimation model**:
  - Separating physics substeps from env steps via a `decimation` parameter (as in `ManagerBasedRlEnvCfg`) and looping over `scene.step()` or solver substeps.
- **Vectorized env semantics**:
  - Treating the first tensor dimension as `num_envs` and maintaining per‑env buffers (`obs_buf`, `rew_buf`, `reset_buf`, `episode_length_buf`, etc.).
- **Asset loading pattern**:
  - Loading a robot from URDF/MJCF, querying joint DOFs and names, and basing MDP terms on these indices.
  - Creating observation and reward terms that operate on batched entity state (positions, velocities, contacts).

### 7.2 Assumptions that break or must be re‑examined in Genesis

Some mjlab/IsaacLab assumptions are **MuJoCo‑ or Isaac‑specific** and require rethinking for Genesis:

- **Single rigid solver assumption**:
  - IsaacLab/mjlab primarily assume a rigid‑body engine with optional kinematic elements.
  - Genesis has multiple solvers (rigid, MPM, SPH, FEM, PBD, SF) and hybrid entities; tasks may involve non‑rigid physics.
  - Manager and term design must **not** hard‑code assumptions like "all entities are rigid".
- **MuJoCo state layout**:
  - mjlab heavily relies on MuJoCo’s `qpos`/`qvel`, `xpos`, `xquat`, `cvel`, `sensordata`, etc., and the timing semantics around `mj_step` and `mj_forward`.
  - Genesis has its own state layout and solver‑specific structures; we cannot replicate MuJoCo’s naming or one‑substep lag semantics.
- **mujoco_warp domain randomization hooks**:
  - mjlab uses `Simulation.expand_model_fields` and `recompute_constants` to randomize parameters and recompute derived constants.
  - Genesis has different parameterization and infrastructure for materials and solvers; GenesisLab must design randomization hooks around Genesis options and solvers rather than MuJoCo fields.
- **Viewer and sensor integration**:
  - mjlab’s viewer relies on MuJoCo renderers; Genesis has its own visualization stack and sensor design.
  - Observation and logging managers must integrate with Genesis’s visualization and sensor APIs instead of MuJoCo viewers.

### 7.3 What must be redesigned for GenesisLab

Given the above, the following components require explicit redesign rather than direct copying:

- **Engine boundary layer**:
  - A Genesis‑specific environment core that wraps `Scene`, `Simulator`, and entity handles, analogous to `ManagerBasedRlEnv`, but:
    - Using Genesis’s `Scene.step()` and entity APIs instead of `Simulation.step`/`Scene.write_data_to_sim`.
    - Aligning decimation and `dt` semantics with `SimOptions` (dt, substeps) and RL control frequencies.
- **Scene entity configuration**:
  - A `SceneEntityCfg`‑like layer for Genesis that:
    - References Genesis entities (robots, terrain, sensors) via names and metadata.
    - Specifies which solver/material/morph combinations are allowed for each entity.
    - Abstracts away solver differences enough for manager term functions to be reusable across tasks.
- **Manager system for Genesis**:
  - A fresh manager stack that:
    - Reuses IsaacLab/mjlab patterns (term configs, reset semantics, compute order).
    - Is implemented in a **solver‑agnostic** way but allows solver‑specific term functions.
  - Observation and reward terms must access Genesis entities and sensors, not MuJoCo structures.
- **Configclass integration**:
  - Aligning GenesisLab’s configuration with the existing `configclass` system used in this repo (RenforceRL / IsaacLab integration), so that:
    - User‑facing tasks are defined via `@configclass`.
    - Under the hood, these configclasses are converted into GenesisLab’s runtime configs (env cfg, manager cfgs, scene cfgs, etc.), analogous to how mjlab consumes outer configclasses.
- **Reset and randomization pipeline**:
  - A structured reset pipeline that:
    - Uses Genesis’s `Scene` / entity APIs to reset physical state deterministically.
    - Attaches randomization mechanisms suitable for Genesis’s materials, solvers, and assets.
    - Integrates with command, curriculum, and event managers at the RL layer.
- **Vectorized buffer storage and stepping**:
  - Defining policy‑facing buffers and semantics (`obs_buf`, `rew_buf`, `terminated`, `truncated`, `extras`) in a way that respects Genesis’s `n_envs` and solver support.
  - Connecting this to RL frameworks (e.g., vecenv wrappers) similarly to `mjlab/rl/vecenv_wrapper.py`, but without assuming MuJoCo or RSL‑RL specifics.

---

## 8. Summary

- **Portable concepts**:
  - Manager‑based decomposition of env logic.
  - Term configuration pattern with function/class duality.
  - Vectorized batched environments with decimation between physics and control steps.
  - Asset loading + DOF indexing for robots and tasks.
- **Non‑portable assumptions**:
  - Direct reliance on MuJoCo state layout, `mj_step`/`mj_forward` semantics, and mujoco_warp APIs.
  - Single rigid solver worldview and strict MuJoCo contact model.
- **Redesign targets for GenesisLab**:
  - A Genesis‑centric engine boundary (`Scene`‑based env core).
  - Genesis‑compatible manager system and scene entity configs.
  - Integration with existing `configclass` infrastructure.
  - Reset, randomization, and vectorized buffer semantics aligned with Genesis’s solvers and parallelization.

These findings set the stage for a controlled, layered GenesisLab design that respects Genesis’s capabilities instead of assuming MuJoCo‑like behavior. Concrete architecture proposals will build on this analysis without introducing hard‑coded engine leaks or prematurely fixing interfaces.

