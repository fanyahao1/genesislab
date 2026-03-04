## GenesisLab preliminary layered design (conceptual, no concrete interfaces)

This document outlines a **conceptual** layered design for GenesisLab based on the reverse‑engineered architectures of IsaacLab, mjlab, and Genesis. It deliberately avoids concrete code, fixed class names, or final interfaces; the goal is to define layers, boundaries, and responsibilities only.

---

## 1. Layer decomposition

From bottom to top, the proposed conceptual layers are:

1. **Genesis Engine Layer** (existing):
   - Components: `Scene`, `Simulator`, solvers, entities, sensors, visualization, and options.
   - Responsibility: simulate physics and sensors over `n_envs` vectorized environments, manage assets and solvers.

2. **Genesis Engine Binding Layer** (to be defined for GenesisLab):
   - Conceptual role: a thin binding that exposes a stable, RL‑centric interface over a `Scene` and a subset of entities.
   - Responsibilities:
     - Hold references to:
       - A Genesis `Scene` instance.
       - One or more entities considered “agents” (e.g., robots).
       - Auxiliary entities (terrain, obstacles).
       - Relevant sensors (cameras, raycasts, IMUs).
     - Encapsulate:
       - How `Scene.build(n_envs)` is called (including `n_envs`, spacing, and backend selection).
       - How `Scene.step()` is invoked within a decimation structure.
       - How state tensors (joint states, root poses, velocities, contacts, sensor outputs) are fetched and cached per environment.
     - Provide:
       - Deterministic reset hooks on top of Genesis’s entity APIs.
       - A small set of query methods for managers (e.g., “get base pose”, “get joint states”, “get contact info”), without leaking solver internals.

3. **GenesisLab Manager Layer** (to be built, inspired by IsaacLab/mjlab):
   - Conceptual role: replicate the manager‑based decomposition (observation, reward, action, termination, command, curriculum, events, metrics) on top of the engine binding layer.
   - Responsibilities:
     - Define:
       - Base manager concepts (manager, term, term config), structurally similar to IsaacLab/mjlab but agnostic to specific solvers.
       - A manager‑based environment core that:
         - Orchestrates manager reset and compute calls.
         - Maintains vectorized episode buffers (`obs`, `rew`, `terminated`, `truncated`, `extras`) over `num_envs`.
         - Drives `Scene.step()` via the binding layer with a configurable decimation factor.
     - Provide:
       - Observation managers that produce grouped, possibly concatenated tensors, with noise/history/delay and NaN policies.
       - Reward managers that combine weighted reward terms and track episodic sums.
       - Termination managers that produce `terminated` and `truncated` signals per env.
       - Action managers that map flat policy actions to entity‑level control targets.
       - Command, curriculum, and metrics managers that schedule targets and compute diagnostics.
     - Ensure:
       - Managers rely only on the engine binding layer and a **Genesis‑specific scene entity configuration**, not on MuJoCo or Isaac Sim fields.

4. **GenesisLab Configuration Layer** (to be integrated with `configclass`):
   - Conceptual role: define user‑facing configuration trees using the existing `configclass` system, and map them into runtime configs for the manager layer and engine binding layer.
   - Responsibilities:
     - Provide:
       - Task configs (e.g., locomotion, manipulation) that describe:
         - Environment parameters (episode length, decimation, control frequency).
         - Scene composition (which robots, terrains, and sensors to instantiate).
         - Manager configs (observation terms, reward terms, term weights, termination conditions, commands, curricula, metrics).
       - Agent configs (RL algorithm, network, learning schedules) that sit outside GenesisLab but reference GenesisLab task configs.
     - Implement:
       - A mapping from `@configclass` trees to GenesisLab’s dataclass or config objects for:
         - Engine binding (scene build options, entity selection).
         - Manager system (term configs and groupings).
       - A way to override base configs for specific tasks or robots without touching code, mirroring IsaacLab’s override patterns.

5. **Task Composition and RL Integration Layer**:
   - Conceptual role: provide high‑level task definitions and RL integration helpers built on top of the manager layer.
   - Responsibilities:
     - Define:
       - Task presets for common robots (e.g., legged locomotion, manipulation) that connect:
         - A GenesisLab task config.
         - A Genesis manager‑based environment instance.
       - Optional wrappers that adapt GenesisLab envs to external RL libraries (e.g., vecenv wrappers, TensorDict‑style wrappers), without leaking into core design.
     - Ensure:
       - RL frameworks interact with GenesisLab through a stable, batched API (`reset`, `step`, `num_envs`, `observation_space`, `action_space`, etc.), while engine details remain internal to GenesisLab.

---

## 2. Engine boundary

The **engine boundary** for GenesisLab should be:

- Centered on a **single Genesis `Scene` per env instance**, with:
  - A determinate `SimOptions` (dt, substeps, backend, gradient settings).
  - A fixed set of entities (robots, terrain, objects) and sensors.
  - A fixed `n_envs` chosen at build time.
- Encapsulated by an **engine binding object** that:
  - Owns the `Scene` and defines:
    - How many envs are batched (`n_envs` and layout).
    - Which entities are considered controllable “agents”.
  - Exposes methods to:
    - Reset selected envs:
      - Set entity states (root pose, joint states, velocities) deterministically.
      - Clear internal buffers and command targets.
      - Optionally invoke solver‑specific reset routines.
    - Step the scene:
      - Accept control targets from the action manager.
      - Run `Scene.step()` with a chosen decimation structure:
        - For example, `k` physics substeps per RL step, where `k` is computed from `SimOptions.dt` and the desired control frequency.
      - Optionally inject per‑step randomization or external forces.
    - Query state:
      - Provide stable methods (not direct solver internals) to fetch:
        - Root poses and velocities of agents.
        - Joint states (positions/velocities) for defined sets of DOFs.
        - Contact indicators or contact force aggregates (per link or per env).
        - Sensor outputs (e.g., camera images, ray distances).

At this boundary:

- Managers **must not**:
  - Reach into Genesis solvers directly.
  - Depend on solver‑specific data structures or intermediate buffers.
- Instead, they:
  - Call engine binding methods that are defined explicitly for manager use.
  - Consume batched `torch` tensors that already have the right shapes and semantics.

---

## 3. Manager placement

The manager system sits **entirely above** the engine binding layer:

- It treats the engine binding as an abstract “physics + sensors” service with:
  - Methods to step the world for a given number of substeps.
  - Methods to read batched state.
  - Methods to reset subsets of environments.
- Managers are responsible for:

1. **Action Manager**:
   - Receives flat action tensors from policies.
   - Splits and maps actions onto per‑entity control targets via engine binding methods.
   - Supports:
     - PD targets (positions/velocities).
     - Force/torque commands, if exposed by entities.
   - Handles action history and optional filters (e.g., smoothing, latency).

2. **Observation Manager**:
   - Constructs observation groups from engine binding state queries and possibly manager‑internal buffers.
   - Supports:
     - Concatenated or dict‑style outputs per group (policy vs critic, etc.).
     - Noise, scaling, clipping, history stacks, delay buffers.
   - Knows nothing about solvers; it only sees engine binding APIs and previously cached data.

3. **Reward Manager**:
   - Evaluates reward terms that rely on:
     - Engine binding state queries (poses, velocities, contacts).
     - Commands and curriculum context from other managers.
   - Aggregates scaled terms into a batched reward buffer, with optional dt scaling.
   - Maintains episodic reward sums per term for logging.

4. **Termination Manager**:
   - Evaluates termination and truncation conditions from GenesisLab state and engine binding.
   - Maintains per‑term done flags and overall `terminated`/`truncated` buffers.

5. **Command, Curriculum, Metrics, and Event Managers**:
   - Command manager:
     - Generates target commands (e.g., desired velocities, poses) per env, based on randomization and curriculum.
   - Curriculum manager:
     - Adjusts difficulty or environment parameters over training, possibly through engine binding hooks (e.g., changing terrain, friction).
   - Metrics manager:
     - Aggregates scalar metrics independent of rewards (e.g., tracking errors, contact durations) for logging.
   - Event manager:
     - Applies structured randomization or staged changes in environment parameters, using safe hooks into Genesis options (materials, surfaces, solver parameters).

Managers should:

- Be implemented in a way that:
  - Does not assume a particular Genesis solver (rigid vs MPM vs SPH), unless explicitly specialized in a term.
  - Allows tasks to selectively use subsets of managers (e.g., no curriculum for simple tasks).
  - Can be extended to support multi‑agent setups in the future (multiple controllable entities per env).

---

## 4. Vectorization strategy

GenesisLab’s vectorization strategy should align with Genesis’s native batching model:

- **Single `Scene` with `n_envs`**:
  - GenesisLab should not introduce an independent multiprocessing or process‑based vector env for core functionality.
  - Instead, it should:
    - Choose `n_envs` at scene build time according to user config.
    - Treat the first tensor dimension of all state and control tensors as the batch/environment dimension.

- **Per‑env buffers in GenesisLab**:
  - The manager‑based env core should maintain:
    - `episode_length_buf` (steps per env).
    - `reset_buf` (envs needing reset at the end of step).
    - `terminated` and `truncated` buffers from the termination manager.
    - `reward_buf` and episodic reward sums.
    - Optional cmd/curriculum state per env.
  - All of these buffers should be `torch` tensors on the same device as the engine state.

- **Decimation and step timing**:
  - Genesis’s `SimOptions.dt` and `substeps` define a low‑level time grid.
  - GenesisLab should define a **control period** in terms of `dt` and substeps (e.g., `k` physics steps per RL step), similar to mjlab’s `decimation` parameter:
    - Action manager applies control targets before a block of `k` `Scene.step()` calls.
    - Reward and termination managers evaluate once per RL step.
    - Command and event managers can run at configurable intervals (per step, per `m` steps, etc.).

- **Parallelization level**:
  - Genesis already decides `_para_level` (NONE/PARTIAL/ALL) based on `n_envs` and backend.
  - GenesisLab should **not** override or interfere with this; it simply relies on scene build and `Scene.step()` to handle parallelization details.

---

## 5. Config hierarchy plan

GenesisLab’s configuration hierarchy should:

- Use the existing `configclass` system for user‑facing configs.
- Map to internal, engine‑agnostic configs in a structured way.

Conceptually, the hierarchy might include:

1. **Top‑level task config**:
   - Encodes:
     - Environment type (e.g., “locomotion_velocity”, “manipulation_lift”).
     - Number of environments (`num_envs`) and backend selection hints.
     - Episode horizon and reset policies (infinite vs finite horizon).
     - Decimation / control frequency relative to physics dt.

2. **Scene config**:
   - Describes:
     - Robots (one or more) to instantiate:
       - Path to URDF/MJCF/other assets.
       - Names of DOFs to control.
     - Terrain and obstacles.
     - Sensors to attach (cameras, rays, IMUs).
     - Visualization defaults (viewer options, env spacing).
   - Maps into:
     - Genesis `SimOptions` and solver options.
     - `Scene.add_entity` calls and morph/material/surface options.

3. **Manager configs**:
   - Per‑manager:
     - Observations:
       - Groups (actor, critic, etc.).
       - Terms with:
         - Functions (or classes).
         - Parameters, including references to scene entities and DOFs.
         - Noise, scaling, history/delay options.
     - Rewards:
       - Terms with weights and function references.
     - Terminations:
       - Terms marked as truncation or termination.
     - Commands:
       - Terms generating command signals (e.g., target velocities) and scheduling parameters.
     - Curriculum and metrics:
       - Term definitions and scheduling parameters.
   - These configs are conceptually similar to IsaacLab/mjlab term configs but must be expressed via `configclass` at the user level and then converted into internal dataclasses or objects.

4. **Mapping layer**:
   - A dedicated mapping mechanism should:
     - Take a `configclass` tree as input.
     - Construct:
       - Engine binding configuration (what scene and which entities/sensors to bind).
       - Manager configurations (terms, groups, weights, options).
     - Validate that:
       - Entity names and DOF names exist in the built Genesis `Scene`.
       - Solver constraints (e.g., batching capabilities) are respected.

This mapping layer is essential to keep GenesisLab’s runtime code independent of `configclass` implementation details while still benefiting from config‑driven construction.

---

## 6. Asset loading flow

The asset loading flow in GenesisLab should conceptually be:

1. **Config phase**:
   - User defines:
     - A robot asset (URDF/MJCF path, fixed vs floating base).
     - Terrain and environment assets.
     - Any additional entities (objects, obstacles, tools).
   - This is described using `configclass` objects in the task config.

2. **Engine binding construction**:
   - GenesisLab:
     - Creates a `Scene` with:
       - `SimOptions` set to match desired dt, substeps, and backend.
       - Solver options tuned for task requirements (e.g., rigid‑only vs hybrid).
     - Calls `Scene.add_entity` for:
       - Terrain.
       - Robots.
       - Additional objects.
     - Optionally adds force fields and sensors.

3. **DOF and entity indexing**:
   - After entities are created but before `Scene.build(n_envs)`:
     - GenesisLab queries:
       - Joint objects by name.
       - DOF indices via joint metadata (`dof_start`, DOF counts).
       - Link indices or frames as needed.
     - Constructs and stores:
       - DOF index tensors for each control group (e.g., motors).
       - Mappings from config names to indices used by observation/reward/action terms.

4. **Scene build and batching**:
   - GenesisLab calls `Scene.build(n_envs)` with:
     - `n_envs` from task config.
     - `env_spacing` and `n_envs_per_row` chosen for visualization.
   - Solvers and sensors allocate batched state structures for `n_envs` environments.

5. **Manager initialization**:
   - Once `Scene` is built and DOF indices are known:
     - GenesisLab constructs managers using internal manager configs derived from task config.
     - Term functions/classes can now safely assume that DOF and link indices are known and valid.

This flow cleanly isolates asset import and index resolution from manager‑level logic, making it possible to reuse task MDP definitions across robots and environments with similar structure.

---

## 7. Validation and testing strategy (conceptual)

Before any substantial GenesisLab implementation, a minimal validation plan should ensure:

1. **Engine sanity**:
   - Genesis can:
     - Load at least one robot (e.g., Go2) via URDF.
     - Step stable physics over many timesteps for `n_envs > 1`.
     - Expose joint and base states as batched `torch` tensors.
     - Reset deterministically:
       - Repeated resets from the same initial configuration produce identical states.

2. **Engine binding layer correctness**:
   - A thin, prototype binding (even before managers) can:
     - Construct a `Scene` according to a simple config (no managers).
     - Maintain per‑env buffers for observations and rewards, using only engine APIs.
     - Execute a basic RL loop (or scripted policy) over multiple envs.

3. **Manager layer correctness**:
   - Once preliminary managers exist, they should be:
     - Verified to operate purely on engine binding APIs.
     - Tested for:
       - Correct vectorization (no unintended broadcasting or shape errors).
       - Correct reset semantics (buffers and internal states align with engine resets).
       - No solver‑specific leaks beyond what is allowed by term implementations.

4. **Config mapping correctness**:
   - A small number of `configclass`‑defined tasks should:
     - Instantiate GenesisLab envs successfully.
     - Show that overrides and inheritance work as expected (e.g., varying reward weights, observation groups, or terrain difficulty).

5. **Minimal locomotion task plan**:
   - Using Go2 or a similar quadruped:
     - Define a minimal velocity command task with:
       - One robot, one plane terrain, basic sensors.
       - A small set of observation and reward terms.
       - A simple termination condition (falling, timeouts).
     - Run an RL experiment:
       - Verify training stability and performance metrics.
       - Inspect manager logs (rewards, terminations, commands) to ensure consistency.

These validation steps should guide incremental development of GenesisLab, ensuring that each layer (engine binding, managers, config mapping) is tested and stable before expanding to more complex tasks or multi‑solver scenarios.

---

This preliminary design intentionally stays at the conceptual level. Concrete class names, method signatures, and specific configuration schemas are deferred until after further experimentation and validation on top of Genesis’s actual capabilities and performance characteristics. 

