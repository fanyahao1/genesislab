## Genesis engine architecture summary

This document summarizes the core architecture of the Genesis physics engine as used in this repository, focusing on the pieces that matter for building a manager‑based training framework similar to IsaacLab/mjlab. The emphasis is on simulator and scene structure, state and control APIs, parallelization, and asset import.

---

## 1. Core engine structure

### 1.1 High‑level project layout

From the project’s architecture guide, Genesis is organized as follows:

- `genesis/`:
  - `__init__.py`: entry point, `gs.init()`, global configuration and backend selection.
  - `engine/`: core simulation engine.
    - `scene.py`: `Scene` class – main user‑facing simulation object.
    - `simulator.py`: `Simulator` – manages all physics solvers and sensors.
    - `entities/`: entity types (rigid bodies, MPM solids, SPH fluids, PBD cloth, FEM bodies, tools, drones, etc.).
    - `solvers/`: per‑physics solvers (rigid, MPM, SPH, FEM, PBD, SF).
    - `materials/`: material models associated with solvers.
    - `couplers/`: inter‑solver coupling strategies.
    - `sensors/`: sensor system and sensor manager.
    - `states/`: state representations (`SimState`, caches for queried states).
  - `options/`: configuration classes (Pydantic models) for morphs, solvers, surfaces, sensors, and simulator.
  - `vis/`: visualization system (`Visualizer`, cameras, viewer).
  - `utils/`: geometry, mesh, and other support utilities.

Genesis is thus split into:

- A **scene layer** (`Scene`) that gathers entities, solvers, visualization, and recording.
- A **simulator layer** (`Simulator`) that encapsulates timestep integration and solver coupling.
- **Entity/morph/material/surface layers** that define physical objects and their properties.
- A **sensor subsystem** that attaches sensors to entities and manages batched sensor data.

---

### 1.2 Simulator core class

- **File**: `genesis/engine/simulator.py`
- **Class**: `Simulator`

The `Simulator` is a scene‑level simulation manager. Its responsibilities are:

- Holding references to:
  - The parent `Scene`.
  - Simulator‑level options: `SimOptions`.
  - Solver‑specific options: `ToolOptions`, `RigidOptions`, `MPMOptions`, `SPHOptions`, `FEMOptions`, `SFOptions`, `PBDOptions`.
  - An inter‑solver coupler (e.g., `SAPCoupler`, `IPCCoupler`, `LegacyCoupler`).
  - A `SensorManager` to manage all sensors.
  - A list of all solver instances and of **active** solvers (ones that have entities).
  - A `QueriedStates` cache to track which states have been requested for gradient propagation.
- Managing global time information:
  - `dt` (step size), `substeps`, `substep_dt`, `requires_grad`.
  - A global substep counter `_cur_substep_global`.
  - Gravity vector `gravity`.
- Managing entities:
  - `_entities`: list of all `Entity` instances.
  - `_add_entity(morph, material, surface, ...)`:
    - Dispatches to the appropriate solver (`ToolSolver`, `RigidSolver`, `MPMSolver`, `SPHSolver`, `PBDSolver`, `FEMSolver`) based on material type.
    - Creates a `HybridEntity` for mixed rigid/soft materials.

**Build process**:

- `build()`:
  - Reads `n_envs`, batch size `_B`, and parallelization level from the associated `Scene`.
  - Calls `build()` on each solver; collects active solvers into `_active_solvers`.
  - Constructs the chosen coupler and builds it.
  - Builds hybrid entities if any (`HybridEntity.build()`).
  - Builds the sensor manager (`_sensor_manager.build()`).
  - Enforces certain constraints (e.g., SF solver does not support batching yet).

**Reset**:

- `reset(state: SimState, envs_idx=None)`:
  - For each solver, sets its state using `solver.set_state(0, solver_state, envs_idx)` from the `SimState`.
  - Resets the coupler.
  - Resets gradient state and clears queried state cache.
  - Resets sensor state via `SensorManager.reset(envs_idx)`.

**Step**:

- `step(in_backward: bool = False)`:
  - Optionally checks for rigid solver errors at a fixed rate (`check_errno`).
  - Two main branches:
    - If only the rigid solver is active and no gradients are required:
      - Runs `rigid_solver.substep` in a tight loop for `_substeps` substeps.
    - Otherwise:
      - Calls `process_input(in_backward=False)` on all active solvers to consume high‑level control inputs (target states, forces, etc.).
      - For each substep:
        - Calls `substep(f)` which:
          - Runs coupler pre‑process, `substep_pre_coupling`, coupler coupling, and `substep_post_coupling` across all active solvers.
        - Increments `_cur_substep_global`.
        - Periodically saves checkpoints for gradient computation (`save_ckpt`).
  - Clears rigid solver external forces after the step.
  - Advances sensors via `_sensor_manager.step()`, including rotating buffered sensor data and updating per‑sensor caches.

**Gradient step**:

- `_step_grad()`:
  - Iterates substeps in reverse, loading checkpoints as needed (`load_ckpt`) and running `sub_step_grad` and `process_input_grad` to backpropagate through time.
  - `collect_output_grads()` walks through stored `SimState` objects and solver‑owned queried states, injecting gradient contributions back into solver states.

**State access**:

- `get_state()`:
  - Constructs a `SimState` object with references to the scene, current global step, local substep, and solver states.
  - Registers this state in `_queried_states` for later gradient collection.
  - Returns `SimState`.
- `set_gravity(gravity, envs_idx=None)`:
  - Loops over solvers to set gravity per environment if needed.

**Observation**: `Simulator` is a pure simulation orchestrator. It does not know about RL environments, rewards, or observations; it only manages solvers, their coupling, sensor updates, and gradient bookkeeping.

---

### 1.3 Scene representation

- **File**: `genesis/engine/scene.py`
- **Class**: `Scene`

The `Scene` is the primary user‑facing object representing a full simulation world. It owns:

- A `Simulator` instance.
- Sets of entities (`scene.entities`) built by calls to `add_entity` (and `add_stage` for USD scenes).
- Visualization components (`Visualizer`, viewer, cameras).
- Recorder management (`RecorderManager`) and FPS tracking for profiling.
- Various options objects: `SimOptions`, `CouplerOptions`, solver options, visualization options, viewer options, profiling options, renderer options.

**Construction**:

- The constructor:
  - Accepts a variety of option objects:
    - `sim_options: SimOptions`
    - `coupler_options: BaseCouplerOptions`
    - `tool_options: ToolOptions`
    - `rigid_options: RigidOptions`
    - `mpm_options: MPMOptions`
    - `sph_options: SPHOptions`
    - `fem_options: FEMOptions`
    - `sf_options: SFOptions`
    - `pbd_options: PBDOptions`
    - `vis_options: VisOptions`
    - `viewer_options: ViewerOptions`
    - `profiling_options: ProfilingOptions`
    - `renderer: RendererOptions`
  - Validates all options and fills defaults where necessary.
  - Copies common fields from `sim_options` into solver options (so they share dt, backend, etc.).
  - Constructs a `Simulator`:
    - `self._sim = Simulator(scene=self, options=sim_options, coupler_options=..., tool_options=..., rigid_options=..., ...)`.
  - Creates a `Visualizer` and a `RecorderManager`.
  - Initializes internal flags (`_is_built`, `_forward_ready`, `_backward_ready`) and a unique ID.

**Entity and asset management**:

- `add_entity(morph, material=None, surface=None, visualize_contact=False, vis_mode=None, name=None)`:
  - Ensures default materials and surfaces when not provided.
  - Validates combinations of morph and material (e.g., URDF/MJCF must be used with rigid or hybrid materials).
  - Selects default visualization mode based on material (e.g., `"visual"`, `"collision"`, `"sdf"`, `"particle"`).
  - For file‑based morphs (URDF, MJCF, USD, etc.), may set defaults like convexification.
  - Delegates actual entity creation to the simulator via `self._sim._add_entity(morph, material, surface, visualize_contact, name)`.
  - Returns the created `Entity` (e.g., `RigidEntity`, `MPMEntity`, etc.), which then provides higher‑level APIs like `get_pos`, `get_quat`, `get_dofs_position`, `set_dofs_kp`, `control_dofs_position`, etc., depending on solver type.
- `add_stage(morph: gs.morphs.USD, ...)`:
  - Uses `parse_usd_stage` to turn a USD stage into many morphs.
  - Calls `add_entity` for each one to populate the scene with multiple entities from a stage.

**Build and step**:

- `build(n_envs=0, env_spacing=(0.0, 0.0), n_envs_per_row=None, center_envs_at_origin=True, compile_kernels=None)`:
  - First, calls `_parallelize(...)` (see section 3) to set:
    - `scene.n_envs`
    - True batch size `_B = max(1, n_envs)`
    - `_envs_idx` (a tensor of environment indices).
    - `envs_offset` for visual arrangement (grid layout in x–y).
    - Parallelization level `_para_level` depending on backend and `GS_PARA_LEVEL`.
  - Then calls `self._sim.build()` to build solvers and sensors.
  - Calls `_reset()` to reset internal state buffers.
  - Marks `_is_built = True`.
  - Optionally triggers a single `self._sim.step()` and `_reset()` to compile simulation kernels ahead of time.
  - Builds the visualizer, recorder manager, and optionally the FPS tracker.
- `step(update_visualizer=True, refresh_visualizer=True)`:
  - Asserts the scene is built and forward‑ready.
  - Calls `self._sim.step()` to advance physics and sensors by one simulation step of duration `SimOptions.dt`.
  - Increments its internal step counter `_t`.
  - Updates the visualizer and FPS tracker (if enabled).
  - Steps the recorder manager using the simulator’s `cur_step_global`.

**State and gradient methods**:

- `get_state()`:
  - Returns `self._sim.get_state()` – a `SimState` containing the full state of the simulation at the current step.
- `_step_grad()`:
  - Calls `self._sim.collect_output_grads()` and `self._sim._step_grad()` to perform a backward pass through the recorded unrolled simulation.

**Observation**: `Scene` is the natural engine boundary for an RL layer. It encapsulates simulator, entities, sensors, and visualization, and exposes `step()` and state queries, as well as asset loading via `add_entity`/`add_stage`.

---

### 1.4 Asset loader and morphs

While not exhaustively inspected here, the architecture guide and examples show:

- Morphs (`gs.morphs.*`) are defined under `genesis/options/morphs.py`.
- Supported morph types include:
  - Primitive shapes: `Box`, `Sphere`, `Plane`, etc.
  - Meshes: OBJ, GLB, PLY, STL.
  - Robot descriptions: `URDF`, `MJCF`, probably also USD robots.
- `add_entity`:
  - Validates combinations of morphs and materials:
    - URDF/MJCF/USD/Terrain morphs must pair with compatible material types (rigid, hybrid, etc.).
  - For file morphs, sets up defaults such as convexification for rigid meshes.
  - Delegates to the simulator’s `_add_entity`, which maps materials to solver‑specific entity types.

From the locomotion example (`examples/locomotion/go2_env.py`):

- A robot is loaded as:
  - `robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=..., quat=...))`.
- A plane is loaded as:
  - `scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))`.
- After adding entities, `scene.build(n_envs=num_envs)` is called to create a batched scene.
- DOF indexing is established by querying joints:
  - `motors_dof_idx = torch.tensor([robot.get_joint(name).dof_start for name in joint_names], ...)`.
  - This reveals that DOFs are indexed per joint by `dof_start`, similar to MuJoCo.

**Observation**:

- Genesis supports **URDF and MJCF** (and other mesh formats) and exposes joint indices and names via entity APIs (`get_joint`, `dof_start`).
- Contact surface details (materials, surfaces) are configured via `gs.materials.*` and `gs.surfaces.*` used when adding entities.

---

## 2. State access model

Genesis uses a combination of:

- `SimState` objects (from `genesis/engine/states/solvers.py`) to expose solver states.
- Entity‑level APIs accessed via `scene.entities` and solver‑specific entity subclasses.

From the Go2 locomotion example, we can infer:

- **Root pose**:
  - `base_pos = robot.get_pos()` returns a batched tensor `(n_envs, 3)`.
  - `base_quat = robot.get_quat()` returns `(n_envs, 4)`.
- **Velocities**:
  - `base_lin_vel = transform_by_quat(robot.get_vel(), inv_base_quat)` – `get_vel()` returns world‑frame velocity for the base link, batched over envs.
  - `base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)` – `get_ang()` returns angular velocity, again batched.
- **DOF state**:
  - `dof_pos = robot.get_dofs_position(motors_dof_idx)` returns joint positions for selected DOFs across all envs.
  - `dof_vel = robot.get_dofs_velocity(motors_dof_idx)` returns joint velocities.
- **Actions and control**:
  - PD gains are set as:
    - `robot.set_dofs_kp([kp] * num_actions, motors_dof_idx)`.
    - `robot.set_dofs_kv([kd] * num_actions, motors_dof_idx)`.
  - Per‑step control:
    - `robot.control_dofs_position(target_dof_pos[:, actions_dof_idx], slice(6, 18))` writes target positions for the motors, with an index range for controlled DOFs.

**Batching and tensors**:

- All these quantities are `torch` tensors with the first dimension being either:
  - `num_envs` (for batched simulations with `n_envs > 0`), or
  - effectively size 1 when `n_envs == 0` (the code guards for this in core solvers).
- Genesis exposes `gs.device` and `gs.tc_float` / `gs.tc_int` to ensure consistent device and dtype.

**Contacts and sensors**:

- While we haven’t exhaustively read sensor code, the architecture and README state:
  - Sensors (camera, raycast, IMU, etc.) are managed by `SensorManager` inside `Simulator`.
  - Sensor classes under `engine/sensors` operate over batched environments and maintain time‑buffered data.
  - `SensorManager.step()` is invoked at the end of each `Simulator.step()` to update sensor caches.
  - Contact forces and other solver states are accessible through entity/solver APIs, often using batched tensor outputs (e.g., link forces, contact points).

**Synchronous stepping**:

- `Scene.step()` unconditionally calls `self._sim.step()` once and then updates visualization and profiling.
- There is no notion of asynchronous stepping exposed at the scene level; for an RL workload, stepping is synchronous and deterministic given the current state and control commands written to solvers/entities.

---

## 3. Parallelization and vectorization model

Genesis explicitly supports vectorized environments via `Scene.build(n_envs=...)`:

- **Scene‑level parallelization**:
  - `build(n_envs, env_spacing, n_envs_per_row, center_envs_at_origin, ...)`:
    - Calls `_parallelize(...)` which:
      - Stores `scene.n_envs = n_envs`.
      - Sets `_B = max(1, n_envs)`, i.e., true batch size is at least 1.
      - Creates `_envs_idx` as a tensor `[0, 1, ..., _B-1]`.
      - Computes per‑env visualization offsets `envs_offset` using a grid layout (`env_spacing` and `n_envs_per_row`).
      - Determines a parallelization level `_para_level`:
        - On CPU: `PARA_LEVEL.NEVER` (no GPU‑style batching; loops may be serialized).
        - On GPU:
          - If `n_envs <= 1`: partial loop parallelization.
          - Otherwise: full parallelization (`PARA_LEVEL.ALL`).
        - Environment variable `GS_PARA_LEVEL` can override this.
  - The simulator subsequently uses `scene.n_envs`, `_B`, and `_para_level` to size solver data structures and control GPU parallelism.

- **Solver‑level parallelization**:
  - Rigid solver and others store `n_envs` and use it to shape internal arrays and kernels.
  - For example, `RigidSolver.build()` sets:
    - `self.n_envs = self.sim.n_envs`.
    - `self._B = self.sim._B`.
  - Many solvers branch logic based on `n_envs` (e.g., special handling when `n_envs == 0`) and limit features when batching is not supported (e.g., SF solver).
  - Constraint and force kernels reshape their outputs based on `(n_envs, n_items, ...)`.

- **Multiple scenes**:
  - There is no explicit restriction against multiple `Scene` instances; `Scene.build` registers each built scene in a global registry (`gs._scene_registry`), and scenes track themselves via weak references.
  - Each scene has its own simulator, solvers, entities, and GPU resources, so multiple scenes can exist concurrently, though resource contention is left to the user.

**Conclusion on parallelization**:

- Genesis supports **native vectorized environments** inside a single `Scene` via `n_envs > 0`, with batched solvers and global `SimState` and entity APIs exposed as batched `torch` tensors.
- This vectorization is implemented directly at the engine level, not via a separate vecenv abstraction, which is important when designing a manager‑based RL layer.

---

## 4. Asset import pipeline

Based on the README, architecture guide, and examples:

- **Supported formats**:
  - URDF, MJCF for robotics.
  - Mesh formats: OBJ, GLB, PLY, STL.
  - USD stages for complex assets and scenes.
- **Morphs and options**:
  - Morphs (file or primitive) are created via `gs.morphs.*` (e.g., `URDF`, `MJCF`, `USD`, `Box`, `Sphere`, `Plane`).
  - Solver‑agnostic geometry and initial pose are encoded in morphs.
  - Materials (`gs.materials.*`) and surfaces (`gs.surfaces.*`) define physics and contact properties.
- **Entity creation**:
  - `Scene.add_entity(morph, material, surface, ...)`:
    - Validates that the morph–material combination is allowed.
    - Adjusts surface attributes (e.g., smoothing, vis_mode) based on morph and material types.
    - For URDF/MJCF/USD morphs, ensures only rigid or hybrid materials are used; other material types are rejected.
    - For file morphs (`FileMorph`), may set defaults like `convexify` for rigid meshes.
    - Passes morph and material to `Simulator._add_entity`, which creates solver‑specific entity objects.
- **DOF indexing, joint naming, and joint interfaces**:
  - For robots loaded from URDF/MJCF, joints are exposed via:
    - `robot.get_joint(name)` which returns an object with fields including `dof_start`.
    - This implies that DOFs are packed into global arrays and each joint has a contiguous segment.
  - High‑level control APIs:
    - `set_dofs_kp`, `set_dofs_kv`, `control_dofs_position`, `control_dofs_velocity`, etc., work over batched DOF indices.
  - This mirrors MuJoCo’s indexing strategy and makes vectorized robot control natural.
- **Contact materials and sensors**:
  - Surfaces and materials determine contact behavior (friction, restitution, etc.).
  - Sensors (e.g., cameras, ray sensors, IMUs) are attached to entities via separate sensor options and are managed by the `SensorManager` in the `Simulator`.
  - Sensor configuration is done through options in `genesis/options/sensors/options.py` and through calls on the `Scene` / `Entity` API.

---

## 5. Genesis parallelization capabilities report

Putting the pieces together:

- **Vectorized environments**:
  - Supported via `Scene.build(n_envs > 0)`.
  - All state and control APIs operate on batched tensors of shape `(n_envs, ...)` (or `_B = max(1, n_envs)`).
- **Multiple robots per scene**:
  - Supported: the user can call `add_entity` multiple times with different morphs and materials. Each environment in the batch then contains replicated entities with appropriate offsets for visualization.
  - DOF indexing and control APIs are defined at the entity level, with environment dimension automatically accounted for.
- **Multiple scenes**:
  - Supported in principle: each `Scene` has its own `Simulator` and GPU resources, and scenes are tracked in a global registry.
  - Interaction or synchronization between scenes is not part of the engine core; each scene is independent.
- **Batched stepping**:
  - `Scene.step()` runs `Simulator.step()` once, which iterates over `_substeps` and steps all active solvers in a batched fashion for all environments.
  - Sensor updates and gradient bookkeeping are also batched.
- **Memory layout**:
  - Internally handled by Quadrants and solver implementations; at the Python level, users see batched `torch` tensors.
  - Solvers carefully consider `n_envs` and `_B` to shape their arrays and to choose specialized kernels (e.g., decisions about tiling and Cholesky factorization based on `n_envs` and DOFs).

---

## 6. Implications for a manager‑based RL layer

Based on the above:

- **Engine boundary**:
  - The natural boundary for a Genesis‑based RL env is the `Scene` object, plus a set of entity handles and possibly some helper wrappers around `SimState`.
  - `Scene.step()` is the physics stepping entry point and should be wrapped inside an environment step loop.
- **State access**:
  - All necessary robot state (root pose, joint states, velocities, contacts, sensors) is accessible through entity APIs and sensor managers, returning batched `torch` tensors on the correct device.
- **Vectorization**:
  - Genesis provides native vectorization at the engine level; an RL layer should treat the first tensor dimension as `num_envs` instead of building a separate multiprocessing‑based vector env.
- **Asset loading and DOF mapping**:
  - DOF indices and names should be resolved once (at env/task initialization) from entity joint APIs and then reused in observation/reward/action logic.
  - This mirrors mlab/mjlab’s pattern of precomputing joint indices and then using them in MDP term implementations.

This analysis will be used later to construct a comparative view against IsaacLab/mjlab and to define a disciplined engine boundary for GenesisLab, without assuming that Genesis behaves identically to MuJoCo. 

