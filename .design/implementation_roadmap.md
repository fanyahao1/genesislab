## GenesisLab implementation roadmap (high‑level, no concrete code)

This roadmap describes a staged plan for implementing GenesisLab on top of Genesis, guided by the reverse‑engineered architectures of IsaacLab, mjlab, and Genesis. It focuses on ordering, dependencies, and validation, not on specific APIs.

---

## 1. Phase A – Foundational Genesis validation

**Goal**: Confirm Genesis has the capabilities needed for a manager‑based RL framework and establish minimal patterns for using it.

- **A1. Engine validation**
  - Load a representative robot (e.g., Go2) and a simple terrain (plane) from URDFs.
  - Build a `Scene` with `n_envs > 1` and confirm:
    - Stable stepping with `Scene.step()` over long horizons.
    - Batched state access via entity APIs (poses, velocities, joint states).
    - Deterministic resets using entity joint and root state setters.
  - Verify that sensors (at least one camera or ray sensor) function correctly in batched mode.

- **A2. Control and timing validation**
  - Establish a clear mapping between:
    - `SimOptions.dt` and `substeps`.
    - Desired RL control frequency (e.g., 50 Hz).
  - Implement a simple control loop in user code (outside GenesisLab) that:
    - Applies PD control to joints.
    - Steps the scene with a chosen decimation (multiple `Scene.step()` per control decision).
    - Tracks basic metrics (e.g., base height, velocity) to verify physical behavior.

Outcome: confidence that Genesis’s `Scene`, entities, and solvers can support vectorized RL workloads with deterministic reset and state access.

---

## 2. Phase B – Engine binding layer

**Goal**: Introduce a thin, well‑scoped binding that standardizes how RL code interacts with Genesis scenes and entities.

- **B1. Minimal engine binding prototype**
  - Define a binding object that:
    - Owns a Genesis `Scene` and a set of “agent” entities.
    - Encapsulates `Scene.build(n_envs)` and `Scene.step()`.
    - Offers methods to:
      - Reset specified environments deterministically.
      - Get root pose, joint states, and contacts for agents.
      - Write joint targets and other control inputs to entities.
  - Start with rigid‑only use cases (locomotion/manipulation) to keep scope manageable.

- **B2. Validation of binding**
  - Re‑implement the existing `Go2Env` example logic on top of the binding, but:
    - Replace direct calls to entity methods with binding methods where appropriate.
    - Ensure the behavior matches the original example (up to expected numeric noise).

Outcome: a stable “engine service” layer that can be consumed by a manager‑based RL env without leaking solver internals.

---

## 3. Phase C – Manager system implementation

**Goal**: Build a Genesis‑compatible manager system inspired by IsaacLab/mjlab, but independent of MuJoCo and tailored to Genesis.

- **C1. Core manager infrastructure**
  - Implement generic, engine‑agnostic manager infrastructure:
    - Term config base (similar to `ManagerTermBaseCfg`).
    - Term base class (similar to `ManagerTermBase`).
    - Manager base class (similar to `ManagerBase`) with:
      - `reset(env_ids)`.
      - `compute(...)`.
      - Term preparation and `Scene`/entity resolution via the binding layer.

- **C2. Essential managers**
  - Implement minimal versions of:
    - Observation manager:
      - Observation groups, term registration, noise, concatenation.
    - Reward manager:
      - Weighted terms, dt scaling, episodic sums.
    - Termination manager:
      - Termination vs truncation flags.
    - Action manager:
      - Mapping from flat action tensors to per‑entity controls via the binding.
  - Defer more advanced managers (curriculum, events, metrics) until the basics are working.

- **C3. Manager‑based env core**
  - Implement a manager‑based env core that:
    - Orchestrates managers in the step pipeline:
      - Process actions → decimated stepping via binding → termination/rewards → optional events/commands → observations.
    - Manages per‑env episode counters and buffers.
    - Exposes a batched RL interface (`reset`, `step`, `num_envs`, `observation_spec`, `action_spec`).

Outcome: a Genesis‑native counterpart of mjlab’s `ManagerBasedRlEnv`, but relying on the binding layer rather than directly on the engine.

---

## 4. Phase D – Configclass integration

**Goal**: Make GenesisLab fully config‑driven using the existing `configclass` system, similar to IsaacLab and mjlab’s outer integration.

- **D1. Internal config schema**
  - Define internal, engine‑agnostic config structures for:
    - Engine binding (scene composition, entity selection, `n_envs`, visualization hints).
    - Manager system (per‑manager term configs and observation/reward/termination/command settings).
  - Ensure these configs can be instantiated without `configclass` to support programmatic use.

- **D2. Mapping from `@configclass`**
  - Define one or more configclass trees that:
    - Represent user‑facing task configs (e.g., “Go2 velocity MDP”).
    - Are agnostic to the underlying engine (apart from referencing Genesis‑specific entity names).
  - Implement mapping code that:
    - Converts configclass objects into the internal config structures.
    - Resolves entity and DOF names using the engine binding and Genesis `Scene`.

- **D3. Example tasks**
  - Port a minimal subset of tasks:
    - One locomotion velocity task for a quadruped.
    - Optionally one manipulation task.
  - For each:
    - Define configclass‑based configs.
    - Validate that the mapping and resulting environment behave as expected.

Outcome: GenesisLab environments can be instantiated purely from configclass‑based configs, aligned with existing tooling in this repo.

---

## 5. Phase E – Validation and RL integration

**Goal**: Validate GenesisLab in real RL training scenarios and refine design where necessary.

- **E1. Minimal RL benchmarks**
  - Run simple baselines (e.g., PPO) on GenesisLab tasks:
    - Monitor stability of training and reward signals.
    - Inspect manager logs for consistency (reward breakdown, terminations).
  - Compare performance and qualitative behavior to:
    - Equivalent tasks in mjlab, if available.
    - The hand‑written `Go2Env` baseline.

- **E2. Stress tests**
  - Scale to larger `n_envs` and longer horizons to probe:
    - Memory usage and performance of the engine binding and manager layers.
    - Sensitivity to randomization and curriculum.
  - Test:
    - Deterministic resets and reproducibility under different seeds.
    - Interaction with various Genesis backends (GPU/CPU where practical).

- **E3. API hardening**
  - Based on RL experiments:
    - Refine the manager API and binding interface where friction appears.
    - Clarify which concepts are part of the stable public API vs internal.

Outcome: evidence that GenesisLab can be used for practical RL research, with known performance and semantic properties.

---

## 6. Phase F – Extensions and multi‑solver support

**Goal**: Extend GenesisLab beyond rigid‑only tasks to leverage Genesis’s broader solver capabilities, while preserving the architectural boundaries.

- **F1. Soft‑body and fluid tasks**
  - Identify prototype tasks (e.g., deformable manipulation, fluid interaction).
  - Extend:
    - Engine binding layer with solver‑specific state queries and control hooks where needed.
    - Manager term libraries (observations and rewards) to account for soft‑body / fluid behaviors.

- **F2. Multi‑agent and multi‑entity support**
  - Generalize:
    - Engine binding and manager configs to handle multiple controllable entities per env.
    - Observation and action grouping across agents.

- **F3. Advanced managers**
  - Add:
    - Curriculum manager tied to solver and asset parameters.
    - Event manager for staged randomization and dynamic environment changes.
    - Metrics manager for richer logging and analysis.

Outcome: GenesisLab becomes a general, extensible framework for a broad class of RL tasks on Genesis, still maintaining a clean separation from engine internals.

---

## 7. Guiding principles (constraints)

Throughout all phases:

- Preserve a **clean engine boundary**:
  - Avoid leaking Genesis solver internals into manager code or user‑facing APIs.
  - Keep all direct `Scene`/solver calls inside the binding layer and term implementations.
- Maintain **config‑driven construction**:
  - Avoid environment setups that cannot be reproduced or modified via configuration.
- Respect **vectorization and scalability**:
  - Design data structures and APIs with `n_envs` as a first‑class dimension.
  - Validate performance and correctness across a range of `n_envs`.
- Avoid **premature specialization**:
  - Do not hard‑code rigid‑only or MuJoCo‑style assumptions.
  - Keep mapping layers flexible so that new solvers and assets can be incorporated later.

This roadmap should be revisited and refined as early phases surface new constraints or opportunities in Genesis and in the target research workloads. 

