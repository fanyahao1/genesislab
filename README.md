## GenesisLab

GenesisLab is a lightweight framework for running and experimenting with reinforcement-learning tasks and environments.

## Features
- **Task library**: Predefined tasks and environments for fast experimentation.
- **Scriptable workflows**: Utility scripts to set up, run, and evaluate experiments.

## Installation

GenesisLab builds on top of the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine,
which is distributed on PyPI.

1. **Create / activate a Python environment** (recommended: Python 3.10+).
2. **Install Genesis**:

   ```bash
   pip install genesis
   ```

3. **Install GenesisLab (editable)** from the project root:

   ```bash
   pip install -e .
   ```

This will make `genesislab` and the `genesis_tasks` package importable in your environment.

## Quick Start
- **Engine smoke test**:

  ```bash
  cd genesislab
  python scripts/test/test_engine.py --backend cpu --num-envs 4
  ```

- **Go2 velocity task sanity check**:

  ```bash
  cd genesislab
  python scripts/test/test_env.py
  python scripts/test/test_env_vectorized.py
  python scripts/test/test_random_rollout.py
  ```

These scripts exercise the Genesis binding and the manager-based RL task for the Unitree Go2 robot.

## License
This project is provided as-is; see the main repository’s license file for details.