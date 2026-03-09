# GenesisLab: Fast and simple to train a robot.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](./LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![OS Linux](https://img.shields.io/badge/OS-Linux-green.svg)
![Genesis](https://img.shields.io/badge/Genesis-Physics%20Engine-orange.svg)
![RSL-RL](https://img.shields.io/badge/RSL--RL-integrated-ff8c00.svg)

<p align="center">
<img src="./.docs/assets/intro/genesislab_banner.png" width="100%">
</p>

**GenesisLab** is a lightweight **robotics reinforcement learning task suite** built on top of the **Genesis physics engine**.  
It provides a compact framework for **developing RL environments, running large-scale vectorized simulations, and validating observation and reward pipelines**.

The project is intended as a **fast experimentation playground for robotics RL**, emphasizing clarity, modularity, and rapid iteration.

# Demo

Example: **Unitree Go2 velocity tracking policy trained with RSL-RL**

<!-- <p align="center">
<img src="./.docs/assets/demo/go2_velocity.gif" width="70%">
</p> -->

Additional environments, benchmarks, and demonstrations will be added as the project evolves.

# Design Philosophy

GenesisLab focuses on providing a **minimal yet practical reinforcement learning experimentation framework** for robotics research.  
Rather than building a large infrastructure, the system emphasizes **readability, modular design, and rapid development cycles**.

<details>
<summary>Core principles</summary>

- **Minimal but complete**  
  The framework includes only the essential components required to construct and train RL environments, avoiding unnecessary abstraction while remaining suitable for real experiments.

- **Readable and hackable implementation**  
  The codebase is intentionally compact so that researchers can understand the system quickly and modify reward functions, observations, or task logic without navigating complex infrastructure.

- **Fast experimentation cycle**  
  Environment templates, simple interfaces, and lightweight configuration allow rapid iteration when designing new RL tasks.

- **Experiment validation first**  
  Built-in debugging utilities and validation scripts help verify simulation stability, observation correctness, and reward behavior before large-scale training.
</details>

# Key Features

- **Lightweight RL task framework**  
  Provides a minimal structure for defining reinforcement learning environments while keeping the implementation small and easy to understand.

- **Vectorized simulation support**  
  Enables batched environment execution for efficient data collection and high-throughput reinforcement learning training.

- **Integrated debugging utilities**  
  Includes scripts for validating physics bindings, environment stepping, observation pipelines, and random rollout behavior.

- **Research-friendly architecture**  
  Designed to allow straightforward implementation of new robotics tasks, reward functions, and observation structures.

# Supported Hardware

GenesisLab inherits hardware compatibility directly from the **Genesis physics engine**.  
Any device supported by Genesis is therefore supported by GenesisLab.

<details>
<summary>Typical supported configurations include:</summary>

- **CPU execution** for debugging, development, and lightweight experiments.
- **GPU-accelerated simulation** for large-scale vectorized reinforcement learning.
- **Parallel multi-environment training** for high-throughput policy optimization.

Hardware compatibility follows the official Genesis runtime environment and backend implementations.

</details>

# Installation

### 1. Create environment

```bash
conda create -n genesislab python=3.10
conda activate genesislab
````

### 2. Install Genesis

```bash
pip install genesis-world
```

### 3. Install GenesisLab

```bash
bash scripts/setup/setup_ext.sh
```

This installs the `genesislab` package in **editable mode** for development.


# Training Example

Train a **Go2 flat velocity tracking policy** using the integrated RSL-RL pipeline.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --env-id Genesis-Velocity-Flat-Go2-v0 \
  --num-envs 4096 \
  --num-iters 3000
```

## Policy Visualization

Render a trained policy and visualize the behavior in a simulation window.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --env-id Genesis-Velocity-Flat-Go2-v0 \
  --window \
  --num-envs 1 \
  --checkpoint <PATH_TO_CHECKPOINT>
```

<detail>
<summary>Engine smoke test</summary>
Verify that the Genesis backend and Python bindings work correctly.

```bash
python scripts/test/test_engine.py --backend cpu --num-envs 4
```
</detail>

<detail>
<summary>Environment validation</summary>
Run sanity checks to verify stepping logic and vectorized rollouts.

```bash
python scripts/test/test_env.py
python scripts/test/test_env_vectorized.py
python scripts/test/test_random_rollout.py
```
</detail>

# Citation

If GenesisLab is used in academic research or open-source projects, please consider citing or referencing this repository.

```
@software{zheng2026@genesislab,
  author = {Ziang Zheng},
  title = {GenesisLab: Fast and simple to train a robot..},
  url = {https://github.com/Renforce-Dynamics/genesislab},
  year = {2026}
}
```

# License

This project is released under the **BSD-3-Clause License**.
See the `LICENSE` file for details.


# Appendix

<details>
<summary>Project directory overview</summary>

```
genesislab
├── source/genesislab
│   ├── cli/                 # command line utilities
│   ├── engine/              # Genesis physics bindings
│   ├── envs/                # RL environment wrappers
│   └── tasks/               # task definitions
│
├── scripts
│   ├── setup
│   │   └── setup_ext.sh
│   └── test
│       ├── test_engine.py
│       ├── test_env.py
│       ├── test_env_vectorized.py
│       └── test_random_rollout.py
│
├── README.md
└── pyproject.toml
```

</details>

<details>
<summary>Adding new RL tasks</summary>

GenesisLab is designed to make task development straightforward.
A typical workflow for implementing a new RL environment is:

1. Duplicate an existing task as a template.
2. Implement observation and reward logic.
3. Add a validation script in `scripts/test`.
4. Run sanity checks before launching large-scale training.

Recommended validation scripts:

```
test_env.py
test_env_vectorized.py
test_random_rollout.py
```

These scripts help ensure environment reset logic, observation generation, and rollout stability behave correctly.

</details>

<details>
<summary>Planned extensions</summary>

* additional locomotion environments
* manipulation tasks based on Genesis
* improved debugging and visualization utilities
* additional reinforcement learning algorithm integrations

</details>
