"""Minimal sanity check test for GenesisLab environments.

This script performs a basic validation of environment construction,
reset, and stepping with random actions.

Usage:
    python scripts/test_env.py
"""

import sys
from pathlib import Path

import torch

from genesis_tasks.velocity.go2.env import Go2VelocityEnv
from genesis_tasks.velocity.go2.cfg import Go2VelocityEnvCfg


def test_env():
    """Run minimal environment test."""
    print("=" * 60)
    print("GenesisLab Environment Test - Single Environment")
    print("=" * 60)

    # Create minimal config
    cfg = Go2VelocityEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.backend = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nCreating environment with {cfg.scene.num_envs} env(s)...")
    try:
        env = Go2VelocityEnv(cfg)
        print(f"✓ Environment created successfully")
        print(f"  - Device: {env.device}")
        print(f"  - Action dim: {env.action_manager.total_action_dim}")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test reset
    print("\nTesting reset...")
    try:
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test stepping
    print("\nTesting step (100 steps with random actions)...")
    try:
        total_reward = 0.0
        done_count = 0

        for step in range(100):
            # Random actions in [-1, 1]
            action = torch.randn(
                (env.num_envs, env.action_manager.total_action_dim),
                device=env.device
            )
            action = torch.clamp(action, -1.0, 1.0)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward.mean().item()
            if terminated.any() or truncated.any():
                done_count += 1
                # Reset if needed
                reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_envs) > 0:
                    obs, info = env.reset(env_ids=reset_envs)

            if (step + 1) % 20 == 0:
                print(f"  Step {step + 1}/100: avg reward = {total_reward / (step + 1):.4f}")

        print(f"✓ Stepping successful")
        print(f"  - Total steps: 100")
        print(f"  - Average reward: {total_reward / 100:.4f}")
        print(f"  - Done count: {done_count}")
    except Exception as e:
        print(f"✗ Stepping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_env()
    sys.exit(0 if success else 1)
