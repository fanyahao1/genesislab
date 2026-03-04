"""Stability test with random policy rollout.

This script runs a long rollout with random Gaussian actions to validate
environment stability and collect statistics.

Usage:
    python scripts/test_random_rollout.py
"""

import sys
from pathlib import Path

import torch

from genesis_tasks.velocity.go2.env import Go2VelocityEnv
from genesis_tasks.velocity.go2.cfg import Go2VelocityEnvCfg


def test_random_rollout():
    """Run stability test with random policy."""
    print("=" * 60)
    print("GenesisLab Random Rollout Stability Test")
    print("=" * 60)

    # Create config
    cfg = Go2VelocityEnvCfg()
    cfg.scene.num_envs = 64
    cfg.scene.backend = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nCreating environment with {cfg.scene.num_envs} env(s)...")
    try:
        env = Go2VelocityEnv(cfg)
        print(f"✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Initialize tracking
    print("\nRunning random policy rollout (1000 steps)...")
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = torch.zeros(env.num_envs, device=env.device)
    current_episode_length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Reset
    obs, info = env.reset()

    try:
        for step in range(1000):
            # Random Gaussian actions
            action = torch.randn(
                (env.num_envs, env.action_manager.total_action_dim),
                device=env.device
            )
            action = torch.clamp(action, -1.0, 1.0)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            # Accumulate rewards
            current_episode_reward += reward
            current_episode_length += 1
            total_reward += reward

            # Handle episode endings
            done = terminated | truncated
            if done.any():
                done_envs = done.nonzero(as_tuple=False).squeeze(-1)

                # Record episode statistics
                for env_idx in done_envs:
                    episode_rewards.append(current_episode_reward[env_idx].item())
                    episode_lengths.append(current_episode_length[env_idx].item())

                # Reset episode tracking for done envs
                current_episode_reward[done_envs] = 0.0
                current_episode_length[done_envs] = 0

                # Reset environments
                obs, info = env.reset(env_ids=done_envs)

            # Progress update
            if (step + 1) % 200 == 0:
                avg_reward = total_reward.mean().item()
                num_episodes = len(episode_rewards)
                print(f"  Step {step + 1}/1000: avg reward = {avg_reward:.4f}, episodes = {num_episodes}")

        # Record any incomplete episodes
        for env_idx in range(env.num_envs):
            if current_episode_length[env_idx] > 0:
                episode_rewards.append(current_episode_reward[env_idx].item())
                episode_lengths.append(current_episode_length[env_idx].item())

        print(f"✓ Rollout completed successfully")

    except Exception as e:
        print(f"✗ Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print statistics
    print("\n" + "=" * 60)
    print("Rollout Statistics")
    print("=" * 60)

    if len(episode_rewards) > 0:
        episode_rewards_tensor = torch.tensor(episode_rewards)
        episode_lengths_tensor = torch.tensor(episode_lengths)

        print(f"\nEpisode Statistics:")
        print(f"  - Total episodes: {len(episode_rewards)}")
        print(f"  - Average episode reward: {episode_rewards_tensor.mean().item():.4f} ± {episode_rewards_tensor.std().item():.4f}")
        print(f"  - Min episode reward: {episode_rewards_tensor.min().item():.4f}")
        print(f"  - Max episode reward: {episode_rewards_tensor.max().item():.4f}")
        print(f"  - Average episode length: {episode_lengths_tensor.mean().item():.2f} ± {episode_lengths_tensor.std().item():.2f}")
        print(f"  - Min episode length: {episode_lengths_tensor.min().item()}")
        print(f"  - Max episode length: {episode_lengths_tensor.max().item()}")
    else:
        print("  No completed episodes recorded")

    print(f"\nOverall Statistics:")
    print(f"  - Total steps: 1000")
    print(f"  - Average step reward: {total_reward.mean().item() / 1000:.4f}")
    print(f"  - Total reward sum: {total_reward.sum().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ Stability test completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_random_rollout()
    sys.exit(0 if success else 1)
