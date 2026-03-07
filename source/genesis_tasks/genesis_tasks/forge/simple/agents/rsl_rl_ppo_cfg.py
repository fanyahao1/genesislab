"""RSL-RL PPO runner configs for Simple Go2 task (GenesisLab).

This configuration aligns with Genesis Forge's simple Go2 example training
parameters, adapted to GenesisLab's RSL-RL integration.
"""

from __future__ import annotations

from genesislab.utils.configclass import configclass
from genesis_rl.rsl_rl.configs import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SimpleGo2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Simple Go2 task.

    This configuration aligns with Genesis Forge's simple Go2 example:
    - actor_hidden_dims: [512, 256, 128]
    - critic_hidden_dims: [512, 256, 128]
    - num_steps_per_env: 24
    - learning_rate: 0.001
    - clip_param: 0.2
    - entropy_coef: 0.01
    - etc.
    """

    num_steps_per_env = 24
    max_iterations = 101  # Genesis Forge default
    save_interval = 100
    experiment_name = "simple"

    # Observation groups mapping – we use the single "policy" group for both
    # actor and critic, matching the current GenesisLab env wrapper.
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # Policy network (aligned with Genesis Forge)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],  # Same as Genesis Forge
        critic_hidden_dims=[512, 256, 128],  # Same as Genesis Forge
        activation="elu",  # Same as Genesis Forge
    )

    # PPO algorithm hyperparameters (aligned with Genesis Forge)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,  # Same as Genesis Forge
        entropy_coef=0.01,  # Same as Genesis Forge
        num_learning_epochs=5,  # Same as Genesis Forge
        num_mini_batches=4,  # Same as Genesis Forge
        learning_rate=1.0e-3,  # Same as Genesis Forge (0.001)
        schedule="adaptive",  # Same as Genesis Forge
        gamma=0.99,  # Same as Genesis Forge
        lam=0.95,  # Same as Genesis Forge
        desired_kl=0.01,  # Same as Genesis Forge
        max_grad_norm=1.0,  # Same as Genesis Forge
    )
