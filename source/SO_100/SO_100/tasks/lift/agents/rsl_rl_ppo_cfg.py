# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "so_arm100_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        noise_std_type="log",  # Pass through as kwargs to fix negative std issue
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftCubeCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Production PPO config for camera-based SO-100 cube lift.

    3-frame stacked spatial softmax + big batch + critic with velocity.
    Designed for convergence in ~1500-3000 iterations from scratch.

    Key changes from previous configs:
    - num_steps_per_env=96 (4x bigger batch: 49K transitions vs 12K)
    - num_learning_epochs=10 (extract more gradient from each batch)
    - num_mini_batches=8 (stable VRAM with larger buffer)
    - learning_rate=1e-4 (larger batch supports higher LR)
    - gamma=0.99, lam=0.95 (longer horizon, larger batch compensates for lower lambda)

    Asymmetric observation spaces:
    - Actor obs: 1561-dim (joint_pos(6) + joint_vel(6) + stacked_visual(1536) + target(7) + action(6))
    - Critic obs: 31-dim (joint_pos(6) + joint_vel(6) + object_pos(3) + object_vel(3) + target(7) + action(6))
    """
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "so_arm100_lift_camera"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        noise_std_type="log",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )