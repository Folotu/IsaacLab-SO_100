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
    """PPO runner config for camera-based SO-100 cube lift with asymmetric actor-critic.

    Asymmetric observation spaces:
    - Actor obs: 537-dim (joint_pos(6) + joint_vel(6) + visual_features(512) + target(7) + action(6))
    - Critic obs: 28-dim (joint_pos(6) + joint_vel(6) + object_position(3) + target(7) + action(6))

    The critic receives privileged ground-truth object position (matching the state-based
    policy exactly), while the actor learns from 512-dim penultimate ResNet18 features.

    Actor hidden dims [512, 256, 128] handle the 537-dim visual input.
    Critic hidden dims [256, 128, 64] match the state-based LiftCubePPORunnerCfg
    (28-dim input is the same dimensionality as the state-based policy).

    RSL-RL reads num_obs and num_privileged_obs dynamically from the env wrapper.
    The RslRlVecEnvWrapper sets num_privileged_obs from the "critic" observation group.
    """
    num_steps_per_env = 24
    max_iterations = 1500
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