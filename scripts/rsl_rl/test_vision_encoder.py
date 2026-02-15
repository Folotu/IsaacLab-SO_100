# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision encoder integration test for SO-100 camera-based training.

Validates that:
1. Camera env produces 1025-dim observations (not 28-dim state-based)
2. Frozen ResNet18 encoder produces non-zero, varying features
3. RslRlVecEnvWrapper correctly reads observation dimensions
4. Multiple environment steps work without errors

Usage:
    cd IsaacLab-SO_100
    python scripts/rsl_rl/test_vision_encoder.py --task SO-ARM100-Lift-Cube-Camera-v0 --num_envs 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test vision encoder integration for SO-100 camera-based training.")
parser.add_argument("--task", type=str, default="SO-ARM100-Lift-Cube-Camera-v0", help="Camera-enabled task name.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments (small for dev testing on T4).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras for camera-based tasks (required for TiledCamera rendering)
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# Expected observation dimension:
# joint_pos(6) + joint_vel(6) + visual_features(1000) + target(7) + action(6) = 1025
EXPECTED_OBS_DIM = 1025


def main():
    """Run vision encoder integration test."""
    num_envs = args_cli.num_envs
    passed = True

    print("=" * 70)
    print("  VISION ENCODER INTEGRATION TEST")
    print("=" * 70)
    print(f"  Task: {args_cli.task}")
    print(f"  Num envs: {num_envs}")
    print(f"  Expected obs dim: {EXPECTED_OBS_DIM}")
    print("=" * 70)

    # --- Step 1: Create environment via gym.make ---
    print("\n[1/6] Creating environment...")
    # Resolve env config class from gym registration and set num_envs
    import importlib
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    env_cfg = getattr(module, class_name)()
    env_cfg.scene.num_envs = num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print(f"  Environment created: {env.unwrapped.num_envs} envs on {env.unwrapped.device}")

    # --- Step 2: Wrap with RslRlVecEnvWrapper (same as train.py) ---
    print("\n[2/6] Wrapping with RslRlVecEnvWrapper...")
    env = RslRlVecEnvWrapper(env, clip_actions=True)
    print(f"  num_obs: {env.num_obs}")
    print(f"  num_actions: {env.num_actions}")

    # Assert num_obs matches expected dimension
    if env.num_obs != EXPECTED_OBS_DIM:
        print(f"\n  FAIL: num_obs = {env.num_obs}, expected {EXPECTED_OBS_DIM}")
        print(f"  This suggests the encoder is NOT active (state-based would be ~28).")
        passed = False
    else:
        print(f"  PASS: num_obs = {EXPECTED_OBS_DIM} (encoder is active)")

    # --- Step 3: Get initial observations ---
    print("\n[3/6] Getting initial observations...")
    obs, extras = env.get_observations()
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dtype: {obs.dtype}")

    # Assert obs shape
    expected_shape = (env.num_envs, EXPECTED_OBS_DIM)
    if obs.shape != torch.Size(expected_shape):
        print(f"\n  FAIL: obs.shape = {obs.shape}, expected {expected_shape}")
        passed = False
    else:
        print(f"  PASS: obs shape matches ({expected_shape[0]}, {expected_shape[1]})")

    # Assert obs contains non-zero values
    if torch.all(obs == 0):
        print(f"\n  FAIL: All observations are zero (encoder may not have run)")
        passed = False
    else:
        print(f"  PASS: Observations contain non-zero values")

    # --- Step 4: Analyze visual features portion ---
    print("\n[4/6] Analyzing visual features (columns 12:1012)...")
    # Obs layout: joint_pos(6) + joint_vel(6) + visual_features(1000) + target(7) + action(6)
    visual_features = obs[:, 12:1012]
    vf_mean = visual_features.mean().item()
    vf_std = visual_features.std().item()
    vf_min = visual_features.min().item()
    vf_max = visual_features.max().item()

    print(f"  Visual features shape: {visual_features.shape}")
    print(f"  Mean: {vf_mean:.4f}")
    print(f"  Std:  {vf_std:.4f}")
    print(f"  Min:  {vf_min:.4f}")
    print(f"  Max:  {vf_max:.4f}")

    if vf_std <= 0:
        print(f"\n  FAIL: Visual features have zero variance (all identical values)")
        print(f"  This suggests the encoder is not producing meaningful features.")
        passed = False
    else:
        print(f"  PASS: Visual features have non-zero variance (std={vf_std:.4f})")

    # Also check proprioceptive portions
    joint_pos = obs[:, 0:6]
    joint_vel = obs[:, 6:12]
    target = obs[:, 1012:1019]
    actions = obs[:, 1019:1025]
    print(f"\n  Proprioceptive breakdown:")
    print(f"    joint_pos [0:6]:   mean={joint_pos.mean().item():.4f}, std={joint_pos.std().item():.4f}")
    print(f"    joint_vel [6:12]:  mean={joint_vel.mean().item():.4f}, std={joint_vel.std().item():.4f}")
    print(f"    target [1012:1019]: mean={target.mean().item():.4f}, std={target.std().item():.4f}")
    print(f"    actions [1019:1025]: mean={actions.mean().item():.4f}, std={actions.std().item():.4f}")

    # --- Step 5: Step environment 5 times with random actions ---
    print("\n[5/6] Stepping environment 5 times with random actions...")
    for step in range(5):
        random_actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        random_actions = random_actions.clamp(-1.0, 1.0)
        obs, rewards, dones, extras = env.step(random_actions)

        # Quick check on each step's visual features
        vf = obs[:, 12:1012]
        print(f"  Step {step + 1}: obs_shape={obs.shape}, vf_mean={vf.mean().item():.4f}, "
              f"vf_std={vf.std().item():.4f}, reward_mean={rewards.mean().item():.4f}")

    print(f"  PASS: 5 steps completed without errors")

    # --- Step 6: Final visual feature statistics ---
    print("\n[6/6] Final observation statistics...")
    final_vf = obs[:, 12:1012]
    print(f"  Final visual features:")
    print(f"    Mean: {final_vf.mean().item():.4f}")
    print(f"    Std:  {final_vf.std().item():.4f}")
    print(f"    Min:  {final_vf.min().item():.4f}")
    print(f"    Max:  {final_vf.max().item():.4f}")

    # Final verdict
    print("\n" + "=" * 70)
    if passed:
        print("  VISION ENCODER INTEGRATION TEST: PASSED")
    else:
        print("  VISION ENCODER INTEGRATION TEST: FAILED")
    print("=" * 70)

    # Clean up
    env.close()

    return passed


if __name__ == "__main__":
    success = main()
    # close sim app
    simulation_app.close()
    if not success:
        sys.exit(1)
