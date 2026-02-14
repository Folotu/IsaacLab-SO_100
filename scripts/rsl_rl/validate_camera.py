# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone script to validate TiledCamera rendering in the SO-100 lift environment.

Creates a camera-enabled environment, steps it to let the scene settle, then saves
rendered RGB images to disk for manual inspection. Also prints camera intrinsics
and per-environment mean pixel intensity for brightness consistency checking.

Modes:
    Normal:       Save images and print diagnostics for a small number of environments.
    --memory_only: Normal flow + print GPU memory usage after each phase.
    --scaling_test: Profile GPU memory at increasing env counts [4, 64, 128, 256, 512, 1024].

Usage:
    # Normal validation (4 envs, save images)
    python validate_camera.py --task SO-ARM100-Lift-Cube-Camera-v0 --num_envs 4 --output_dir /tmp/cam_test

    # Normal + memory reporting
    python validate_camera.py --task SO-ARM100-Lift-Cube-Camera-v0 --num_envs 4 --memory_only

    # GPU memory scaling test (creates/destroys envs at increasing counts)
    python validate_camera.py --task SO-ARM100-Lift-Cube-Camera-v0 --scaling_test
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate TiledCamera rendering in the SO-100 environment.")
parser.add_argument("--task", type=str, default="SO-ARM100-Lift-Cube-Camera-v0", help="Camera-enabled task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments (small for validation).")
parser.add_argument("--output_dir", type=str, default="camera_validation_output", help="Directory to save rendered images.")
parser.add_argument("--num_steps", type=int, default=10, help="Number of sim steps before capturing (lets scene settle).")
parser.add_argument("--scaling_test", action="store_true", default=False,
                    help="Run GPU memory scaling test at [4, 64, 128, 256, 512, 1024] envs. Skips normal image-save flow.")
parser.add_argument("--memory_only", action="store_true", default=False,
                    help="Normal validation flow but additionally prints GPU memory usage after each phase.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras for validation
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gc
import math
import os
import traceback

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401

# Memory budget: 60GB peak leaves 4GB headroom on a 64GB GPU for optimizer/gradients
MEMORY_BUDGET_GB = 60.0
SCALING_ENV_COUNTS = [4, 64, 128, 256, 512, 1024]


def _gpu_mem_gb() -> float:
    """Return peak GPU memory allocated in GB (since last reset)."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _gpu_mem_current_gb() -> float:
    """Return current GPU memory allocated in GB."""
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _print_memory(label: str):
    """Print current and peak GPU memory with a label."""
    current = _gpu_mem_current_gb()
    peak = _gpu_mem_gb()
    print(f"  [MEM] {label}: current={current:.2f} GB, peak={peak:.2f} GB")


def run_scaling_test():
    """Profile GPU memory at increasing environment counts.

    For each count in SCALING_ENV_COUNTS:
      1. Create a camera environment with that many envs
      2. Run 5 sim steps to fully initialize the rendering pipeline
      3. Record peak GPU memory
      4. Close the environment and clear cache
      5. Report results in a formatted table

    If environment creation fails (OOM), the error is caught and that count
    is reported as "OOM". Scaling stops at the first OOM.

    Falls back to single-environment creation at the target count (1024) if
    the loop approach fails due to simulator reinitialization issues.
    """
    from SO_100.tasks.lift.lift_env_cfg import SoArm100CubeLiftCameraEnvCfg

    print("\n" + "=" * 70)
    print("  GPU MEMORY SCALING TEST")
    print("=" * 70)
    print(f"  Task: {args_cli.task}")
    print(f"  Env counts to test: {SCALING_ENV_COUNTS}")
    print(f"  Memory budget: {MEMORY_BUDGET_GB:.0f} GB (64 GB GPU - 4 GB headroom)")
    print(f"  Steps per count: 5 (to initialize rendering pipeline)")
    print("=" * 70)

    results = []  # list of (count, peak_gb, status)
    max_safe_count = 0
    loop_failed = False

    for count in SCALING_ENV_COUNTS:
        print(f"\n--- Testing {count} environments ---")

        # Reset peak memory tracker before this iteration
        torch.cuda.reset_peak_memory_stats()
        initial_mem = _gpu_mem_current_gb()
        print(f"  Initial GPU memory: {initial_mem:.2f} GB")

        try:
            # Create env config with the target num_envs
            env_cfg = SoArm100CubeLiftCameraEnvCfg()
            env_cfg.scene.num_envs = count

            # Create environment
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
            print(f"  Created {env.unwrapped.num_envs} envs on {env.unwrapped.device}")

            # Reset and step to fully initialize rendering
            env.reset()
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for step in range(5):
                env.step(actions)

            # Record peak memory after stepping
            peak_gb = _gpu_mem_gb()
            status = "OK" if peak_gb < MEMORY_BUDGET_GB else "RISK"
            results.append((count, peak_gb, status))

            if status == "OK":
                max_safe_count = count

            print(f"  Peak GPU memory: {peak_gb:.2f} GB [{status}]")

            # Close environment and free memory
            env.close()
            del env, env_cfg, actions
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                print(f"  OOM at {count} envs: {error_msg[:200]}")
                peak_gb = _gpu_mem_gb()
                results.append((count, peak_gb, "OOM"))
                # Try to clean up
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  Stopping scaling test (OOM reached).")
                break
            else:
                # Non-OOM RuntimeError -- may be simulator reinitialization issue
                print(f"  RuntimeError at {count} envs: {error_msg[:200]}")
                traceback.print_exc()
                if count == SCALING_ENV_COUNTS[0]:
                    # First iteration failed -- loop approach does not work
                    loop_failed = True
                    print("\n  [WARN] Loop approach failed on first iteration.")
                    print("  [WARN] Falling back to single-environment test at target count.")
                    break
                else:
                    results.append((count, 0.0, "ERR"))
                    print(f"  Stopping scaling test (runtime error).")
                    break

        except Exception as e:
            print(f"  Unexpected error at {count} envs: {e}")
            traceback.print_exc()
            if count == SCALING_ENV_COUNTS[0]:
                loop_failed = True
                print("\n  [WARN] Loop approach failed on first iteration.")
                print("  [WARN] Falling back to single-environment test at target count.")
                break
            else:
                results.append((count, 0.0, "ERR"))
                break

    # Fallback: if loop approach fails, try a single creation at target count
    if loop_failed:
        target_count = SCALING_ENV_COUNTS[-1]  # 1024
        print(f"\n--- Fallback: Single test at {target_count} environments ---")
        torch.cuda.reset_peak_memory_stats()

        try:
            env_cfg = SoArm100CubeLiftCameraEnvCfg()
            env_cfg.scene.num_envs = target_count
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
            env.reset()
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for step in range(5):
                env.step(actions)

            peak_gb = _gpu_mem_gb()
            status = "OK" if peak_gb < MEMORY_BUDGET_GB else "RISK"
            results.append((target_count, peak_gb, status))
            if status == "OK":
                max_safe_count = target_count

            print(f"  Peak GPU memory at {target_count} envs: {peak_gb:.2f} GB [{status}]")

            env.close()
            del env, env_cfg, actions
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                peak_gb = _gpu_mem_gb()
                results.append((target_count, peak_gb, "OOM"))
                print(f"  OOM at {target_count} envs: {error_msg[:200]}")
            else:
                results.append((target_count, 0.0, "ERR"))
                print(f"  Error at {target_count} envs: {error_msg[:200]}")
            gc.collect()
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 70)
    print("  SCALING TEST RESULTS")
    print("=" * 70)
    print(f"  {'Env Count':<12} {'Peak GPU (GB)':<16} {'Status':<10}")
    print(f"  {'-' * 12} {'-' * 16} {'-' * 10}")
    for count, peak_gb, status in results:
        print(f"  | {count:<10} | {peak_gb:<14.2f} | {status:<8} |")
    print(f"  {'-' * 12} {'-' * 16} {'-' * 10}")

    # Recommendation
    print(f"\n  Memory budget: {MEMORY_BUDGET_GB:.0f} GB (peak < {MEMORY_BUDGET_GB:.0f} GB, "
          f"leaving {64 - MEMORY_BUDGET_GB:.0f} GB for optimizer/gradients)")
    if max_safe_count > 0:
        print(f"  RECOMMENDATION: Maximum safe env count = {max_safe_count}")
        if max_safe_count >= 1024:
            print(f"  1024 camera environments fits within GPU memory budget.")
        else:
            print(f"  WARNING: 1024 envs exceeds budget. Consider reducing to {max_safe_count}.")
    else:
        print(f"  WARNING: No tested env count fits within the {MEMORY_BUDGET_GB:.0f} GB budget.")
        print(f"  Consider reducing camera resolution or using fewer environments.")

    print("=" * 70)


def run_normal_validation():
    """Run normal camera validation: create env, step, save images, print diagnostics.

    If --memory_only is set, additionally prints GPU memory after each phase.
    """
    # create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)

    if args_cli.memory_only:
        torch.cuda.reset_peak_memory_stats()
        _print_memory("Before env creation")

    # create environment
    env = gym.make(args_cli.task, cfg=None, render_mode=None)

    print(f"[INFO] Created environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO] Device: {env.unwrapped.device}")

    if args_cli.memory_only:
        _print_memory("After env creation")

    # reset environment
    env.reset()

    if args_cli.memory_only:
        _print_memory("After env.reset()")

    # step with zero actions to let the scene settle and render
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    print(f"[INFO] Running {args_cli.num_steps} steps with zero actions to settle scene...")
    for step in range(args_cli.num_steps):
        env.step(actions)

    if args_cli.memory_only:
        _print_memory(f"After {args_cli.num_steps} sim steps")

    # access the TiledCamera sensor from the scene
    camera = env.unwrapped.scene["tiled_camera"]

    # print camera configuration summary
    print("\n--- Camera Configuration ---")
    print(f"  Prim path: {camera.cfg.prim_path}")
    print(f"  Resolution: {camera.cfg.width}x{camera.cfg.height}")
    print(f"  Data types: {camera.cfg.data_types}")
    print(f"  Update period: {camera.cfg.update_period}")
    if hasattr(camera.cfg.spawn, "focal_length"):
        print(f"  Focal length: {camera.cfg.spawn.focal_length} mm")
        print(f"  Horizontal aperture: {camera.cfg.spawn.horizontal_aperture} mm")
        print(f"  Focus distance: {camera.cfg.spawn.focus_distance} m")
        print(f"  Clipping range: {camera.cfg.spawn.clipping_range}")
        hfov = 2 * math.atan(camera.cfg.spawn.horizontal_aperture / (2 * camera.cfg.spawn.focal_length))
        hfov_deg = math.degrees(hfov)
        print(f"  Approx horizontal FOV: {hfov_deg:.1f} degrees")

    # read camera data
    # TiledCamera output["rgb"] shape: [num_envs, H, W, 4] (RGBA uint8)
    rgb_data = camera.data.output["rgb"]
    print(f"\n--- Camera Data ---")
    print(f"  RGB tensor shape: {rgb_data.shape}")
    print(f"  RGB tensor dtype: {rgb_data.dtype}")

    if args_cli.memory_only:
        _print_memory("After reading camera data")

    # save images for each environment
    num_envs = rgb_data.shape[0]
    print(f"\n--- Saving {num_envs} images to {args_cli.output_dir} ---")

    intensities = []
    for i in range(num_envs):
        # slice to RGB only (drop alpha channel)
        rgb_image = rgb_data[i, :, :, :3].cpu().numpy().astype(np.uint8)
        image_path = os.path.join(args_cli.output_dir, f"env_{i}_rgb.png")
        Image.fromarray(rgb_image).save(image_path)

        # compute mean pixel intensity for brightness consistency check
        mean_intensity = rgb_image.astype(np.float32).mean()
        per_channel_mean = rgb_image.astype(np.float32).mean(axis=(0, 1))
        intensities.append(mean_intensity)
        print(f"  env_{i}: saved to {image_path} | mean intensity: {mean_intensity:.1f} | "
              f"per-channel (R,G,B): ({per_channel_mean[0]:.1f}, {per_channel_mean[1]:.1f}, {per_channel_mean[2]:.1f})")

    # Brightness consistency analysis
    if len(intensities) > 1:
        intensities_arr = np.array(intensities)
        mean_all = intensities_arr.mean()
        if mean_all > 0:
            variation_pct = (intensities_arr.max() - intensities_arr.min()) / mean_all * 100
        else:
            variation_pct = 0.0
        print(f"\n--- Brightness Consistency ---")
        print(f"  Mean intensity across envs: {mean_all:.1f}")
        print(f"  Min: {intensities_arr.min():.1f}, Max: {intensities_arr.max():.1f}")
        print(f"  Variation: {variation_pct:.1f}%")
        if variation_pct < 10.0:
            print(f"  Status: PASS (variation < 10%)")
        else:
            print(f"  Status: WARN (variation >= 10% -- document for Phase 3 normalization)")

    # print camera world pose for documentation
    if hasattr(camera.data, "pos_w") and hasattr(camera.data, "quat_w_ros"):
        print(f"\n--- Camera World Poses ---")
        for i in range(min(num_envs, 4)):
            pos = camera.data.pos_w[i].cpu().numpy()
            quat = camera.data.quat_w_ros[i].cpu().numpy()
            print(f"  env_{i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                  f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    if args_cli.memory_only:
        _print_memory("After saving images")
        print(f"\n--- Memory Summary ---")
        print(f"  Peak GPU memory (entire run): {_gpu_mem_gb():.2f} GB")

    print(f"\n[INFO] Validation complete. {num_envs} images saved to {args_cli.output_dir}")
    print("[INFO] Inspect images to verify: robot gripper, table, and cube visible from wrist perspective.")
    print("[INFO] Check brightness consistency: mean intensities should be similar across environments.")

    # close environment
    env.close()


def main():
    """Entry point: dispatch to scaling test or normal validation."""
    if args_cli.scaling_test:
        run_scaling_test()
    else:
        run_normal_validation()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
