# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone script to validate TiledCamera rendering in the SO-100 lift environment.

Creates a camera-enabled environment, steps it to let the scene settle, then saves
rendered RGB images to disk for manual inspection. Also prints camera intrinsics
and per-environment mean pixel intensity for brightness consistency checking.

Usage:
    python validate_camera.py --task SO-ARM100-Lift-Cube-Camera-v0 --num_envs 4 --output_dir /tmp/cam_test
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

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401


def main():
    """Validate camera rendering by saving images to disk."""
    # create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # create environment
    env = gym.make(args_cli.task, cfg=None, render_mode=None)

    # override num_envs if specified on command line
    # Note: num_envs is set in env config; CLI override happens via env_cfg before make()
    # For validation, the default of 4 in the script is appropriate

    print(f"[INFO] Created environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO] Device: {env.unwrapped.device}")

    # reset environment
    env.reset()

    # step with zero actions to let the scene settle and render
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    print(f"[INFO] Running {args_cli.num_steps} steps with zero actions to settle scene...")
    for step in range(args_cli.num_steps):
        env.step(actions)

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
        # Calculate approximate horizontal FOV
        import math
        hfov = 2 * math.atan(camera.cfg.spawn.horizontal_aperture / (2 * camera.cfg.spawn.focal_length))
        hfov_deg = math.degrees(hfov)
        print(f"  Approx horizontal FOV: {hfov_deg:.1f} degrees")

    # read camera data
    # TiledCamera output["rgb"] shape: [num_envs, H, W, 4] (RGBA uint8)
    rgb_data = camera.data.output["rgb"]
    print(f"\n--- Camera Data ---")
    print(f"  RGB tensor shape: {rgb_data.shape}")
    print(f"  RGB tensor dtype: {rgb_data.dtype}")

    # save images for each environment
    num_envs = rgb_data.shape[0]
    print(f"\n--- Saving {num_envs} images to {args_cli.output_dir} ---")

    for i in range(num_envs):
        # slice to RGB only (drop alpha channel)
        rgb_image = rgb_data[i, :, :, :3].cpu().numpy().astype(np.uint8)
        image_path = os.path.join(args_cli.output_dir, f"env_{i}_rgb.png")
        Image.fromarray(rgb_image).save(image_path)

        # compute mean pixel intensity for brightness consistency check
        mean_intensity = rgb_image.astype(np.float32).mean()
        per_channel_mean = rgb_image.astype(np.float32).mean(axis=(0, 1))
        print(f"  env_{i}: saved to {image_path} | mean intensity: {mean_intensity:.1f} | "
              f"per-channel (R,G,B): ({per_channel_mean[0]:.1f}, {per_channel_mean[1]:.1f}, {per_channel_mean[2]:.1f})")

    # print camera world pose for documentation
    if hasattr(camera.data, "pos_w") and hasattr(camera.data, "quat_w_ros"):
        print(f"\n--- Camera World Poses ---")
        for i in range(min(num_envs, 4)):
            pos = camera.data.pos_w[i].cpu().numpy()
            quat = camera.data.quat_w_ros[i].cpu().numpy()
            print(f"  env_{i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                  f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    print(f"\n[INFO] Validation complete. {num_envs} images saved to {args_cli.output_dir}")
    print("[INFO] Inspect images to verify: robot gripper, table, and cube visible from wrist perspective.")
    print("[INFO] Check brightness consistency: mean intensities should be similar across environments.")

    # close environment
    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
