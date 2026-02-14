"""Camera angle sweep: renders the scene from many orientations so you can pick the best one.

Usage:
    python camera_angle_sweep.py

Generates a grid of images at different pitch/yaw angles and saves them to /tmp/cam_sweep/.
Each image filename encodes the pitch and yaw angles used.
Pick the image that matches your real wrist camera, then we'll use those angles.
"""

import argparse
import math
import os
import sys

# Isaac Lab requires AppLauncher before any other imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Camera angle sweep")
parser.add_argument("--output_dir", type=str, default="/tmp/cam_sweep")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now safe to import Isaac Lab
import torch
import numpy as np
from PIL import Image
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401

from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils


def euler_to_quat_opengl(pitch_deg, yaw_deg):
    """Convert pitch (up/down) and yaw (left/right) to quaternion in opengl convention.

    pitch_deg: positive = look down, negative = look up
    yaw_deg: positive = look right, negative = look left

    In opengl, camera looks along -Z, Y is up.
    """
    # Convert to radians
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # Build rotation: first yaw around Y, then pitch around X
    # Quaternion for rotation around Y (yaw)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    qy = (cy, 0, sy, 0)  # (w, x, y, z)

    # Quaternion for rotation around X (pitch)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    qp = (cp, sp, 0, 0)  # (w, x, y, z)

    # Combined: qy * qp (yaw first, then pitch)
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qp
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Define sweep angles
    # Pitch: -30 (look up) to +90 (look straight down)
    # Yaw: -45 (look left) to +45 (look right)
    pitch_angles = [-15, 0, 15, 30, 45, 60, 75, 90]
    yaw_angles = [-30, 0, 30]

    print(f"Sweeping {len(pitch_angles)} pitch x {len(yaw_angles)} yaw = {len(pitch_angles)*len(yaw_angles)} angles")
    print(f"Output: {args_cli.output_dir}")

    # Load the camera env config
    env_cfg_entry = gym.spec("SO-ARM100-Lift-Cube-Camera-v0").kwargs["env_cfg_entry_point"]
    import importlib
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    env_cfg_cls = getattr(module, class_name)

    for pitch in pitch_angles:
        for yaw in yaw_angles:
            quat = euler_to_quat_opengl(pitch, yaw)

            # Create fresh config for each angle
            env_cfg = env_cfg_cls()
            env_cfg.scene.num_envs = 1

            # Override camera rotation
            env_cfg.scene.tiled_camera.offset.rot = quat
            env_cfg.scene.tiled_camera.offset.convention = "opengl"

            print(f"\n--- Pitch={pitch:+3d}, Yaw={yaw:+3d} | quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}) ---")

            try:
                env = gym.make("SO-ARM100-Lift-Cube-Camera-v0", cfg=env_cfg, render_mode=None)
                env.reset()

                # Step a few times to settle
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                for _ in range(5):
                    env.step(actions)

                # Get camera data
                camera = env.unwrapped.scene["tiled_camera"]
                rgb = camera.data.output["rgb"]  # [1, H, W, 3 or 4]
                img_np = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)

                filename = f"pitch{pitch:+03d}_yaw{yaw:+03d}.png"
                filepath = os.path.join(args_cli.output_dir, filename)
                Image.fromarray(img_np).save(filepath)
                print(f"  Saved: {filename} | mean={img_np.mean():.1f}")

                env.close()
            except Exception as e:
                print(f"  ERROR: {e}")
                try:
                    env.close()
                except:
                    pass

    print(f"\n\nDone! {len(pitch_angles)*len(yaw_angles)} images saved to {args_cli.output_dir}")
    print("Browse the images and pick the pitch/yaw that matches your real camera.")
    print("Then tell me the pitch and yaw values from the filename.")

    simulation_app.close()


if __name__ == "__main__":
    main()
