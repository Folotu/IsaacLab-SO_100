"""Launch camera environment with GUI for interactive viewing.

Usage:
    python camera_gui_viewer.py

This launches the camera-enabled environment in GUI mode (not headless).
The simulation will run and pause so you can:
1. In the Stage panel (left), find: World > envs > env_0 > Robot > Fixed_Gripper > wrist_cam
2. In the Viewport, click the camera icon (top-left) or go to:
   Viewport menu > Cameras > /World/envs/env_0/Robot/Fixed_Gripper/wrist_cam
3. This switches the viewport to show exactly what the wrist camera sees
4. Press SPACE or click Play to step the sim and see the camera move with the arm

Press Ctrl+C in the terminal to quit.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Camera GUI Viewer")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# NOT headless -- we want the GUI
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401

def main():
    # Load config
    env_cfg_entry = gym.spec("SO-ARM100-Lift-Cube-Camera-v0").kwargs["env_cfg_entry_point"]
    import importlib
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    env_cfg_cls = getattr(module, class_name)

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("SO-ARM100-Lift-Cube-Camera-v0", cfg=env_cfg, render_mode=None)
    env.reset()

    print("\n" + "="*60)
    print("  CAMERA GUI VIEWER")
    print("="*60)
    print()
    print("The environment is running with the GUI.")
    print()
    print("To view through the wrist camera:")
    print("  1. Look at the Viewport (main area)")
    print("  2. Click the camera dropdown (top-left of viewport)")
    print("  3. Select: /World/envs/env_0/Robot/Fixed_Gripper/wrist_cam")
    print()
    print("To switch back to free camera: select 'Perspective'")
    print()
    print("The sim is stepping with RANDOM actions so you can see the arm move.")
    print("The camera should follow the gripper in real-time.")
    print("Press Ctrl+C in terminal to quit.")
    print("="*60 + "\n")

    try:
        step = 0
        while simulation_app.is_running():
            actions = torch.randn(env.action_space.shape, device=env.unwrapped.device) * 0.3
            env.step(actions)
            step += 1
            if step % 500 == 0:
                print(f"  [step {step}] simulation running...")
    except KeyboardInterrupt:
        print("\nShutting down...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
