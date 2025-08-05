# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher  # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--num_episodes", type=int, default=10, help="Number of episodes to play. Set to -1 to run until the number of steps specified in the config."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import SO_100.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play a trained policy from a checkpoint."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # get checkpoint path
    if args_cli.checkpoint:
        # check if the checkpoint is a file
        if os.path.isfile(args_cli.checkpoint) and (args_cli.checkpoint.endswith(".pt") or args_cli.checkpoint.endswith(".pth")):
            checkpoint_path = args_cli.checkpoint
        else:
            # get the checkpoint path for the trained model
            checkpoint_path = get_checkpoint_path(args_cli.checkpoint, "skrl", args_cli.algorithm)
    elif args_cli.use_pretrained_checkpoint:
        # get the checkpoint path for the pretrained model
        checkpoint_path = get_published_pretrained_checkpoint(args_cli.task, "skrl", args_cli.algorithm)
    else:
        raise ValueError("Either checkpoint or use_pretrained_checkpoint must be provided.")

    # load agent configuration
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    agent_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(checkpoint_path), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # instantiate the agent
    runner = Runner(env, agent_cfg)
    runner.agent.load(checkpoint_path)

    # set agent to evaluation mode
    for model in runner.agent.models.values():
        model.eval()

    # configure the runner
    if args_cli.num_episodes > 0:
        runner._cfg["trainer"]["timesteps"] = -1
        runner._cfg["trainer"]["episodes"] = args_cli.num_episodes

    # play the agent
    # manually run the environment for a fixed number of episodes
    timestep = 0
    for episode in range(args_cli.num_episodes):
        print(f"[INFO] Playing episode: {episode + 1} / {args_cli.num_episodes}")
        # reset the environment
        obs, info = env.reset()
        # run the episode
        while True:
            # compute agent's actions
            with torch.no_grad():
                # get the actions from the agent
                # during playback, we are interested in the deterministic actions (mean actions)
                # but the agent still returns the stochastic actions (sampled from the distribution)
                _, _, outputs = runner.agent.act(obs, timestep=timestep, timesteps=-1)
                actions = outputs["mean_actions"]
            timestep += 1
            # step the environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            # check if the episode is over
            if terminated.any() or truncated.any():
                break

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()