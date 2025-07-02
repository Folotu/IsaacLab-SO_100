# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train an RL agent with skrl.

This script is an example of how to train a reinforcement learning (RL) agent with skrl in Isaac Lab.

It is designed to be used with the `rl_games_train.sh` script. This script will launch the training script
with the correct arguments, and will also handle the multi-node training case.

.. code-block:: bash

    # single-gpu
    ./rl_games_train.sh --task Isaac-Ant-v0

    # multi-gpu
    ./rl_games_train.sh --task Isaac-Ant-v0 --num_gpus 2

"""


import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--logdir", type=str, default=None, help="Root directory for logging.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import SO_100.tasks  # noqa: F401


def main():
    """Train with skrl agent."""
    # hydra configuration
    @hydra_task_config
    def hydra_main(cfg: dict):
        # print configuration
        print_dict(cfg)

        # multi-agent to single-agent environment
        if "multi_agent" in cfg and cfg["multi_agent"]:
            # check if the task is a multi-agent environment
            if not issubclass(cfg["task"]["class"], DirectMARLEnv):
                raise ValueError(
                    f"The task {cfg['task']['name']} is not a multi-agent environment. "
                    f"Please set the 'multi_agent' flag to False."
                )
            # update number of agents
            cfg["task"]["env"]["num_agents"] = cfg["task"]["num_agents"]
            # create a single-agent environment from a multi-agent environment
            env = multi_agent_to_single_agent(gym.make(cfg["task"]["name"], cfg=cfg["task"]["env"], **cfg["render"]))
        else:
            # create environment
            env = gym.make(cfg["task"]["name"], cfg=cfg["task"]["env"], **cfg["render"])

        # wrap the environment for skrl
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # log directory
        if args_cli.logdir is None:
            log_dir = os.path.join("logs", cfg["task"]["name"], args_cli.algorithm)
            if args_cli.distributed:
                log_dir = os.path.join(log_dir, f"distributed_seed_{args_cli.seed}")
            else:
                log_dir = os.path.join(log_dir, f"seed_{args_cli.seed}")
            # add timestamp to log directory
            log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            log_dir = args_cli.logdir

        # create a temporary directory for the rl-games configuration
        # this is needed because the rl-games runner only accepts a path to a configuration file
        # and hydra does not save the configuration file until the end of the training
        # so we need to save the configuration file ourselves
        os.makedirs(log_dir, exist_ok=True)
        # save the configuration file
        config_path = os.path.join(log_dir, "config.yaml")
        dump_yaml(config_path, cfg)
        # save the task configuration file
        task_cfg_path = os.path.join(log_dir, "task.pkl")
        dump_pickle(task_cfg_path, cfg["task"])

        # get the path to the skrl configuration file
        # if the path is not absolute, it is assumed to be relative to the task's config directory
        skrl_config_path = cfg["rl_games"].pop("config_path")
        if not os.path.isabs(skrl_config_path):
            skrl_config_path = os.path.join(os.path.dirname(cfg["task"]["config_path"]), skrl_config_path)

        # create runner
        runner = Runner(env, args=args_cli, agent_class=args_cli.algorithm, agent_config=skrl_config_path)
        # configure and launch the runner
        runner.configure(log_dir, checkpoint=args_cli.checkpoint)
        runner.launch()

    # run the training
    hydra_main()

    # close the environment
    env.close()


if __name__ == "__main__":
    # set random seed
    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 1000000)

    # run the main function
    main()

    # close the simulation
    simulation_app.close()
