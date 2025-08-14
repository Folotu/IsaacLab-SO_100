# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--logdir", type=str, default=None, help="Path to the directory where checkpoints will be saved.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import os
import sys
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SO_100.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    # determine rank based on distributed training setup
    if args_cli.distributed:
        # Use AWS Batch environment variables directly - more reliable than our exports
        aws_node_index = os.environ.get('AWS_BATCH_JOB_NODE_INDEX')
        if aws_node_index is not None:
            node_rank = int(aws_node_index)
        else:
            # Fallback to our exported NODE_RANK or default to 0
            node_rank = int(os.environ.get('NODE_RANK', 0))
        
        local_rank = app_launcher.local_rank
        # global rank = node_rank * processes_per_node + local_rank
        # for single GPU per node: global_rank = node_rank
        rank = node_rank
        device = f"cuda:{local_rank}"
        print(f"[DEBUG] AWS Batch variables: AWS_BATCH_JOB_NODE_INDEX='{os.environ.get('AWS_BATCH_JOB_NODE_INDEX', 'NOT_SET')}', NODE_RANK='{os.environ.get('NODE_RANK', 'NOT_SET')}', CHECKPOINT_LOGDIR='{os.environ.get('CHECKPOINT_LOGDIR', 'NOT_SET')}'")
        print(f"[DEBUG] Distributed mode - Node rank: {node_rank}, Local rank: {local_rank}, Global rank: {rank}")
    else:
        rank = 0
        device = "cpu"
        print(f"[DEBUG] Single process: args_cli.logdir = '{args_cli.logdir}'")
    
    # In distributed mode, ensure all nodes use consistent paths
    if args_cli.distributed:
        # Get base log directory from environment variable or use default
        checkpoint_logdir = os.environ.get('CHECKPOINT_LOGDIR')
        if checkpoint_logdir:
            log_root_path = os.path.abspath(checkpoint_logdir)
            print(f"[INFO] Rank {rank}: Using distributed base log directory: {log_root_path}")
        else:
            log_root_path = os.path.join("/mnt/efs/checkpoints", "rsl_rl")
            print(f"[WARNING] Rank {rank}: CHECKPOINT_LOGDIR not set. Using default: {log_root_path}")
        
        # Check if this is a resume job by looking for resume arguments
        is_resume_job = agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
        
        if is_resume_job:
            # RESUME MODE: Use the full timestamped checkpoint directory for consistent logging
            # CHECKPOINT_LOGDIR is used for Isaac Lab checkpoint loading (experiment dir)
            # CHECKPOINT_FULL_PATH is used for actual logging/videos (full timestamped path)
            checkpoint_full_path = os.environ.get('CHECKPOINT_FULL_PATH')
            if checkpoint_full_path:
                log_dir = checkpoint_full_path
                print(f"[INFO] Rank {rank}: Resume mode - using full timestamped log directory: {log_dir}")
            else:
                # Fallback to original behavior if CHECKPOINT_FULL_PATH not set
                log_dir = log_root_path
                print(f"[WARNING] Rank {rank}: CHECKPOINT_FULL_PATH not set, using base directory: {log_dir}")
        else:
            # NEW TRAINING MODE: Create new timestamped directory
            # Use shared file approach to synchronize run_name across nodes
            run_name_file = os.path.join(log_root_path, ".run_name_sync")
            
            if rank == 0:
                # Rank 0 creates the timestamp and writes to shared file
                run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if agent_cfg.run_name:
                    run_name += f"_{agent_cfg.run_name}"
                print(f"[INFO] Rank 0: Generated run_name: {run_name}")
                
                # Ensure directory exists and write run_name to shared file
                os.makedirs(log_root_path, exist_ok=True)
                with open(run_name_file, 'w') as f:
                    f.write(run_name)
            else:
                # Worker nodes wait for and read run_name from shared file
                print(f"[INFO] Rank {rank}: Waiting for run_name from rank 0...")
                import time
                max_wait = 60  # seconds
                wait_time = 0
                while not os.path.exists(run_name_file) and wait_time < max_wait:
                    time.sleep(1)
                    wait_time += 1
                
                if os.path.exists(run_name_file):
                    with open(run_name_file, 'r') as f:
                        run_name = f.read().strip()
                    print(f"[INFO] Rank {rank}: Using run_name from rank 0: {run_name}")
                else:
                    # Fallback if file doesn't exist
                    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    print(f"[WARNING] Rank {rank}: Timeout waiting for run_name, using fallback: {run_name}")
            
            # Create the full log directory path for new training
            if agent_cfg.experiment_name:
                log_dir = os.path.join(log_root_path, agent_cfg.experiment_name, run_name)
            else:
                log_dir = os.path.join(log_root_path, run_name)
    
    else:
        # Single process mode: use original logic
        if args_cli.logdir:
            log_root_path = os.path.abspath(args_cli.logdir)
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if agent_cfg.run_name:
                run_name += f"_{agent_cfg.run_name}"
            
            if agent_cfg.experiment_name:
                log_dir = os.path.join(log_root_path, agent_cfg.experiment_name, run_name)
            else:
                log_dir = os.path.join(log_root_path, run_name)
        else:
            # Use default directory if no logdir is provided
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # Save resume path before creating runner
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # For distributed mode, we need log_root_path to be defined
        if not args_cli.distributed:
            if args_cli.logdir:
                log_root_path = os.path.abspath(args_cli.logdir)
            else:
                log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # clear CUDA cache to prevent memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
