# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


def object_height_progress(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for any upward movement of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    # Reward upward movement from ground level (0.025m baseline)
    height_progress = torch.clamp(object.data.root_pos_w[:, 2] - 0.025, 0.0, 0.1)
    return height_progress * 10.0


def gripper_close_to_object(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward closing the gripper when near the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Distance between object and end-effector
    distance = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    proximity_reward = 1.0 - distance
    
    # Gripper closure reward (assuming last 2 joints are gripper)
    gripper_closure = 1.0 - env.scene["robot"].data.joint_pos[:, -2:].mean(dim=1)
    
    # Combine proximity and closure
    return (proximity_reward * gripper_closure).clamp(0.0, 1.0)


def gripper_orientation(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward proper gripper orientation by computing orientation alignment directly."""
    # Get end-effector and object transforms
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get end-effector orientation (quaternion)
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Compute a simple orientation reward based on gripper pointing down
    # For a proper grasp, we want the gripper to be oriented downward (negative z)
    # Extract z-component of the rotation matrix from quaternion
    # For quaternion [w, x, y, z], the z-axis direction is [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
    w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
    z_axis_z = 1 - 2 * (x*x + y*y)  # z-component of gripper's z-axis
    
    # Reward when gripper points downward (z_axis_z is negative)
    orientation_reward = torch.clamp(-z_axis_z, 0.0, 1.0)
    
    return orientation_reward
