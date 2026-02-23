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
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper when near the object.

    Combines proximity (tanh kernel) with gripper closure. Encourages the agent
    to close the gripper specifically when it is close to the object, not in free space.

    The Gripper joint is the last joint (index -1) on the SO-100. Joint position
    is 0.0 when fully closed and 0.5 when fully open (BinaryJointPositionActionCfg).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Tanh proximity kernel (bounded [0, 1])
    distance = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    proximity = 1.0 - torch.tanh(distance / std)

    # Gripper closure: joint_pos[-1] is the Gripper joint
    # 0.0 = fully closed, 0.5 = fully open -> closure = 1 - 2*joint_pos
    gripper_pos = env.scene["robot"].data.joint_pos[:, -1]
    closure = torch.clamp(1.0 - 2.0 * gripper_pos, 0.0, 1.0)

    return proximity * closure


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


def gripper_object_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """V2 DEPRECATED: Binary contact reward. Saturates with single-body touch.
    Kept for reference. Use gripper_grasp_quality (V3) instead."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    max_force = force_magnitude.max(dim=-1).values
    return (max_force > threshold).float()


def gripper_grasp_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    min_force: float = 0.5,
    max_force: float = 20.0,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Continuous grasp quality with gradient bridge for single-to-dual jaw contact.

    Uses a weighted blend: alpha * max(f1,f2) + (1-alpha) * sqrt(f1*f2).
    The max term acts as a "magnet" (non-zero reward for single-jaw touch, keeping
    the hand on the object). The geometric mean term acts as the "lock" (higher
    reward when both jaws engage). This avoids the pure geometric mean's
    zero-gradient canyon where one jaw missing kills the gradient for the other.

    Contact sensor bodies: index 0 = Fixed_Gripper, index 1 = Moving_Jaw.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    # net_forces_w shape: (num_envs, num_bodies, 3)
    force_per_body = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    # force_per_body shape: (num_envs, 2)

    # Continuous force scaling per body: 0 below min_force, 1 at max_force
    scaled = torch.clamp((force_per_body - min_force) / (max_force - min_force), 0.0, 1.0)

    # Weighted blend: magnet (single-jaw) + lock (dual-jaw)
    single_jaw = scaled.max(dim=-1).values
    dual_jaw = torch.sqrt(scaled[:, 0] * scaled[:, 1] + 1e-8)
    return alpha * single_jaw + (1.0 - alpha) * dual_jaw


def is_grasped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_threshold: float = 1.0,
    min_height: float = 0.03,
) -> torch.Tensor:
    """V2 DEPRECATED: Binary contact gate * height. Kept for reference.
    Use is_grasped_v3 instead."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    has_contact = (force_magnitude.max(dim=-1).values > contact_threshold).float()
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    height_above_table = torch.clamp(height - min_height, min=0.0)
    return has_contact * height_above_table * 10.0


def is_grasped_v3(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_force: float = 0.5,
    max_force: float = 20.0,
    min_height: float = 0.03,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Grasp + lift reward using blended grasp quality * height.

    Uses the same gradient bridge as gripper_grasp_quality: single-jaw contact
    provides partial signal (via alpha * max), dual-jaw provides full signal
    (via geometric mean). Multiplied by height above table for continuous
    gradient from poor grasp through strong grasp + lift.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    force_per_body = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    scaled = torch.clamp((force_per_body - min_force) / (max_force - min_force), 0.0, 1.0)

    single_jaw = scaled.max(dim=-1).values
    dual_jaw = torch.sqrt(scaled[:, 0] * scaled[:, 1] + 1e-8)
    grasp_quality = alpha * single_jaw + (1.0 - alpha) * dual_jaw

    obj: RigidObject = env.scene[object_cfg.name]
    height_above_table = torch.clamp(obj.data.root_pos_w[:, 2] - min_height, min=0.0)

    return grasp_quality * height_above_table * 10.0


def object_upward_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward any upward object velocity. Provides immediate gradient for lifting
    before height thresholds are crossed — the missing signal between contact and lift."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.clamp(obj.data.root_lin_vel_w[:, 2], 0.0, 1.0)


def lifting_success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-time bonus when the object first crosses the lift threshold.

    Unlike per-step lifting rewards that fire every step above threshold,
    this fires ONCE per crossing event. GAE propagates this massive spike
    back through the entire trajectory, creating high advantage at the
    grasping action that preceded the lift.

    Resets when the object drops back below threshold (so each distinct
    lift attempt gets its own bonus). Also naturally resets on env reset
    since the object returns to its initial height.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    above = obj.data.root_pos_w[:, 2] > threshold

    if not hasattr(env, '_lift_bonus_given'):
        env._lift_bonus_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Bonus fires on the step where above transitions from False to True
    first_cross = above & (~env._lift_bonus_given)
    env._lift_bonus_given = env._lift_bonus_given | above

    # Reset when object drops back below threshold (catches env resets too)
    env._lift_bonus_given = env._lift_bonus_given & above

    return first_cross.float()
