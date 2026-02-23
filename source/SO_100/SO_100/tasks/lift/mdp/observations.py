# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def penultimate_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """Extract 512-dim penultimate ResNet18 features (avgpool) from camera images.

    Uses torchvision's create_feature_extractor to tap the 'flatten' node of
    ResNet18, which outputs the 512-dim vector after adaptive average pooling
    but before the final FC classification layer. These features retain spatial
    layout information that is lost in the 1000-dim ImageNet logits.

    The model is frozen (no gradients) and cached on the env object for reuse.
    ImageNet normalization and 224x224 resize are applied to raw camera images.

    Note: This is a plain observation function (not ManagerTermBase) to avoid
    deferred initialization issues. Follows same pattern as
    object_position_in_robot_root_frame.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the camera sensor. Defaults to "tiled_camera".
        data_type: Camera data type to read. Defaults to "rgb".

    Returns:
        Tensor of shape (num_envs, 512) with penultimate ResNet18 features.
    """
    # Lazy initialization: build and cache the feature extractor on first call
    if not hasattr(env, "_penultimate_encoder"):
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.models.feature_extraction import create_feature_extractor

        # Load pretrained ResNet18 and extract up to the flatten node (512-dim)
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = create_feature_extractor(
            base_model, return_nodes={"flatten": "features"}
        )
        # Freeze all parameters and set eval mode
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        feature_extractor.to(env.device)
        env._penultimate_encoder = feature_extractor

        # Cache ImageNet normalization constants on the correct device
        env._penultimate_img_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=env.device
        ).view(1, 3, 1, 1)
        env._penultimate_img_std = torch.tensor(
            [0.229, 0.224, 0.225], device=env.device
        ).view(1, 3, 1, 1)

    # Read camera data
    camera = env.scene[sensor_cfg.name]
    raw_rgba = camera.data.output.get(data_type)

    if raw_rgba is None:
        # Camera not yet rendered -- return zeros
        return torch.zeros(env.num_envs, 512, device=env.device)

    # raw_rgba shape: (num_envs, H, W, 4) uint8 RGBA
    # Extract RGB and convert to float NCHW [0, 1]
    rgb = raw_rgba[:, :, :, :3]
    rgb_nchw = rgb.permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)

    # Resize to 224x224 if needed (ResNet18 expects 224x224 input)
    if rgb_nchw.shape[2] != 224 or rgb_nchw.shape[3] != 224:
        rgb_nchw = torch.nn.functional.interpolate(
            rgb_nchw, size=(224, 224), mode="bilinear", align_corners=False
        )

    # Apply ImageNet normalization
    rgb_nchw = (rgb_nchw - env._penultimate_img_mean) / env._penultimate_img_std

    # Extract features (no gradient computation)
    with torch.no_grad():
        output = env._penultimate_encoder(rgb_nchw)

    return output["features"]  # shape: (num_envs, 512)


def spatial_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Extract 512-dim spatial keypoints from ResNet18 layer3 via spatial softmax.

    Instead of global average pooling (which destroys spatial information), this
    extracts the layer3 feature maps (256 channels, 8x8 at 128x128 input) and
    applies spatial softmax to compute expected (x, y) coordinates per channel.
    This produces 256 * 2 = 512 values encoding WHERE each visual feature
    activates — critical for distinguishing "gripper above cube" from "gripper
    around cube" at sub-centimeter precision.

    The ResNet18 backbone is frozen (no gradients). Only the spatial softmax
    computation runs each step.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the camera sensor.
        data_type: Camera data type to read.
        temperature: Softmax temperature. Lower = sharper keypoints.

    Returns:
        Tensor of shape (num_envs, 512) with spatial keypoint coordinates.
    """
    # Lazy initialization
    if not hasattr(env, "_spatial_encoder"):
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.models.feature_extraction import create_feature_extractor

        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Extract layer3: 256 channels, preserves 8x8 spatial grid at 128x128 input
        feature_extractor = create_feature_extractor(
            base_model, return_nodes={"layer3": "features"}
        )
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        feature_extractor.to(env.device)
        env._spatial_encoder = feature_extractor

        # Cache ImageNet normalization constants
        env._spatial_img_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=env.device
        ).view(1, 3, 1, 1)
        env._spatial_img_std = torch.tensor(
            [0.229, 0.224, 0.225], device=env.device
        ).view(1, 3, 1, 1)

        # Coordinate grids will be created on first forward pass (need H, W from feature map)
        env._spatial_coords_cached = False

    # Read camera data
    camera = env.scene[sensor_cfg.name]
    raw_rgba = camera.data.output.get(data_type)

    if raw_rgba is None:
        return torch.zeros(env.num_envs, 512, device=env.device)

    # RGBA → RGB float NCHW [0, 1]
    rgb = raw_rgba[:, :, :, :3]
    rgb_nchw = rgb.permute(0, 3, 1, 2).float() / 255.0

    # Resize to 128x128 for spatial resolution balance (8x8 at layer3)
    if rgb_nchw.shape[2] != 128 or rgb_nchw.shape[3] != 128:
        rgb_nchw = torch.nn.functional.interpolate(
            rgb_nchw, size=(128, 128), mode="bilinear", align_corners=False
        )

    # ImageNet normalization
    rgb_nchw = (rgb_nchw - env._spatial_img_mean) / env._spatial_img_std

    # Extract layer3 feature maps
    with torch.no_grad():
        output = env._spatial_encoder(rgb_nchw)
    feat = output["features"]  # (N, 256, H, W)

    # Build coordinate grids on first call (cached for performance)
    if not env._spatial_coords_cached:
        _, _, H, W = feat.shape
        pos_x = torch.linspace(-1.0, 1.0, W, device=env.device).view(1, 1, 1, W)
        pos_y = torch.linspace(-1.0, 1.0, H, device=env.device).view(1, 1, H, 1)
        env._spatial_pos_x = pos_x.expand(1, 1, H, W)
        env._spatial_pos_y = pos_y.expand(1, 1, H, W)
        env._spatial_coords_cached = True

    # Spatial softmax: treat each channel as a probability distribution over (x, y)
    N, C, H, W = feat.shape
    attention = torch.softmax(feat.view(N, C, -1) / temperature, dim=-1).view(N, C, H, W)

    # Expected coordinates per channel
    expected_x = (attention * env._spatial_pos_x).sum(dim=(2, 3))  # (N, 256)
    expected_y = (attention * env._spatial_pos_y).sum(dim=(2, 3))  # (N, 256)

    return torch.cat([expected_x, expected_y], dim=-1)  # (N, 512)


def stacked_spatial_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    temperature: float = 1.0,
    num_frames: int = 3,
) -> torch.Tensor:
    """3-frame stacked spatial softmax features for temporal awareness.

    A single spatial softmax frame encodes WHERE visual features are, but cannot
    encode motion — "object moving up" and "object hovering" look identical in one
    frame. Stacking 3 consecutive frames lets the MLP compute velocity (frame
    differences) and acceleration, making lifting observable from the actor's
    visual input.

    Returns (num_envs, 512 * num_frames) = (num_envs, 1536) for 3 frames.
    Zero-pads history for the first few steps of each episode.
    """
    # Compute current frame's spatial features
    current = spatial_image_features(env, sensor_cfg, data_type, temperature)  # (N, 512)
    feat_dim = current.shape[-1]

    # Initialize frame buffer on first call
    if not hasattr(env, '_frame_stack'):
        env._frame_stack = torch.zeros(
            env.num_envs, num_frames, feat_dim, device=env.device
        )

    # Zero out history for freshly reset envs (episode_length_buf == 0 after reset)
    reset_mask = env.episode_length_buf < num_frames
    env._frame_stack[reset_mask] = 0.0

    # Shift buffer: drop oldest frame, append current
    env._frame_stack = torch.roll(env._frame_stack, shifts=-1, dims=1)
    env._frame_stack[:, -1] = current

    # Return flattened: (N, 512 * num_frames)
    return env._frame_stack.reshape(env.num_envs, -1)


def object_linear_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity in world frame (privileged, for critic).

    Gives the critic the ability to distinguish "object moving up" from "object
    hovering at height" — critical for accurate value estimation during lifting.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w[:, :3]
