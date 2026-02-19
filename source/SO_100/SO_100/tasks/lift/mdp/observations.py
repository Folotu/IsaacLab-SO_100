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
