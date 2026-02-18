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


def augmented_image_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    model_name: str = "resnet18",
    augment: bool = True,
) -> torch.Tensor:
    """Extract visual features from camera with GPU-based image augmentation.

    Applies Kornia color jitter augmentation to the camera tensor BEFORE encoding,
    providing visual domain randomization without slow USD/Replicator stage modifications.
    This is ~1000x faster than Replicator-based randomization at 256+ envs.

    The augmentation (brightness, contrast, saturation, hue jitter) forces the frozen
    ResNet18 encoder to produce appearance-invariant features for sim-to-real transfer.
    """
    from isaaclab.envs.mdp.observations import image_features as _base_image_features

    if not augment or not getattr(env, "_kornia_augment", None):
        # First call or augmentation disabled: initialize Kornia augmentation pipeline
        if augment:
            try:
                import kornia.augmentation as K
                env._kornia_augment = K.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0
                )
            except ImportError:
                print("[WARNING] kornia not installed -- skipping image augmentation. pip install kornia")
                env._kornia_augment = None
                augment = False

    if augment and env._kornia_augment is not None and env.scene["tiled_camera"].data.output.get("rgb") is not None:
        # Get raw camera tensor: (num_envs, H, W, 4) uint8 RGBA
        raw_rgb = env.scene["tiled_camera"].data.output["rgb"][:, :, :, :3]  # drop alpha
        # Convert to float [0,1] and NCHW for Kornia
        images_float = raw_rgb.float() / 255.0  # (N, H, W, 3)
        images_nchw = images_float.permute(0, 3, 1, 2)  # (N, 3, H, W)
        # Apply augmentation on GPU (vectorized across all envs)
        with torch.no_grad():
            augmented = env._kornia_augment(images_nchw)
        # Write back to camera buffer as uint8 NHWC for image_features to consume
        aug_uint8 = (augmented.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        # Reconstruct RGBA by appending alpha channel
        alpha = torch.full((*aug_uint8.shape[:3], 1), 255, dtype=torch.uint8, device=aug_uint8.device)
        env.scene["tiled_camera"].data.output["rgb"] = torch.cat([aug_uint8, alpha], dim=-1)

    # Call the original image_features which runs the frozen encoder
    return _base_image_features(env, sensor_cfg=sensor_cfg, data_type=data_type, model_name=model_name)


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
