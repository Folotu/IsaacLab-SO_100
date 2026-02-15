# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual domain randomization event functions for camera-based training.

These are plain event functions (not ManagerTermBase classes) that work reliably
with mode="reset" in ManagerBasedRLEnv. The ManagerTermBase versions from
isaaclab.envs.mdp have a deferred initialization mechanism via timeline callbacks
that can fail to fire before the first env.reset() in the RL env workflow.

This approach mirrors isaaclab_tasks/.../stack/mdp/franka_stack_events.py which
uses plain functions for the same reason.
"""

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_object_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    colors: dict[str, tuple[float, float]],
    event_name: str = "object_color_randomizer",
):
    """Randomize the visual color of an asset using Replicator API.

    Simplified plain-function version of isaaclab.envs.mdp.randomize_visual_color.
    Samples random RGB colors from the specified ranges and applies them via Replicator.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        asset_cfg: The scene entity configuration for the target asset.
        colors: Dictionary with keys "r", "g", "b" mapping to (low, high) tuples.
        event_name: Name for the Replicator event trigger.
    """
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    asset = env.scene[asset_cfg.name]
    mesh_prim_path = f"{asset.cfg.prim_path}/.*"

    # Sample random color
    r = random.uniform(colors["r"][0], colors["r"][1])
    g = random.uniform(colors["g"][0], colors["g"][1])
    b = random.uniform(colors["b"][0], colors["b"][1])

    prims_group = rep.get.prims(path_pattern=mesh_prim_path)
    with prims_group:
        rep.randomizer.color(colors=rep.distribution.uniform((r, g, b), (r, g, b)))


def randomize_table_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    textures: list[str],
    texture_rotation: tuple[float, float] = (0.0, 0.0),
    event_name: str = "table_texture_randomizer",
):
    """Randomize the visual texture of an asset using Replicator API.

    Simplified plain-function version of isaaclab.envs.mdp.randomize_visual_texture_material.
    Samples a random texture from the provided list and applies it via Replicator.

    This follows the same pattern as franka_stack_events.randomize_visual_texture_material.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        asset_cfg: The scene entity configuration for the target asset.
        textures: List of texture file paths (NVIDIA Nucleus paths).
        texture_rotation: Tuple of (min, max) rotation in radians.
        event_name: Name for the Replicator event trigger.
    """
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize visual texture with scene replication enabled."
            " Set replicate_physics=False in InteractiveSceneCfg."
        )

    # Convert rotation from radians to degrees
    rotation_deg = tuple(math.degrees(angle) for angle in texture_rotation)

    asset = env.scene[asset_cfg.name]

    # Build prim path pattern -- try /visuals first, fall back to broad match
    body_names_regex = ".*"
    if hasattr(asset, "cfg"):
        prim_path = f"{asset.cfg.prim_path}/{body_names_regex}/visuals"
    else:
        prim_path = f"{asset.prim_paths[0]}/visuals"

    prims_group = rep.get.prims(path_pattern=prim_path)
    with prims_group:
        rep.randomizer.texture(
            textures=textures,
            project_uvw=True,
            texture_rotate=rep.distribution.uniform(*rotation_deg),
        )
