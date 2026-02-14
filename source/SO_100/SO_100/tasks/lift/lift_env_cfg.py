# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import os
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


from . import mdp as local_mdp
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from SO_100.robots import SO_ARM100_CFG

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # optional tiled camera sensor (set by camera-enabled env variants)
    tiled_camera: TiledCameraCfg | None = None

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Dense reaching reward - matches working reference configuration
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Main lifting reward - matches working reference configuration  
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # Action penalties for smooth control
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    # Note: Keep dense reaching reward throughout early learning; remove reaching decay to avoid premature signal loss

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Dynamically determine per-rank number of environments to keep total global envs constant
        # Priority of env vars for total desired envs: TOTAL_ENVS > NUM_ENVS > default 4096
        try:
            total_envs = int(os.getenv("TOTAL_ENVS") or os.getenv("NUM_ENVS") or 4096)
        except Exception:
            total_envs = 4096
        try:
            world_size = int(os.getenv("WORLD_SIZE") or 1)
        except Exception:
            world_size = 1
        per_rank_envs = max(1, (total_envs + world_size - 1) // world_size)
        # Override the scene's num_envs set at construction time
        self.scene.num_envs = per_rank_envs
        # Optional sanity logging
        if os.getenv("LIFT_ENV_DEBUG", "0") == "1":
            rank_env = os.getenv("RANK") or os.getenv("LOCAL_RANK") or "0"
            try:
                rank_env = int(rank_env)
            except Exception:
                rank_env = rank_env
            print(
                f"[LiftEnvCfg] WORLD_SIZE={world_size} RANK={rank_env} TOTAL_ENVS={total_envs} "
                f"PER_RANK_ENVs={per_rank_envs} SCENE.num_envs={self.scene.num_envs}")

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SoArm100CubeCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"], 
            scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Gripper"],
            open_command_expr={"Gripper": 0.5},
            close_command_expr={"Gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["Fixed_Gripper"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, 0.1],
                    ),
                ),
            ],
        )

        # Configure cube marker with different color and path
        cube_marker_cfg = FRAME_MARKER_CFG.copy()
        cube_marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        }
        cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
        
        self.scene.cube_marker = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            debug_vis=True,
            visualizer_cfg=cube_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="cube",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )


@configclass
class SoArm100CubeCubeLiftEnvCfg_PLAY(SoArm100CubeCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm100CubeLiftCameraEnvCfg(SoArm100CubeCubeLiftEnvCfg):
    """Camera-enabled variant of the SO-100 cube lift environment.

    Adds a TiledCamera sensor mounted on the robot's Fixed_Jaw link (wrist camera position).
    Uses 15m env_spacing to prevent cross-environment visual contamination (camera far clip is 2.0m).
    Observation pipeline is IDENTICAL to state-based variant; Phase 3 will swap in vision encoder.
    """

    def __post_init__(self):
        # post init of parent (sets up robot, actions, scene, etc.)
        super().__post_init__()

        # Increase env spacing to prevent cross-env visual contamination
        # SO-100 arm reach is ~0.3m, camera far clip is 2.0m, so 15m ensures complete isolation
        self.scene.env_spacing = 15.0

        # Camera mode uses fewer envs due to rendering memory overhead
        try:
            camera_total = int(os.getenv("CAMERA_NUM_ENVS") or os.getenv("TOTAL_ENVS") or os.getenv("NUM_ENVS") or 1024)
        except Exception:
            camera_total = 1024
        try:
            world_size = int(os.getenv("WORLD_SIZE") or 1)
        except Exception:
            world_size = 1
        self.scene.num_envs = max(1, (camera_total + world_size - 1) // world_size)

        # Disable debug visualizers -- they show up in camera images and would confuse the vision encoder
        # (frame markers, command pose arrows, cube markers are not present on the real robot)
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
        self.scene.cube_marker.debug_vis = False

        # Mount TiledCamera on the Fixed_Gripper link (gripper body where a real wrist camera mounts)
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper/wrist_cam",
            update_period=0.0,  # update every render step
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.12,
                focus_distance=0.5,
                horizontal_aperture=3.6,
                clipping_range=(0.01, 2.0),  # far=2.0m ensures 15m-spaced neighbors are invisible
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0074, -0.05101, 0.00263),  # tuned in Isaac Sim GUI to match real SO-100 wrist camera
                rot=(0.1132, 0.9936, 0.0, 0.0),  # Euler X=167deg - forward-looking, matching real camera view
                convention="opengl",
            ),
        )


@configclass
class SoArm100CubeLiftCameraEnvCfg_PLAY(SoArm100CubeLiftCameraEnvCfg):
    """Play/evaluation variant of the camera-enabled SO-100 cube lift environment."""

    def __post_init__(self):
        # post init of parent (sets up camera, spacing, etc.)
        super().__post_init__()
        # Smaller scene for play/evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 15.0  # maintain camera spacing even in play mode
        # Disable observation corruption for evaluation
        self.observations.policy.enable_corruption = False
