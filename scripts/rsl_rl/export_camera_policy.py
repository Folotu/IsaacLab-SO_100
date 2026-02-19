#!/usr/bin/env python3
"""Export camera-based policy for standalone deployment.

Combines a frozen ResNet18 vision encoder (penultimate 512-dim features) with
the trained RSL-RL MLP actor into a single PyTorch model that can run inference
using ONLY torch and torchvision -- no Isaac Lab, Isaac Sim, or RSL-RL required.

The RSL-RL checkpoint (model_N.pt) only stores the MLP actor-critic weights.
During training, the ResNet18 encoder runs inside Isaac Lab's observation
pipeline (mdp.image_features with create_feature_extractor). At deployment on
a real robot, there is no Isaac Lab, so this script creates a single model that:

  1. Takes raw RGB image (H, W, 3) uint8 + proprioceptive state (25-dim)
  2. Runs ResNet18 penultimate layer (avgpool) to produce 512-dim features
  3. Concatenates: joint_pos(6) + joint_vel(6) + visual_features(512) + target(7) + last_action(6) = 537
  4. Runs the trained MLP actor to produce 6-dim joint actions

Usage:
  # Export from a specific checkpoint
  python export_camera_policy.py --checkpoint logs/rsl_rl/so_arm100_lift_camera/2026-01-01_00-00-00/model_1500.pt

  # Auto-detect latest checkpoint
  python export_camera_policy.py --auto

  # Specify output path
  python export_camera_policy.py --auto --output /path/to/camera_policy_exported.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor


class CameraPolicy(nn.Module):
    """Standalone camera-based policy for SO-ARM-100 deployment.

    Combines a frozen ResNet18 penultimate-layer encoder (512-dim avgpool
    features) with a trained MLP actor. Takes raw camera image +
    proprioceptive state, outputs joint actions.

    Input:
        image: (B, H, W, 3) uint8 tensor -- raw camera image (any resolution)
            OR (B, 3, H, W) float32 tensor -- pre-normalized CHW format
        proprio: (B, 25) float32 tensor
            Layout: [joint_pos(6), joint_vel(6), target_pose(7), last_action(6)]

    Output:
        actions: (B, 6) float32 tensor -- joint position targets

    The observation concatenation order matches the training pipeline:
        joint_pos(6) + joint_vel(6) + visual_features(512) + target_pose(7) + last_action(6) = 537

    Encoder output: 512-dim penultimate features (avgpool, before FC layer).
    Uses create_feature_extractor to extract the 'flatten' node output.
    """

    # Architecture constants matching the training configuration
    ENCODER_OUTPUT_DIM = 512  # ResNet18 penultimate layer (avgpool, before FC)
    ACTOR_HIDDEN_DIMS = [512, 256, 128]
    OBS_DIM = 537  # 6 + 6 + 512 + 7 + 6
    ACTION_DIM = 6
    PROPRIO_DIM = 25  # 6 + 6 + 7 + 6

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        super().__init__()

        # Frozen ResNet18 encoder -- penultimate layer (512-dim avgpool output)
        # Uses create_feature_extractor to get the 'flatten' node, which is the
        # output of avgpool flattened to (B, 512), BEFORE the FC classification layer.
        # This matches Isaac Lab's mdp.image_features with create_feature_extractor.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = create_feature_extractor(resnet, return_nodes={"flatten": "features"})
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # ImageNet normalization transform
        self.normalize = T.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD,
        )

        # Resize transform for non-224x224 inputs
        self.resize = T.Resize(
            [224, 224],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # MLP actor matching RSL-RL architecture: [512, 256, 128] with ELU
        # Layer indices: 0=Linear, 1=ELU, 2=Linear, 3=ELU, 4=Linear, 5=ELU, 6=Linear
        self.actor = nn.Sequential(
            nn.Linear(self.OBS_DIM, self.ACTOR_HIDDEN_DIMS[0]),   # 0: 537 -> 512
            nn.ELU(),                                              # 1
            nn.Linear(self.ACTOR_HIDDEN_DIMS[0], self.ACTOR_HIDDEN_DIMS[1]),  # 2: 512 -> 256
            nn.ELU(),                                              # 3
            nn.Linear(self.ACTOR_HIDDEN_DIMS[1], self.ACTOR_HIDDEN_DIMS[2]),  # 4: 256 -> 128
            nn.ELU(),                                              # 5
            nn.Linear(self.ACTOR_HIDDEN_DIMS[2], self.ACTION_DIM),  # 6: 128 -> 6
        )

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert raw image to normalized (B, 3, 224, 224) float32 tensor.

        Handles two input formats:
          - (B, H, W, 3) uint8: raw camera output -- convert to CHW float, normalize, resize
          - (B, 3, H, W) float32: already in CHW format -- normalize and resize if needed
        """
        if image.dtype == torch.uint8:
            # HWC uint8 -> CHW float32 [0, 1]
            image = image.permute(0, 3, 1, 2).float() / 255.0
        elif image.dtype in (torch.float32, torch.float16):
            # Already float -- check if CHW format
            if image.ndim == 4 and image.shape[1] != 3 and image.shape[3] == 3:
                # (B, H, W, 3) float -> (B, 3, H, W)
                image = image.permute(0, 3, 1, 2)
            image = image.float()
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # Resize to 224x224 if needed (ResNet18 expects 224x224)
        if image.shape[2] != 224 or image.shape[3] != 224:
            image = self.resize(image)

        # Apply ImageNet normalization
        image = self.normalize(image)

        return image

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Run full inference: image -> ResNet18 -> concat with proprio -> MLP -> actions.

        Args:
            image: (B, H, W, 3) uint8 or (B, 3, H, W) float32 camera image.
            proprio: (B, 25) float32 proprioceptive state.
                Layout: [joint_pos(6), joint_vel(6), target_pose(7), last_action(6)]

        Returns:
            actions: (B, 6) float32 joint position targets.
        """
        # Step 1: Preprocess image to (B, 3, 224, 224) normalized float
        processed_image = self._preprocess_image(image)

        # Step 2: Run frozen ResNet18 penultimate encoder -> (B, 512) features
        with torch.no_grad():
            encoder_output = self.encoder(processed_image)
            visual_features = encoder_output["features"]  # (B, 512)

        # Step 3: Concatenate in training observation order
        # joint_pos(6) + joint_vel(6) + visual_features(512) + target_pose(7) + last_action(6) = 537
        joint_pos = proprio[:, :6]
        joint_vel = proprio[:, 6:12]
        target_pose = proprio[:, 12:19]
        last_action = proprio[:, 19:25]

        obs = torch.cat([
            joint_pos,       # 6-dim
            joint_vel,       # 6-dim
            visual_features, # 512-dim
            target_pose,     # 7-dim
            last_action,     # 6-dim
        ], dim=-1)  # -> (B, 537)

        # Step 4: Run trained MLP actor -> (B, 6) actions
        actions = self.actor(obs)

        return actions

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """Load actor MLP weights from an RSL-RL checkpoint.

        The RSL-RL checkpoint contains:
          - model_state_dict: full actor-critic state dict
          - optimizer_state_dict: optimizer state
          - iter: training iteration number

        We extract only the actor weights (not critic, not log_std).
        The state dict keys are like "actor.0.weight", "actor.0.bias", etc.

        Args:
            checkpoint_path: Path to RSL-RL model_N.pt file.

        Returns:
            Dictionary with checkpoint metadata (iteration, etc.).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state_dict"]

        # Extract actor weights only (keys like "actor.0.weight" -> "0.weight")
        actor_state = {}
        for key, value in model_state.items():
            if key.startswith("actor."):
                new_key = key[len("actor."):]
                actor_state[new_key] = value

        if not actor_state:
            raise ValueError(
                f"No actor weights found in checkpoint. "
                f"Available keys: {list(model_state.keys())[:10]}"
            )

        # Load into the actor MLP
        self.actor.load_state_dict(actor_state)

        return {
            "iteration": ckpt.get("iter", "unknown"),
            "infos": ckpt.get("infos", {}),
        }


def find_latest_checkpoint(logs_root: Path) -> Path:
    """Find the latest camera training checkpoint.

    Searches logs_root/so_arm100_lift_camera/*/model_*.pt and returns the
    highest-iteration checkpoint from the most recent run.

    Args:
        logs_root: Root logs directory (typically logs/rsl_rl/).

    Returns:
        Path to the latest model_N.pt checkpoint.
    """
    camera_logs = logs_root / "so_arm100_lift_camera"
    if not camera_logs.exists():
        raise FileNotFoundError(
            f"No camera training logs found at {camera_logs}. "
            f"Run camera training first, or use --checkpoint to specify a path."
        )

    # Find all run directories, sorted by name (timestamp-based names sort chronologically)
    runs = sorted([d for d in camera_logs.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(
            f"No training runs found in {camera_logs}. "
            f"Run camera training first."
        )

    # Take the latest run
    latest_run = runs[-1]

    # Find all model checkpoints in the run, sort by iteration number
    checkpoints = sorted(
        latest_run.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No model checkpoints found in {latest_run}. "
            f"Training may not have saved any checkpoints yet."
        )

    return checkpoints[-1]


def export_policy(
    checkpoint_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Export a camera policy from an RSL-RL checkpoint.

    Creates a standalone .pt file containing:
      - model_state_dict: CameraPolicy weights (ResNet18 encoder + MLP actor)
      - metadata: Architecture details, observation layout, training info

    Args:
        checkpoint_path: Path to RSL-RL model_N.pt file.
        output_path: Where to save the exported model. Defaults to
            camera_policy_exported.pt in the same directory as the checkpoint.

    Returns:
        Path to the exported model file.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if output_path is None:
        output_path = checkpoint_path.parent / "camera_policy_exported.pt"
    else:
        output_path = Path(output_path).resolve()

    print(f"=== Camera Policy Export ===\n")
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Output path:      {output_path}")

    # Step 1: Create CameraPolicy and load actor weights from checkpoint
    model = CameraPolicy()
    ckpt_info = model.load_from_checkpoint(checkpoint_path)
    iteration = ckpt_info["iteration"]
    model.eval()

    print(f"Loaded actor weights from iteration {iteration}")
    print(f"Encoder: ResNet18 penultimate (frozen, 512-dim avgpool features)")
    print(f"Actor: MLP {[CameraPolicy.OBS_DIM] + CameraPolicy.ACTOR_HIDDEN_DIMS + [CameraPolicy.ACTION_DIM]} with ELU")

    # Step 2: Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    actor_params = sum(p.numel() for p in model.actor.parameters())
    print(f"Parameters: {total_params:,} total ({encoder_params:,} encoder + {actor_params:,} actor)")

    # Step 3: Save exported model with metadata
    export_data = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "encoder": "resnet18",
            "encoder_weights": "IMAGENET1K_V1",
            "encoder_output_dim": CameraPolicy.ENCODER_OUTPUT_DIM,
            "actor_hidden_dims": CameraPolicy.ACTOR_HIDDEN_DIMS,
            "actor_activation": "elu",
            "obs_dim": CameraPolicy.OBS_DIM,
            "action_dim": CameraPolicy.ACTION_DIM,
            "proprio_dim": CameraPolicy.PROPRIO_DIM,
            "proprio_layout": "joint_pos(6) + joint_vel(6) + target_pose(7) + last_action(6)",
            "obs_layout": "joint_pos(6) + joint_vel(6) + visual_features(512) + target_pose(7) + last_action(6)",
            "encoder_type": "penultimate (avgpool, not FC)",
            "image_resolution": "any (resized to 224x224 internally)",
            "image_normalization": {
                "type": "imagenet",
                "mean": CameraPolicy.IMAGENET_MEAN,
                "std": CameraPolicy.IMAGENET_STD,
            },
            "training_iteration": iteration,
            "source_checkpoint": str(checkpoint_path),
            "export_date": datetime.now(timezone.utc).isoformat(),
            "export_script": "export_camera_policy.py",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_data, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExported model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Step 4: Reload and verify the export
    print("\n--- Verifying export ---")
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    verify_model = CameraPolicy()
    verify_model.load_state_dict(loaded["model_state_dict"])
    verify_model.eval()

    dummy_image = torch.randint(0, 255, (1, 84, 84, 3), dtype=torch.uint8)
    dummy_proprio = torch.randn(1, 25)
    with torch.no_grad():
        actions = verify_model(dummy_image, dummy_proprio)

    assert actions.shape == (1, 6), f"Expected (1, 6), got {actions.shape}"
    assert torch.isfinite(actions).all(), "Actions contain NaN/Inf"
    print(f"Export verified: actions shape {actions.shape}, range [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"\nMetadata:")
    for key, value in loaded["metadata"].items():
        print(f"  {key}: {value}")

    print(f"\n=== Export complete ===")
    return output_path


def export_with_random_weights(output_path: Path | None = None) -> Path:
    """Export a CameraPolicy with random (untrained) actor weights.

    Useful for verifying the export pipeline before training is complete.
    The exported model will produce actions from the randomly initialized MLP
    combined with the pretrained ResNet18 encoder.

    Args:
        output_path: Where to save. Defaults to ./camera_policy_exported.pt

    Returns:
        Path to the exported model file.
    """
    if output_path is None:
        output_path = Path("camera_policy_exported.pt")
    output_path = Path(output_path).resolve()

    print(f"=== Camera Policy Export (Random Weights) ===\n")
    print(f"No checkpoint specified -- exporting with random actor weights.")
    print(f"This is useful for verifying the architecture and export pipeline.\n")

    model = CameraPolicy()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    actor_params = sum(p.numel() for p in model.actor.parameters())
    print(f"Parameters: {total_params:,} total ({encoder_params:,} encoder + {actor_params:,} actor)")

    export_data = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "encoder": "resnet18",
            "encoder_weights": "IMAGENET1K_V1",
            "encoder_output_dim": CameraPolicy.ENCODER_OUTPUT_DIM,
            "actor_hidden_dims": CameraPolicy.ACTOR_HIDDEN_DIMS,
            "actor_activation": "elu",
            "obs_dim": CameraPolicy.OBS_DIM,
            "action_dim": CameraPolicy.ACTION_DIM,
            "proprio_dim": CameraPolicy.PROPRIO_DIM,
            "proprio_layout": "joint_pos(6) + joint_vel(6) + target_pose(7) + last_action(6)",
            "obs_layout": "joint_pos(6) + joint_vel(6) + visual_features(512) + target_pose(7) + last_action(6)",
            "encoder_type": "penultimate (avgpool, not FC)",
            "image_resolution": "any (resized to 224x224 internally)",
            "image_normalization": {
                "type": "imagenet",
                "mean": CameraPolicy.IMAGENET_MEAN,
                "std": CameraPolicy.IMAGENET_STD,
            },
            "training_iteration": "N/A (random weights)",
            "source_checkpoint": "none",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "export_script": "export_camera_policy.py",
            "warning": "Actor weights are RANDOM -- this model has not been trained.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_data, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Quick verify
    print("\n--- Verifying export ---")
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    verify_model = CameraPolicy()
    verify_model.load_state_dict(loaded["model_state_dict"])
    verify_model.eval()

    dummy_image = torch.randint(0, 255, (1, 84, 84, 3), dtype=torch.uint8)
    dummy_proprio = torch.randn(1, 25)
    with torch.no_grad():
        actions = verify_model(dummy_image, dummy_proprio)

    assert actions.shape == (1, 6), f"Expected (1, 6), got {actions.shape}"
    assert torch.isfinite(actions).all(), "Actions contain NaN/Inf"
    print(f"Export verified: actions shape {actions.shape}, range [{actions.min():.4f}, {actions.max():.4f}]")

    print(f"\n=== Export complete (random weights -- train first for real deployment) ===")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export camera-based policy for standalone deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from specific checkpoint
  python export_camera_policy.py --checkpoint logs/rsl_rl/so_arm100_lift_camera/2026-01-01/model_1500.pt

  # Auto-detect latest checkpoint
  python export_camera_policy.py --auto

  # Export with random weights (for architecture verification)
  python export_camera_policy.py --random
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to RSL-RL model_N.pt checkpoint file",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect latest camera training checkpoint from logs/rsl_rl/",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Export with random (untrained) actor weights for architecture verification",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for exported model (default: camera_policy_exported.pt alongside checkpoint)",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/rsl_rl",
        help="Root logs directory for --auto mode (default: logs/rsl_rl)",
    )

    args = parser.parse_args()

    # Determine which mode to run
    if args.random:
        export_with_random_weights(
            output_path=Path(args.output) if args.output else None,
        )
    elif args.checkpoint:
        export_policy(
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output) if args.output else None,
        )
    elif args.auto:
        try:
            checkpoint = find_latest_checkpoint(Path(args.logs_dir))
            print(f"Auto-detected checkpoint: {checkpoint}")
            export_policy(
                checkpoint_path=checkpoint,
                output_path=Path(args.output) if args.output else None,
            )
        except FileNotFoundError as e:
            print(f"\nNo trained checkpoint found: {e}")
            print(f"\nFalling back to random-weights export for architecture verification...")
            export_with_random_weights(
                output_path=Path(args.output) if args.output else None,
            )
    else:
        parser.print_help()
        print("\nError: Specify --checkpoint, --auto, or --random")
        sys.exit(1)


if __name__ == "__main__":
    main()
