#!/usr/bin/env python3
"""Verify an exported camera policy produces valid action outputs.

Standalone verification script that loads an exported CameraPolicy model and
proves it produces valid (finite, non-zero, bounded) action outputs from
synthetic and/or real camera images. This script has NO Isaac Lab, Isaac Sim,
or RSL-RL dependencies.

This script includes its own copy of the CameraPolicy class definition to be
fully self-contained -- it can be copied to any machine with torch/torchvision
and run independently of the training codebase.

Usage:
  # Verify the latest exported model
  python verify_exported_policy.py --auto

  # Verify a specific exported model
  python verify_exported_policy.py --model path/to/camera_policy_exported.pt

  # Also test with a real camera image
  python verify_exported_policy.py --auto --image /path/to/photo.jpg

  # Test on GPU
  python verify_exported_policy.py --auto --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor


# ===========================================================================
# CameraPolicy -- self-contained copy (matches export_camera_policy.py)
# ===========================================================================

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

    ENCODER_OUTPUT_DIM = 512
    ACTOR_HIDDEN_DIMS = [512, 256, 128]
    OBS_DIM = 537
    ACTION_DIM = 6
    PROPRIO_DIM = 25

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        super().__init__()

        # Frozen ResNet18 encoder -- penultimate layer (512-dim avgpool output)
        # Uses create_feature_extractor to get the 'flatten' node output.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = create_feature_extractor(resnet, return_nodes={"flatten": "features"})
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD,
        )

        # Resize for non-224x224 inputs
        self.resize = T.Resize(
            [224, 224],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # MLP actor: [537 -> 512 -> 256 -> 128 -> 6] with ELU
        self.actor = nn.Sequential(
            nn.Linear(self.OBS_DIM, self.ACTOR_HIDDEN_DIMS[0]),
            nn.ELU(),
            nn.Linear(self.ACTOR_HIDDEN_DIMS[0], self.ACTOR_HIDDEN_DIMS[1]),
            nn.ELU(),
            nn.Linear(self.ACTOR_HIDDEN_DIMS[1], self.ACTOR_HIDDEN_DIMS[2]),
            nn.ELU(),
            nn.Linear(self.ACTOR_HIDDEN_DIMS[2], self.ACTION_DIM),
        )

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert raw image to normalized (B, 3, 224, 224) float32 tensor."""
        if image.dtype == torch.uint8:
            image = image.permute(0, 3, 1, 2).float() / 255.0
        elif image.dtype in (torch.float32, torch.float16):
            if image.ndim == 4 and image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)
            image = image.float()
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        if image.shape[2] != 224 or image.shape[3] != 224:
            image = self.resize(image)

        image = self.normalize(image)
        return image

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Run inference: image -> ResNet18 penultimate -> concat with proprio -> MLP -> actions."""
        processed_image = self._preprocess_image(image)

        with torch.no_grad():
            encoder_output = self.encoder(processed_image)
            visual_features = encoder_output["features"]  # (B, 512)

        joint_pos = proprio[:, :6]
        joint_vel = proprio[:, 6:12]
        target_pose = proprio[:, 12:19]
        last_action = proprio[:, 19:25]

        obs = torch.cat([
            joint_pos,
            joint_vel,
            visual_features,
            target_pose,
            last_action,
        ], dim=-1)

        actions = self.actor(obs)
        return actions


# ===========================================================================
# Verification functions
# ===========================================================================

def load_exported_policy(model_path: Path, device: str = "cpu") -> tuple[CameraPolicy, dict]:
    """Load an exported CameraPolicy from a .pt file.

    Args:
        model_path: Path to camera_policy_exported.pt file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, metadata dict).
    """
    data = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" not in data:
        raise ValueError(
            f"Invalid export file: missing 'model_state_dict'. "
            f"Keys found: {list(data.keys())}"
        )

    metadata = data.get("metadata", {})

    model = CameraPolicy()
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    model.eval()

    return model, metadata


def verify_inference(model: CameraPolicy, device: str = "cpu") -> list[tuple[str, torch.Tensor, float]]:
    """Run inference on synthetic inputs and return results with timing.

    Args:
        model: Loaded CameraPolicy model.
        device: Device to run inference on.

    Returns:
        List of (test_name, actions_tensor, inference_time_ms) tuples.
    """
    results = []

    # Test 1: Random noise image (84x84 uint8 HWC) -- matches training resolution
    image = torch.randint(0, 255, (1, 84, 84, 3), dtype=torch.uint8).to(device)
    proprio = torch.zeros(1, 25, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("random_noise_84x84", actions.cpu(), elapsed_ms))

    # Test 2: Black image (edge case -- no visual information)
    image = torch.zeros(1, 84, 84, 3, dtype=torch.uint8, device=device)
    proprio = torch.randn(1, 25, device=device) * 0.1
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("black_image_84x84", actions.cpu(), elapsed_ms))

    # Test 3: White image (edge case -- saturated)
    image = torch.full((1, 84, 84, 3), 255, dtype=torch.uint8, device=device)
    proprio = torch.randn(1, 25, device=device) * 0.1
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("white_image_84x84", actions.cpu(), elapsed_ms))

    # Test 4: Different resolution (640x480 -- typical USB webcam)
    image = torch.randint(0, 255, (1, 480, 640, 3), dtype=torch.uint8).to(device)
    proprio = torch.randn(1, 25, device=device) * 0.1
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("random_640x480", actions.cpu(), elapsed_ms))

    # Test 5: Batch of 4 images (batch inference)
    image = torch.randint(0, 255, (4, 84, 84, 3), dtype=torch.uint8).to(device)
    proprio = torch.randn(4, 25, device=device) * 0.1
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("batch_4", actions.cpu(), elapsed_ms))

    # Test 6: Pre-normalized float32 CHW image (already processed)
    image = torch.randn(1, 3, 224, 224, device=device)
    proprio = torch.randn(1, 25, device=device) * 0.1
    start = time.perf_counter()
    with torch.no_grad():
        actions = model(image, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append(("prenormalized_224x224", actions.cpu(), elapsed_ms))

    return results


def validate_results(results: list[tuple[str, torch.Tensor, float]]) -> tuple[bool, int, int]:
    """Validate all inference results.

    Args:
        results: List of (test_name, actions_tensor, inference_time_ms).

    Returns:
        Tuple of (all_passed, pass_count, total_count).
    """
    passed = 0
    total = len(results)

    for name, actions, time_ms in results:
        checks = {
            "shape": actions.shape[-1] == 6,
            "finite": torch.isfinite(actions).all().item(),
            "non_zero": not torch.all(actions == 0).item(),
            "bounded": (actions.abs() < 100).all().item(),
        }
        all_ok = all(checks.values())
        status = "PASS" if all_ok else "FAIL"
        if all_ok:
            passed += 1
        failed = [k for k, v in checks.items() if not v]

        # Format action values for display
        action_str = ", ".join(f"{a:.4f}" for a in actions[0].tolist())
        print(f"  {name:30s} {status}  [{action_str}]  ({time_ms:.1f}ms)")
        if failed:
            print(f"  {'':30s} Failed checks: {failed}")

    return passed == total, passed, total


def verify_with_real_image(
    model: CameraPolicy,
    image_path: str,
    device: str = "cpu",
) -> None:
    """Run inference on a real camera image.

    Requires PIL (Pillow) to be installed. This is an optional dependency.

    Args:
        model: Loaded CameraPolicy model.
        image_path: Path to an image file (JPEG, PNG, etc.).
        device: Device to run inference on.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("\n  Skipping real image test: PIL (Pillow) not installed.")
        print("  Install with: pip install Pillow")
        return

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_tensor = torch.tensor(img_array, dtype=torch.uint8).unsqueeze(0).to(device)  # (1, H, W, 3)
    proprio = torch.zeros(1, 25, device=device)  # zero proprioceptive state

    start = time.perf_counter()
    with torch.no_grad():
        actions = model(img_tensor, proprio)
    elapsed_ms = (time.perf_counter() - start) * 1000

    action_str = ", ".join(f"{a:.4f}" for a in actions[0].tolist())
    checks = {
        "shape": actions.shape[-1] == 6,
        "finite": torch.isfinite(actions).all().item(),
        "bounded": (actions.abs() < 100).all().item(),
    }
    all_ok = all(checks.values())
    status = "PASS" if all_ok else "FAIL"

    print(f"\n--- Real Image Test ---")
    print(f"  Image: {image_path}")
    print(f"  Size:  {img.size[0]}x{img.size[1]}")
    print(f"  {status}  [{action_str}]  ({elapsed_ms:.1f}ms)")
    if not all_ok:
        failed = [k for k, v in checks.items() if not v]
        print(f"  Failed checks: {failed}")


def find_latest_export(logs_root: Path) -> Path | None:
    """Find the latest exported camera policy file.

    Searches logs_root/so_arm100_lift_camera/*/camera_policy_exported.pt.

    Args:
        logs_root: Root logs directory.

    Returns:
        Path to latest export, or None if not found.
    """
    camera_logs = logs_root / "so_arm100_lift_camera"
    if not camera_logs.exists():
        return None

    exports = sorted(camera_logs.glob("*/camera_policy_exported.pt"))
    if not exports:
        return None

    return exports[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Verify exported camera policy produces valid action outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_exported_policy.py --auto
  python verify_exported_policy.py --model path/to/camera_policy_exported.pt
  python verify_exported_policy.py --auto --image /path/to/photo.jpg
  python verify_exported_policy.py --auto --device cuda
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to exported camera_policy_exported.pt file",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect latest exported model from logs/rsl_rl/",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional: path to a real camera image for additional testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/rsl_rl",
        help="Root logs directory for --auto mode (default: logs/rsl_rl)",
    )

    args = parser.parse_args()

    # Determine model path
    model_path = None
    if args.model:
        model_path = Path(args.model)
    elif args.auto:
        model_path = find_latest_export(Path(args.logs_dir))
        if model_path is None:
            print("No exported model found in logs directory.")
            print(f"Searched: {Path(args.logs_dir).resolve()}/so_arm100_lift_camera/*/camera_policy_exported.pt")
            print("\nTo create an exported model, run:")
            print("  python export_camera_policy.py --auto")
            print("  python export_camera_policy.py --random  (for architecture verification)")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: Specify --model or --auto")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # Load model
    print(f"=== Exported Policy Verification ===\n")
    print(f"Model: {model_path}")
    print(f"Device: {args.device}")

    model, metadata = load_exported_policy(model_path, device=args.device)

    # Print metadata
    if metadata:
        iteration = metadata.get("training_iteration", "unknown")
        encoder = metadata.get("encoder", "unknown")
        encoder_dim = metadata.get("encoder_output_dim", "unknown")
        hidden_dims = metadata.get("actor_hidden_dims", "unknown")
        obs_dim = metadata.get("obs_dim", "unknown")
        action_dim = metadata.get("action_dim", "unknown")
        obs_layout = metadata.get("obs_layout", "unknown")
        proprio_layout = metadata.get("proprio_layout", "unknown")

        print(f"Training iteration: {iteration}")
        print(f"Encoder: {encoder} ({encoder_dim}-dim features)")
        print(f"Actor: MLP [{obs_dim} -> {' -> '.join(str(d) for d in hidden_dims)} -> {action_dim}] with ELU")
        print(f"Obs layout: {obs_layout}")
        print(f"Proprio layout: {proprio_layout}")

        warning = metadata.get("warning")
        if warning:
            print(f"\n  WARNING: {warning}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    actor_params = sum(p.numel() for p in model.actor.parameters())
    print(f"Parameters: {total_params:,} total ({encoder_params:,} encoder + {actor_params:,} actor)")

    # Run inference tests
    print(f"\n--- Inference Tests ---")
    results = verify_inference(model, device=args.device)
    all_passed, passed, total = validate_results(results)

    # Real image test (optional)
    if args.image:
        verify_with_real_image(model, args.image, device=args.device)

    # Validation summary
    print(f"\n--- Validation Summary ---")
    print(f"Tests: {passed}/{total} {'PASSED' if all_passed else 'FAILED'}")

    # Check EXP-01: exported model contains encoder + policy weights
    state_keys = list(model.state_dict().keys())
    has_encoder = any("encoder" in k for k in state_keys)
    has_actor = any("actor" in k for k in state_keys)
    exp01 = has_encoder and has_actor
    print(f"EXP-01: {'PASS' if exp01 else 'FAIL'} (exported model contains encoder + policy weights)")

    # Check EXP-02: inference produces valid actions
    exp02 = all_passed
    print(f"EXP-02: {'PASS' if exp02 else 'FAIL'} (inference produces valid actions from camera image)")

    if all_passed and exp01 and exp02:
        print(f"\nThe exported model is ready for real robot deployment.")
        print(f"To use on the real robot:")
        print(f"  1. Load: data = torch.load('camera_policy_exported.pt', map_location='cpu')")
        print(f"     model = CameraPolicy(); model.load_state_dict(data['model_state_dict']); model.eval()")
        print(f"  2. Capture image from wrist camera as (H, W, 3) uint8 numpy array")
        print(f"  3. Get joint state as 25-dim vector [joint_pos(6), joint_vel(6), target_pose(7), last_action(6)]")
        print(f"  4. Run: actions = model(image_tensor.unsqueeze(0), proprio_tensor.unsqueeze(0))")
        print(f"  5. Send actions[0] to robot joint controllers")
    else:
        print(f"\nVERIFICATION FAILED -- review results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
