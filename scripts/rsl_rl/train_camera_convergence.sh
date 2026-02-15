#!/bin/bash
# Camera-based training convergence script for SO-ARM-100 cube lift
#
# Usage:
#   ./train_camera_convergence.sh [--mode dev|production] [--iterations N] [--num-envs N] [--baseline] [--logdir DIR]
#
# Presets:
#   dev:        T4 16GB  -- 2 envs, 200 iterations (smoke test, ~7 min)
#   production: g5/g6    -- 512 envs, 3000 iterations (full convergence, ~2-4 hours)
#
# Environment requirements:
#   - conda env 'embodied-ai' with isaaclab, isaaclab_rl, rsl_rl, SO_100 installed
#   - Isaac Sim _isaac_sim/setup_conda_env.sh sourced for omni/isaacsim modules
#   - Or: set ISAACLAB_PATH env var pointing to IsaacLab root
#   - On AWS Batch/production: these are pre-configured in the AMI/container
#
# The camera variant (SO-ARM100-Lift-Cube-Camera-v0) uses:
#   - Frozen ResNet18 encoder producing 1000-dim visual features
#   - 1025-dim observations (6 joint_pos + 6 joint_vel + 1000 visual + 7 target + 6 action)
#   - [512, 256, 128] hidden dims MLP policy
#   - Domain randomization (object color + table texture)
#
# State-based baseline (SO-ARM100-Lift-Cube-v0) uses:
#   - 28-dim observations (direct state)
#   - [256, 128, 64] hidden dims MLP policy
#   - max_iterations=1500 in LiftCubePPORunnerCfg
#
# For CAM-06 validation: camera reward must exceed 50% of state-based baseline reward
# at convergence. Run state-based baseline first, then camera, then compare with
# compare_baselines.py.
#
# Production (AWS Batch g5/g6):
#   1. Ensure IsaacLab + SO_100 extension installed on AMI
#   2. Run: ./train_camera_convergence.sh --mode production
#   3. Monitor: tensorboard --logdir logs/rsl_rl/so_arm100_lift_camera
#   4. Full convergence takes ~2-4 hours on g5.xlarge (A10G 24GB)
#   5. After camera finishes: ./train_camera_convergence.sh --mode production --baseline
#   6. Compare: python compare_baselines.py --auto

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Environment setup ---
# Resolve Isaac Lab root for sourcing Isaac Sim conda env
if [ -z "${ISAACLAB_PATH:-}" ]; then
    if [ -f "$HOME/Documents/IsaacLab/_isaac_sim/setup_conda_env.sh" ]; then
        ISAACLAB_PATH="$HOME/Documents/IsaacLab"
    fi
fi

# Activate embodied-ai conda env if not already active
if [ "${CONDA_DEFAULT_ENV:-}" != "embodied-ai" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        echo "[INFO] Activating conda env: embodied-ai"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate embodied-ai
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
        conda activate embodied-ai
    fi
fi

# Source Isaac Sim conda env setup (adds isaacsim/omni to PYTHONPATH)
if [ -n "${ISAACLAB_PATH:-}" ] && [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    echo "[INFO] Sourcing Isaac Sim conda env from: ${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
    source "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi

# --- Parse arguments ---
MODE="dev"
ITERATIONS=""
NUM_ENVS=""
BASELINE=false
LOGDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --baseline)
            BASELINE=true
            shift
            ;;
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mode dev|production] [--iterations N] [--num-envs N] [--baseline] [--logdir DIR]"
            echo ""
            echo "Options:"
            echo "  --mode dev|production   Hardware preset (default: dev)"
            echo "  --iterations N          Override max iterations"
            echo "  --num-envs N            Override number of environments"
            echo "  --baseline              Run state-based baseline instead of camera"
            echo "  --logdir DIR            Override log directory"
            echo ""
            echo "Presets:"
            echo "  dev:        T4 16GB  -- 2 envs (camera) / 4 envs (state), 200 iterations"
            echo "  production: g5/g6    -- 512 envs, 3000 iterations"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- Apply mode presets ---
if [ "$BASELINE" = true ]; then
    TASK="SO-ARM100-Lift-Cube-v0"
    TASK_LABEL="state-based"
    case "$MODE" in
        dev)
            : "${NUM_ENVS:=4}"
            : "${ITERATIONS:=200}"
            ;;
        production)
            : "${NUM_ENVS:=512}"
            : "${ITERATIONS:=3000}"
            ;;
        *)
            echo "Error: Unknown mode '$MODE'. Use 'dev' or 'production'."
            exit 1
            ;;
    esac
    ENABLE_CAMERAS=""
else
    TASK="SO-ARM100-Lift-Cube-Camera-v0"
    TASK_LABEL="camera-based"
    case "$MODE" in
        dev)
            : "${NUM_ENVS:=2}"
            : "${ITERATIONS:=200}"
            ;;
        production)
            : "${NUM_ENVS:=512}"
            : "${ITERATIONS:=3000}"
            ;;
        *)
            echo "Error: Unknown mode '$MODE'. Use 'dev' or 'production'."
            exit 1
            ;;
    esac
    ENABLE_CAMERAS="--enable_cameras"
fi

echo "============================================"
echo "  SO-ARM100 Training: ${TASK_LABEL}"
echo "============================================"
echo "  Mode:       $MODE"
echo "  Task:       $TASK"
echo "  Envs:       $NUM_ENVS"
echo "  Iterations: $ITERATIONS"
echo "  Log dir:    ${LOGDIR:-<default>}"
echo "============================================"
echo ""

cd "$PROJECT_DIR"

# Build command -- train.py handles --enable_cameras and --headless via AppLauncher args
CMD="python scripts/rsl_rl/train.py"
CMD="$CMD --task $TASK"
CMD="$CMD --num_envs $NUM_ENVS"
CMD="$CMD --max_iterations $ITERATIONS"
CMD="$CMD --headless"

if [ -n "$ENABLE_CAMERAS" ]; then
    CMD="$CMD $ENABLE_CAMERAS"
fi

if [ -n "$LOGDIR" ]; then
    CMD="$CMD --logdir $LOGDIR"
fi

echo "Running: $CMD"
echo ""

# Execute training
eval $CMD

echo ""
echo "============================================"
echo "  Training complete: ${TASK_LABEL}"
echo "============================================"
echo "  Check logs in: logs/rsl_rl/"
echo "  TensorBoard:   tensorboard --logdir logs/rsl_rl/"
echo "  Compare:        python scripts/rsl_rl/compare_baselines.py --auto"
echo "============================================"
