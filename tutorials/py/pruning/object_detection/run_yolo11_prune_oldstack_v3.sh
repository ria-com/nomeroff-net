#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-yolo11-prune-oldstack:cu118-torch200-modelopt029-v3}"
CONTAINER_NAME="${CONTAINER_NAME:-yolo11-prune-oldstack}"
GPU_SPEC="${GPU_SPEC:-all}"
SHM_SIZE="${SHM_SIZE:-16g}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-$SCRIPT_DIR/Dockerfile.yolo11-prune-oldstack.v3}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$SCRIPT_DIR}"

print_help() {
  cat <<USAGE
Usage:
  ./run_yolo11_prune_oldstack_v3.sh build
  ./run_yolo11_prune_oldstack_v3.sh shell
  ./run_yolo11_prune_oldstack_v3.sh test
  ./run_yolo11_prune_oldstack_v3.sh train -- <your command>

Environment variables:
  IMAGE_NAME       Docker image tag
  CONTAINER_NAME   Docker container name
  GPU_SPEC         Docker --gpus value (default: all)
  SHM_SIZE         Docker --shm-size value (default: 16g)
  PROJECT_ROOT     Host project root to mount into /workspace
  DOCKERFILE_PATH  Path to Dockerfile
  BUILD_CONTEXT    Docker build context (default: script directory)

Examples:
  ./run_yolo11_prune_oldstack_v3.sh build
  GPU_SPEC=device=1 PROJECT_ROOT=/mnt/raid/var/www/nomeroff-net ./run_yolo11_prune_oldstack_v3.sh test
  GPU_SPEC=device=1 PROJECT_ROOT=/mnt/raid/var/www/nomeroff-net ./run_yolo11_prune_oldstack_v3.sh train -- \
    python tutorials/ju/train/object_detection/yolo-pruning-train.py \
      --dataset-yaml ./data/dataset/Detector/npdata/numberplate_config.yaml \
      --weights ./data/models/Detector/yolov11x-keypoints-2026-01-21.pt \
      --gpu 0 --target-flops 66% --epochs 50 --batch 16
USAGE
}

build_image() {
  docker build --progress=plain -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$BUILD_CONTEXT"
}

docker_run() {
  local tty_flags=("$@")
  shift "$#" || true
}

run_container() {
  local tty_mode="$1"
  shift
  local -a cmd=("$@")
  local -a tty_args=()

  case "$tty_mode" in
    tty)
      tty_args=( -it )
      ;;
    stdin)
      tty_args=( -i )
      ;;
    auto)
      if [ -t 0 ] && [ -t 1 ]; then
        tty_args=( -it )
      else
        tty_args=( -i )
      fi
      ;;
    none)
      tty_args=()
      ;;
    *)
      echo "Unknown tty mode: $tty_mode"
      exit 1
      ;;
  esac

  docker run --rm "${tty_args[@]}" \
    --name "$CONTAINER_NAME" \
    --gpus "$GPU_SPEC" \
    --ipc=host \
    --shm-size "$SHM_SIZE" \
    --ulimit memlock=-1 \
    -v /mnt/raid/var/www/nomeroff-net:/var/www/nomeroff-net \
    --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    "${cmd[@]}"
}

MODE="${1:-help}"
shift || true

case "$MODE" in
  build)
    build_image
    ;;
  shell)
    run_container tty bash
    ;;
  test)
    run_container stdin python - <<'PY'
import torch
from ultralytics import YOLO
import modelopt.torch.prune as mtp

print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('modelopt prune import ok:', mtp is not None)

model = YOLO('yolo11n-pose.pt')
print('task:', model.task)
print('model loaded OK')
PY
    ;;
  train)
    if [[ "${1:-}" == "--" ]]; then
      shift
    fi
    if [[ $# -eq 0 ]]; then
      echo "No train command supplied."
      exit 1
    fi
    run_container auto "$@"
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    echo "Unknown mode: $MODE"
    print_help
    exit 1
    ;;
esac
