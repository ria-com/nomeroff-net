#!/usr/bin/env python3
"""
Compare base and pruned YOLO11 models on a validation dataset.
Measures: mAP, #params, GPU memory usage (MB), and inference speed (ms/image).

Usage:
    python3.12 tutorials/py/pruning/object_detection/eval_pruned_models.py \
        --dataset-yaml ./data/dataset/Detector/npdata/numberplate_config.yaml \
        --gpu 3 \
        --models \
            ./data/models/Detector/yolov11x-keypoints-2026-01-21.pt:base  \
            runs/prune/yolo11_pruned10/weights/best_rebuilt.pt:pruned_80pct \
            runs/prune/yolo11_pruned9/weights/best_rebuilt.pt:pruned_60pct \
            
"""
import argparse
import os
import sys
import time

for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from ultralytics import YOLO

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def count_params(model_obj) -> int:
    return sum(p.numel() for p in model_obj.parameters())


def benchmark_speed_and_memory(model_obj, imgsz: int, device: torch.device) -> dict:
    """Run warmup + timed inference to measure GPU memory delta and speed."""
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    model_obj.eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Baseline: memory already occupied (model weights, etc.)
    mem_baseline = torch.cuda.memory_allocated(device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model_obj(dummy)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Timed runs — measure peak above baseline
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            model_obj(dummy)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    # Total GPU memory = model weights + peak activation overhead
    peak_total = torch.cuda.max_memory_allocated(device)
    gpu_mem_mb = peak_total / 1e6  # full GPU footprint during inference
    ms_per_image = elapsed / BENCHMARK_RUNS * 1000

    return {"gpu_mem_mb": gpu_mem_mb, "ms_per_image": ms_per_image}


def eval_model(weight_path: str, label: str, dataset_yaml: str, imgsz: int, batch: int, gpu: int) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({weight_path})")
    print(f"{'=' * 60}")

    model = YOLO(weight_path)
    params = count_params(model.model)

    # Validation metrics
    metrics = model.val(data=dataset_yaml, imgsz=imgsz, batch=batch, device=gpu, verbose=True, plots=False)

    box = metrics.box
    pose = getattr(metrics, "pose", None)

    # Free the validation model from GPU before benchmarking to get clean memory readings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Reload model fresh for memory benchmark
    model = YOLO(weight_path)
    model.model.to(device)
    bench = benchmark_speed_and_memory(model.model, imgsz, device)

    # Free again after benchmarking
    del model
    torch.cuda.empty_cache()

    result = {
        "label": label,
        "weight": weight_path,
        "params_M": params / 1e6,
        "gpu_mem_mb": bench["gpu_mem_mb"],
        "ms_per_image": bench["ms_per_image"],
        "box_mAP50": float(box.map50),
        "box_mAP50_95": float(box.map),
    }
    if pose is not None:
        result["pose_mAP50"] = float(pose.map50)
        result["pose_mAP50_95"] = float(pose.map)

    print(f"  GPU peak mem : {bench['gpu_mem_mb']:.1f} MB")
    print(f"  Speed        : {bench['ms_per_image']:.2f} ms/image  ({1000/bench['ms_per_image']:.1f} FPS)")
    return result


def print_table(results: list[dict]):
    has_pose = any("pose_mAP50" in r for r in results)

    col_w = {
        "label": 22, "params": 10, "mem": 10, "speed": 12, "fps": 8,
        "bmap50": 10, "bmap": 13, "pmap50": 11, "pmap": 14,
    }

    header = (
        f"{'Model':<{col_w['label']}} {'Params(M)':>{col_w['params']}} "
        f"{'GPU(MB)':>{col_w['mem']}} {'ms/img':>{col_w['speed']}} {'FPS':>{col_w['fps']}} "
        f"{'Box mAP50':>{col_w['bmap50']}} {'Box mAP50-95':>{col_w['bmap']}}"
    )
    if has_pose:
        header += f" {'Pose mAP50':>{col_w['pmap50']}} {'Pose mAP50-95':>{col_w['pmap']}}"

    sep = '=' * len(header)
    print(f"\n{sep}")
    print(header)
    print('-' * len(header))

    # Find reference model — label 'base' if present, otherwise the first one
    base_label = next((r for r in results if r["label"] == "base"), results[0])
    base = base_label
    for r in results:
        param_ratio = 100.0 * r["params_M"] / base["params_M"]
        mem_ratio = 100.0 * r["gpu_mem_mb"] / base["gpu_mem_mb"]
        speedup = base["ms_per_image"] / r["ms_per_image"]
        map_diff = r["box_mAP50_95"] - base["box_mAP50_95"]

        fps = 1000.0 / r["ms_per_image"]
        row = (
            f"{r['label']:<{col_w['label']}} {r['params_M']:>{col_w['params']}.2f} "
            f"{r['gpu_mem_mb']:>{col_w['mem']}.1f} {r['ms_per_image']:>{col_w['speed']}.2f} "
            f"{fps:>{col_w['fps']}.1f} "
            f"{r['box_mAP50']:>{col_w['bmap50']}.4f} {r['box_mAP50_95']:>{col_w['bmap']}.4f}"
        )
        if has_pose:
            row += f" {r.get('pose_mAP50', 0):>{col_w['pmap50']}.4f} {r.get('pose_mAP50_95', 0):>{col_w['pmap']}.4f}"
        print(row)

        if r is not base:
            print(
                f"  {'↳ vs base:':<22} params {param_ratio:.1f}%  "
                f"GPU mem {mem_ratio:.1f}%  speedup {speedup:.2f}x  "
                f"box_mAP50-95 {map_diff:+.4f}"
            )

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare YOLO11 models (base vs pruned)")
    parser.add_argument("--dataset-yaml", required=True, help="Path to dataset yaml")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for val")
    parser.add_argument(
        "--models", nargs="+", required=True,
        metavar="PATH:LABEL",
        help="Weight paths with optional label, e.g. model.pt:base pruned.pt:pruned_80pct",
    )
    args = parser.parse_args()

    entries = []
    for spec in args.models:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path = spec
            label = os.path.basename(spec)
        entries.append((path, label))

    results = []
    for path, label in entries:
        res = eval_model(path, label, args.dataset_yaml, args.imgsz, args.batch, args.gpu)
        results.append(res)

    print_table(results)


if __name__ == "__main__":
    main()
