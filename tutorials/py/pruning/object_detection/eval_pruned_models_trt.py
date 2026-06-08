#!/usr/bin/env python3
"""
Compare YOLO11 models: PyTorch FP32 + TensorRT FP16.
Each benchmark runs in a SEPARATE subprocess for clean GPU memory measurement.

Usage:
    python3.12 tutorials/py/pruning/object_detection/eval_pruned_models_trt.py \
        --dataset-yaml ./data/dataset/Detector/npdata/numberplate_config.yaml \
        --gpu 3 \
        --models \
            ./data/models/Detector/yolov11x-keypoints-2026-01-21.pt:base \
            runs/prune/yolo11_pruned10/weights/best_rebuilt.pt:pruned_80pct \
            runs/prune/yolo11_pruned9/weights/best_rebuilt.pt:pruned_60pct
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Worker script executed in a fresh subprocess per model/backend
# ─────────────────────────────────────────────────────────────────────
WORKER_SCRIPT = textwrap.dedent(r'''
import json, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch, numpy as np
from ultralytics import YOLO

WARMUP = 10
RUNS   = 100

def vram_used_mb():
    """Real VRAM used (NVML), same as nvidia-smi."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 1e6

def vram_baseline_mb():
    """Measure VRAM before loading any model (CUDA runtime overhead only)."""
    # Force CUDA context init
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    return vram_used_mb()

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def do_val(weight, data_yaml, imgsz, batch):
    model = YOLO(weight)
    params = count_params(model.model)
    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=0,
                        verbose=False, plots=False)
    box = metrics.box
    pose = getattr(metrics, "pose", None)
    r = {"params_M": params / 1e6,
         "box_mAP50": float(box.map50), "box_mAP50_95": float(box.map)}
    if pose is not None:
        r["pose_mAP50"] = float(pose.map50)
        r["pose_mAP50_95"] = float(pose.map)
    del model; torch.cuda.empty_cache()
    return r

def do_pt_bench(weight, imgsz):
    baseline = vram_baseline_mb()
    model = YOLO(weight)
    model.model.to("cuda:0").eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda:0")
    with torch.no_grad():
        for _ in range(WARMUP):
            model.model(dummy)
    torch.cuda.synchronize()
    vram_delta = vram_used_mb() - baseline
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model.model(dummy)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / RUNS * 1000
    del model; torch.cuda.empty_cache()
    return {"gpu_mb": max(vram_delta, 0.0), "ms": ms}

def do_trt_bench(engine_path, imgsz):
    baseline = vram_baseline_mb()
    model = YOLO(engine_path)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    vram_delta = vram_used_mb() - baseline
    start = time.perf_counter()
    for _ in range(RUNS):
        model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / RUNS * 1000
    del model; torch.cuda.empty_cache()
    return {"gpu_mb": max(vram_delta, 0.0), "ms": ms}

# --- main ---
args   = json.loads(sys.argv[2])
action = args["action"]
result = {}

if action == "val":
    result = do_val(args["weight"], args["data_yaml"], args["imgsz"], args["batch"])
elif action == "pt_bench":
    result = do_pt_bench(args["weight"], args["imgsz"])
elif action == "trt_export":
    engine = str(Path(args["weight"]).with_suffix(".engine"))
    if not Path(engine).exists():
        m = YOLO(args["weight"])
        m.export(format="engine", half=True, batch=1, imgsz=args["imgsz"],
                 device=0, workspace=4)
        del m; torch.cuda.empty_cache()
    result = {"engine_path": engine}
elif action == "trt_bench":
    result = do_trt_bench(args["engine_path"], args["imgsz"])

print("__RESULT__" + json.dumps(result))
''')


def run_worker(gpu: int, action: str, **kwargs) -> dict:
    """Spawn a fresh python process, run one benchmark action, return JSON result."""
    payload = json.dumps({"action": action, **kwargs})
    cmd = [sys.executable, "-c", WORKER_SCRIPT, str(gpu), payload]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        print(f"  WORKER STDERR:\n{proc.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"Worker failed (action={action}): exit {proc.returncode}")

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])

    raise RuntimeError(f"Worker produced no result (action={action})")


# ─────────────────────────────────────────────────────────────────────
# Evaluate one model
# ─────────────────────────────────────────────────────────────────────

def eval_model(weight_path: str, label: str, data_yaml: str,
               imgsz: int, batch: int, gpu: int, skip_trt: bool) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({weight_path})")
    print(f"{'=' * 60}")

    # 1) Validation (mAP) — separate process
    print(f"  [1/3] Running validation …")
    val = run_worker(gpu, "val", weight=weight_path, data_yaml=data_yaml,
                     imgsz=imgsz, batch=batch)
    print(f"        mAP50-95={val['box_mAP50_95']:.4f}")

    # 2) PyTorch speed + mem — separate process
    print(f"  [2/3] PyTorch benchmark …")
    pt = run_worker(gpu, "pt_bench", weight=weight_path, imgsz=imgsz)
    pt_fps = 1000.0 / pt["ms"]
    print(f"        GPU={pt['gpu_mb']:.0f} MB  {pt['ms']:.2f} ms/img  ({pt_fps:.1f} FPS)")

    # 3) TRT export + bench — two separate processes
    trt: dict = {}
    if not skip_trt:
        print(f"  [3/3] TensorRT FP16 …")
        try:
            exp = run_worker(gpu, "trt_export", weight=weight_path, imgsz=imgsz)
            engine_path = exp["engine_path"]
            trt = run_worker(gpu, "trt_bench", engine_path=engine_path, imgsz=imgsz)
            trt_fps = 1000.0 / trt["ms"]
            print(f"        GPU={trt['gpu_mb']:.0f} MB  {trt['ms']:.2f} ms/img  ({trt_fps:.1f} FPS)")
        except Exception as e:
            print(f"        TRT failed: {e}")
    else:
        print(f"  [3/3] TRT skipped")

    return {
        "label": label,
        "params_M": val["params_M"],
        "pt_gpu_mb": pt["gpu_mb"],  "pt_ms": pt["ms"],
        "trt_gpu_mb": trt.get("gpu_mb"), "trt_ms": trt.get("ms"),
        "box_mAP50": val["box_mAP50"], "box_mAP50_95": val["box_mAP50_95"],
        "pose_mAP50": val.get("pose_mAP50"), "pose_mAP50_95": val.get("pose_mAP50_95"),
    }


# ─────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], has_trt: bool):
    has_pose = any(r.get("pose_mAP50") is not None for r in results)
    base = next((r for r in results if r["label"] == "base"), results[0])

    LW = 26
    W  = 10

    header = (f"{'Model':<{LW}} {'Params(M)':>{W}} {'GPU(MB)':>{W}}"
              f" {'ms/img':>{W}} {'FPS':>{W}}"
              f" {'Box mAP50':>{W}} {'mAP50-95':>{W}}")
    if has_pose:
        header += f" {'PosemAP50':>{W}} {'mAP50-95':>{W}}"

    sep  = "=" * len(header)
    thin = "-" * len(header)

    def row(tag, params, gpu, ms, map50=None, map95=None, pm50=None, pm95=None):
        fps = 1000.0 / ms
        s = (f"{tag:<{LW}} {params:>{W}.2f} {gpu:>{W}.0f}"
             f" {ms:>{W}.2f} {fps:>{W}.1f}")
        if map50 is not None:
            s += f" {map50:>{W}.4f} {map95:>{W}.4f}"
        else:
            s += f" {'—':>{W}} {'—':>{W}}"
        if has_pose:
            if pm50 is not None:
                s += f" {pm50:>{W}.4f} {pm95:>{W}.4f}"
            else:
                s += f" {'—':>{W}} {'—':>{W}}"
        return s

    print(f"\n{sep}\n{header}\n{thin}")

    for r in results:
        # PT row
        print(row(f"{r['label']}  [PT]", r["params_M"], r["pt_gpu_mb"], r["pt_ms"],
                  r["box_mAP50"], r["box_mAP50_95"],
                  r.get("pose_mAP50"), r.get("pose_mAP50_95")))

        # TRT row
        if has_trt and r["trt_ms"] is not None:
            print(row(f"{r['label']}  [TRT FP16]", r["params_M"],
                      r["trt_gpu_mb"], r["trt_ms"]))
            # TRT vs PT (same model)
            su = r["pt_ms"] / r["trt_ms"]
            mr = 100.0 * r["trt_gpu_mb"] / r["pt_gpu_mb"]
            print(f"  {'TRT vs PT:':<{LW-2}} speedup {su:.2f}x  mem {mr:.1f}% of PT")

        # vs base
        if r is not base:
            pr = 100.0 * r["params_M"] / base["params_M"]
            md = r["box_mAP50_95"] - base["box_mAP50_95"]
            # PT vs base PT
            pt_su = base["pt_ms"] / r["pt_ms"]
            pt_mr = 100.0 * r["pt_gpu_mb"] / base["pt_gpu_mb"]
            print(f"  {'PT vs base PT:':<{LW-2}} params {pr:.1f}%  mem {pt_mr:.1f}%"
                  f"  speedup {pt_su:.2f}x  mAP50-95 {md:+.4f}")
            # TRT vs base TRT (same backend comparison)
            if has_trt and r["trt_ms"] is not None and base.get("trt_ms"):
                trt_su = base["trt_ms"] / r["trt_ms"]
                trt_mr = 100.0 * r["trt_gpu_mb"] / base["trt_gpu_mb"]
                print(f"  {'TRT vs base TRT:':<{LW-2}} mem {trt_mr:.1f}%"
                      f"  speedup {trt_su:.2f}x")
            # Total: pruned TRT vs base PT (the real-world upgrade)
            if has_trt and r["trt_ms"] is not None:
                total_su = base["pt_ms"] / r["trt_ms"]
                total_mr = 100.0 * r["trt_gpu_mb"] / base["pt_gpu_mb"]
                print(f"  {'⚡ TRT vs base PT:':<{LW-2}} mem {total_mr:.1f}%"
                      f"  speedup {total_su:.2f}x  mAP50-95 {md:+.4f}")
        print(thin)
    print(sep)


# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11 models (PyTorch + TensorRT FP16) with subprocess isolation")
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument("--models", nargs="+", required=True, metavar="PATH:LABEL")
    args = parser.parse_args()

    entries = []
    for spec in args.models:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, Path(spec).name
        entries.append((path, label))

    results = []
    for path, label in entries:
        r = eval_model(path, label, args.dataset_yaml,
                       args.imgsz, args.batch, args.gpu, args.skip_trt)
        results.append(r)

    has_trt = not args.skip_trt and any(r["trt_ms"] is not None for r in results)
    print_table(results, has_trt)


if __name__ == "__main__":
    main()
