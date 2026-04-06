#!/usr/bin/env python3
"""
Compare YOLO11 models on CPU: PyTorch / ONNX Runtime / OpenVINO.
Each benchmark runs in a SEPARATE subprocess for clean memory measurement.

Usage:
    python3.12 tutorials/py/pruning/object_detection/eval_pruned_models_cpu.py \
        --dataset-yaml ./data/dataset/Detector/npdata/numberplate_config.yaml \
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

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ─────────────────────────────────────────────────────────────────────
# Worker script — runs in a fresh subprocess per benchmark
# ─────────────────────────────────────────────────────────────────────
WORKER_SCRIPT = textwrap.dedent(r'''
import json, os, sys, time, traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

WARMUP = 5
RUNS   = 30  # CPU is slower, fewer runs

def rss_mb():
    """Current process RSS in MB (portable)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB->MB
    except Exception:
        return 0.0

def count_params(m):
    import torch
    return sum(p.numel() for p in m.parameters())

def do_val(weight, data_yaml, imgsz, batch):
    import torch
    from ultralytics import YOLO
    model = YOLO(weight)
    params = count_params(model.model)
    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device="cpu",
                        verbose=False, plots=False)
    box = metrics.box
    pose = getattr(metrics, "pose", None)
    r = {"params_M": params / 1e6,
         "box_mAP50": float(box.map50), "box_mAP50_95": float(box.map)}
    if pose is not None:
        r["pose_mAP50"] = float(pose.map50)
        r["pose_mAP50_95"] = float(pose.map)
    return r

def do_pt_cpu_bench(weight, imgsz):
    import torch
    from ultralytics import YOLO
    rss_before = rss_mb()
    model = YOLO(weight)
    model.model.to("cpu").eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cpu")
    with torch.no_grad():
        for _ in range(WARMUP):
            model.model(dummy)
    rss_after = rss_mb()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model.model(dummy)
    ms = (time.perf_counter() - start) / RUNS * 1000
    return {"ram_mb": rss_after - rss_before, "ms": ms, "backend": "PyTorch CPU"}

def do_onnx_bench(weight, imgsz):
    """Export to ONNX and benchmark with ONNX Runtime CPU."""
    import torch
    from ultralytics import YOLO
    onnx_path = str(Path(weight).with_suffix(".onnx"))
    if not Path(onnx_path).exists():
        m = YOLO(weight)
        m.export(format="onnx", imgsz=imgsz, half=False, simplify=True)
        del m

    import onnxruntime as ort
    rss_before = rss_mb()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {inp_name: dummy})
    rss_after = rss_mb()
    start = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {inp_name: dummy})
    ms = (time.perf_counter() - start) / RUNS * 1000
    return {"ram_mb": rss_after - rss_before, "ms": ms, "backend": "ONNX Runtime"}

def do_openvino_bench(weight, imgsz):
    """Export to OpenVINO IR and benchmark."""
    import torch
    from ultralytics import YOLO
    weight_p = Path(weight)
    ov_dir = str(weight_p.parent / (weight_p.stem + "_openvino_model"))
    if not Path(ov_dir).exists():
        m = YOLO(weight)
        m.export(format="openvino", imgsz=imgsz, half=False)
        del m
    # Find the .xml inside the export directory
    xml_path = None
    if Path(ov_dir).exists():
        for f in Path(ov_dir).glob("*.xml"):
            xml_path = str(f)
            break
    if xml_path is None:
        raise FileNotFoundError(f"OpenVINO .xml not found in {ov_dir}")

    from openvino import Core
    rss_before = rss_mb()
    core = Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    infer_req = compiled.create_infer_request()
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(WARMUP):
        infer_req.infer({0: dummy})
    rss_after = rss_mb()
    start = time.perf_counter()
    for _ in range(RUNS):
        infer_req.infer({0: dummy})
    ms = (time.perf_counter() - start) / RUNS * 1000
    return {"ram_mb": rss_after - rss_before, "ms": ms, "backend": "OpenVINO"}

def do_torchscript_bench(weight, imgsz):
    """Export to TorchScript and benchmark."""
    import torch
    from ultralytics import YOLO
    ts_path = str(Path(weight).with_suffix(".torchscript"))
    if not Path(ts_path).exists():
        m = YOLO(weight)
        m.export(format="torchscript", imgsz=imgsz)
        del m

    rss_before = rss_mb()
    model = torch.jit.load(ts_path, map_location="cpu")
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
    rss_after = rss_mb()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model(dummy)
    ms = (time.perf_counter() - start) / RUNS * 1000
    return {"ram_mb": rss_after - rss_before, "ms": ms, "backend": "TorchScript"}

# --- main ---
args   = json.loads(sys.argv[1])
action = args["action"]
result = {}

try:
    if action == "val":
        result = do_val(args["weight"], args["data_yaml"], args["imgsz"], args["batch"])
    elif action == "pt_cpu":
        result = do_pt_cpu_bench(args["weight"], args["imgsz"])
    elif action == "onnx":
        result = do_onnx_bench(args["weight"], args["imgsz"])
    elif action == "openvino":
        result = do_openvino_bench(args["weight"], args["imgsz"])
    elif action == "torchscript":
        result = do_torchscript_bench(args["weight"], args["imgsz"])
    result["ok"] = True
except Exception as e:
    result = {"ok": False, "error": str(e), "backend": args.get("action", "?")}
    traceback.print_exc(file=sys.stderr)

print("__RESULT__" + json.dumps(result))
''')


def run_worker(action: str, **kwargs) -> dict:
    """Spawn a fresh process for one benchmark, return JSON result."""
    payload = json.dumps({"action": action, **kwargs})
    cmd = [sys.executable, "-c", WORKER_SCRIPT, payload]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  WORKER STDERR (last 1500 chars):\n{proc.stderr[-1500:]}", file=sys.stderr)
        raise RuntimeError(f"Worker failed (action={action}): exit {proc.returncode}")

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])

    raise RuntimeError(f"Worker produced no result (action={action})")


# ─────────────────────────────────────────────────────────────────────
# Detect available backends
# ─────────────────────────────────────────────────────────────────────

def detect_backends() -> list[str]:
    """Return list of available CPU backends."""
    backends = ["pt_cpu"]  # always available

    try:
        import onnxruntime
        backends.append("onnx")
        print(f"  ✓ ONNX Runtime {onnxruntime.__version__}")
    except ImportError:
        print("  ✗ ONNX Runtime not installed (pip install onnxruntime)")

    try:
        import openvino
        backends.append("openvino")
        print(f"  ✓ OpenVINO {openvino.__version__}")
    except ImportError:
        print("  ✗ OpenVINO not installed (pip install openvino)")

    backends.append("torchscript")  # always available with torch
    print(f"  ✓ TorchScript (via PyTorch)")

    return backends


# ─────────────────────────────────────────────────────────────────────
# Evaluate one model across all backends
# ─────────────────────────────────────────────────────────────────────

BACKEND_LABELS = {
    "pt_cpu": "PyTorch CPU",
    "onnx": "ONNX Runtime",
    "openvino": "OpenVINO",
    "torchscript": "TorchScript",
}


def eval_model(weight_path: str, label: str, data_yaml: str,
               imgsz: int, batch: int, backends: list[str]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({weight_path})")
    print(f"{'=' * 60}")

    # 1) Validation (mAP) on CPU
    print(f"  [val] Running validation on CPU …")
    val = run_worker("val", weight=weight_path, data_yaml=data_yaml,
                     imgsz=imgsz, batch=batch)
    print(f"        mAP50-95={val['box_mAP50_95']:.4f}")

    # 2) Benchmark each backend
    bench_results = {}
    for i, backend in enumerate(backends, 1):
        blabel = BACKEND_LABELS.get(backend, backend)
        print(f"  [{i}/{len(backends)}] {blabel} …")
        try:
            res = run_worker(backend, weight=weight_path, imgsz=imgsz)
            if res.get("ok"):
                fps = 1000.0 / res["ms"]
                print(f"        RAM delta={res['ram_mb']:.0f} MB  "
                      f"{res['ms']:.1f} ms/img  ({fps:.1f} FPS)")
                bench_results[backend] = res
            else:
                print(f"        FAILED: {res.get('error', '?')}")
        except Exception as e:
            print(f"        FAILED: {e}")

    return {
        "label": label,
        "params_M": val["params_M"],
        "box_mAP50": val["box_mAP50"],
        "box_mAP50_95": val["box_mAP50_95"],
        "pose_mAP50": val.get("pose_mAP50"),
        "pose_mAP50_95": val.get("pose_mAP50_95"),
        "benchmarks": bench_results,
    }


# ─────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], backends: list[str]):
    has_pose = any(r.get("pose_mAP50") is not None for r in results)
    base = next((r for r in results if r["label"] == "base"), results[0])

    LW = 28
    W  = 10

    header = (f"{'Model':<{LW}} {'Params(M)':>{W}} {'RAM(MB)':>{W}}"
              f" {'ms/img':>{W}} {'FPS':>{W}}"
              f" {'Box mAP50':>{W}} {'mAP50-95':>{W}}")
    if has_pose:
        header += f" {'PosemAP50':>{W}} {'mAP50-95':>{W}}"

    sep  = "=" * len(header)
    thin = "-" * len(header)

    def fmt(tag, params, ram, ms, map50=None, map95=None, pm50=None, pm95=None):
        fps = 1000.0 / ms if ms > 0 else 0
        s = (f"{tag:<{LW}} {params:>{W}.2f} {ram:>{W}.0f}"
             f" {ms:>{W}.1f} {fps:>{W}.1f}")
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

    # Find the base PT benchmark for "total speedup" calculations
    base_pt = base["benchmarks"].get("pt_cpu", {})
    base_pt_ms = base_pt.get("ms", 1.0)

    for r in results:
        benches = r["benchmarks"]

        for backend in backends:
            if backend not in benches:
                continue
            b = benches[backend]
            blabel = BACKEND_LABELS.get(backend, backend)
            tag = f"{r['label']}  [{blabel}]"

            if backend == "pt_cpu":
                print(fmt(tag, r["params_M"], b["ram_mb"], b["ms"],
                          r["box_mAP50"], r["box_mAP50_95"],
                          r.get("pose_mAP50"), r.get("pose_mAP50_95")))
            else:
                print(fmt(tag, r["params_M"], b["ram_mb"], b["ms"]))
                # speedup vs PT CPU (same model)
                pt_b = benches.get("pt_cpu")
                if pt_b:
                    su = pt_b["ms"] / b["ms"]
                    print(f"  {'vs own PT CPU:':<{LW-2}} speedup {su:.2f}x")

        # vs base comparisons
        if r is not base:
            md = r["box_mAP50_95"] - base["box_mAP50_95"]
            pr = 100.0 * r["params_M"] / base["params_M"]

            # PT vs base PT
            r_pt = benches.get("pt_cpu")
            if r_pt and base_pt:
                su = base_pt_ms / r_pt["ms"]
                print(f"  {'PT vs base PT:':<{LW-2}} params {pr:.1f}%"
                      f"  speedup {su:.2f}x  mAP50-95 {md:+.4f}")

            # Best optimized vs base PT (the real-world upgrade)
            best_backend = None
            best_ms = 999999
            for bk in backends:
                if bk in benches and benches[bk]["ms"] < best_ms:
                    best_ms = benches[bk]["ms"]
                    best_backend = bk
            if best_backend and best_backend != "pt_cpu":
                total_su = base_pt_ms / best_ms
                bl = BACKEND_LABELS.get(best_backend, best_backend)
                print(f"  {'⚡ Best vs base PT:':<{LW-2}} [{bl}]"
                      f"  speedup {total_su:.2f}x  mAP50-95 {md:+.4f}")

        print(thin)
    print(sep)

    # Summary: best backend
    print("\n📊 Best CPU backend per model:")
    for r in results:
        benches = r["benchmarks"]
        if not benches:
            continue
        best_bk = min(benches, key=lambda k: benches[k]["ms"])
        b = benches[best_bk]
        bl = BACKEND_LABELS.get(best_bk, best_bk)
        print(f"  {r['label']:<20} → {bl:<16} {b['ms']:.1f} ms/img  "
              f"({1000/b['ms']:.1f} FPS)")


# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPU benchmark: PyTorch / ONNX Runtime / OpenVINO / TorchScript")
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Batch for val (CPU is slow)")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip mAP validation (CPU val is very slow)")
    parser.add_argument("--backends", nargs="*", default=None,
                        help="Subset of backends: pt_cpu onnx openvino torchscript")
    parser.add_argument("--models", nargs="+", required=True, metavar="PATH:LABEL")
    args = parser.parse_args()

    print("Detecting available CPU backends …")
    available = detect_backends()

    if args.backends:
        backends = [b for b in args.backends if b in available]
    else:
        backends = available
    print(f"Will benchmark: {', '.join(BACKEND_LABELS.get(b, b) for b in backends)}\n")

    entries = []
    for spec in args.models:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, Path(spec).name
        entries.append((path, label))

    results = []
    for path, label in entries:
        r = eval_model(path, label, args.dataset_yaml, args.imgsz, args.batch, backends)
        results.append(r)

    print_table(results, backends)


if __name__ == "__main__":
    main()
