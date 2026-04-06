#!/usr/bin/env python3
import os
import sys

# CRITICAL: set env BEFORE any CUDA-related imports
PHYSICAL_GPU_ID = "3"   # фізична GPU
LOCAL_GPU_ID = "0"      # всередині процесу після CUDA_VISIBLE_DEVICES

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU_ID
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/TensorRT/lib:/usr/local/cuda/lib64:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import warnings
import io
import json
import torch
import pathlib
import subprocess
from glob import glob
from pprint import pprint

# Debug versions
import tensorrt as trt
print(f"[DEBUG] Python TensorRT version: {trt.__version__}")
# print(f"[DEBUG] IRuntime Version: {trt.Runtime(trt.Logger(trt.Logger.INFO)).get_version()}")


print(f"[DEBUG] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")


# Full path to trtexec
TRTEXEC = "/usr/local/TensorRT/bin/trtexec"

# Rebuild options: can be True (rebuild all), False (rebuild none), 
# or a list of types: ["yolo", "classification", "ocr"]
FORCE_REBUILD = False

# TIP: Run with:
# LD_LIBRARY_PATH=/usr/local/TensorRT/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 python3.12 tutorials/ju/benchmark/benchmark_pipeline_trt.py

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(nomeroff_net_dir)

from nomeroff_net.pipelines.number_plate_detection_and_reading_trt_runtime import NumberPlateDetectionAndReadingTrtRuntime
from nomeroff_net.tools.mcm import get_device_torch, get_device_name
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector

DEVICE_TORCH = get_device_torch()
DEVICE_NAME_STR = get_device_name().replace(" ", '-').lower()

# Paths
PRUNED_YOLO = os.path.join(nomeroff_net_dir, "runs/prune/yolo11_pruned9/weights/best_rebuilt.pt")
PRUNED_CLASS = os.path.join(nomeroff_net_dir, "pruning_classification_all_results/best_pruned_model.pth")
OCR_PRUNED_BASE = os.path.join(nomeroff_net_dir, "tutorials/ju/train/ocr")

ENGINE_DIR = os.path.join(nomeroff_net_dir, "data/model_repository/pruned_engines")
pathlib.Path(ENGINE_DIR).mkdir(parents=True, exist_ok=True)

def read_trt_engine_bytes(engine_path: str) -> bytes:
    """Read raw TensorRT engine bytes from plain TRT files and Ultralytics metadata-wrapped engines."""
    with open(engine_path, "rb") as f:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder="little", signed=True)
            if meta_len <= 0 or meta_len > (1 << 20):
                raise ValueError("invalid metadata length")
            json.loads(f.read(meta_len).decode("utf-8"))
            return f.read()
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            f.seek(0)
            return f.read()


def is_trt_engine_compatible(engine_path: str) -> bool:
    """Return True when an existing TensorRT engine can be deserialized by the current runtime."""
    if not os.path.exists(engine_path):
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_data = read_trt_engine_bytes(engine_path)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine is not None


def build_trt_engine_from_onnx(onnx_path: str, engine_path: str, fp16: bool = True) -> bool:
    """Build TRT engine using Python TRT API (ensures version matches Python runtime)."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # parse_from_file correctly resolves relative paths to external .onnx.data files
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"[TRT Parser Error] {parser.get_error(i)}")
        return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GiB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)

        min_shape = []
        opt_shape = []
        max_shape = []

        for dim_idx, d in enumerate(inp.shape):
            if d != -1:
                min_shape.append(d)
                opt_shape.append(d)
                max_shape.append(d)
            else:
                if dim_idx == 0:  # batch
                    min_shape.append(1)
                    opt_shape.append(4)
                    max_shape.append(16)
                elif dim_idx == 2:  # height
                    min_shape.append(320)
                    opt_shape.append(640)
                    max_shape.append(1280)
                elif dim_idx == 3:  # width
                    min_shape.append(320)
                    opt_shape.append(640)
                    max_shape.append(1280)
                else:
                    raise ValueError(
                        f"Unexpected dynamic dim at index {dim_idx} for input {inp.name}: {inp.shape}"
                    )

        profile.set_shape(
            inp.name,
            tuple(min_shape),
            tuple(opt_shape),
            tuple(max_shape),
        )
    config.add_optimization_profile(profile)

    print(f"[TRT] Building engine from {onnx_path} (FP16={fp16 and builder.platform_has_fast_fp16})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[TRT] Failed to build engine!")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[TRT] Engine saved to {engine_path} ({os.path.getsize(engine_path) // 1024 // 1024} MiB)")
    return True


def ensure_trt_engine(pt_path, engine_path, model_type="yolo", name=None):
    should_rebuild = False
    if isinstance(FORCE_REBUILD, bool):
        should_rebuild = FORCE_REBUILD
    elif isinstance(FORCE_REBUILD, list):
        should_rebuild = model_type in FORCE_REBUILD

    if os.path.exists(engine_path) and not should_rebuild:
        if is_trt_engine_compatible(engine_path):
            return engine_path
        print(f"[TRT] Existing engine is incompatible with the current runtime/device, rebuilding: {engine_path}")
    
    print(f"[CONVERT] Converting {pt_path} to {engine_path} (Type: {model_type})...")
    
    if model_type == "yolo":
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(format="engine", device=LOCAL_GPU_ID, half=True, dynamic=True, simplify=True, opset=18)

        created_engine = pt_path.replace(".pt", ".engine")
        if not os.path.exists(created_engine):
            raise RuntimeError(f"YOLO TensorRT export failed for {pt_path}")

        os.replace(created_engine, engine_path)
        print(f"[TRT] Engine saved to {engine_path} ({os.path.getsize(engine_path) // 1024 // 1024} MiB)")

        onnx_file = pt_path.replace(".pt", ".onnx")
        onnx_data_file = f"{onnx_file}.data"
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
        if os.path.exists(onnx_data_file):
            os.remove(onnx_data_file)
    
    elif model_type == "classification":
        detector = OptionsDetector()
        detector.load(pt_path)
        model = detector.model.to(DEVICE_TORCH)
        onnx_path = engine_path.replace(".trt", ".onnx")
        x = torch.randn(1, detector.color_channels, detector.height, detector.width).to(DEVICE_TORCH)
        
        model.to_onnx(onnx_path, x, opset_version=18,
                      input_names=['inp_conv'], output_names=['fc3_line', 'fc3_reg'],
                      dynamic_axes={'inp_conv': {0: 'batch_size'},
                                    'fc3_line': {0: 'batch_size'},
                                    'fc3_reg': {0: 'batch_size'}})
        
        ok = build_trt_engine_from_onnx(onnx_path, engine_path)
        if not ok:
            raise RuntimeError(f"TRT engine build failed for {onnx_path}")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    elif model_type == "ocr":
        text_detector = TextDetector({name: {"for_regions": "__all__", "model_path": pt_path}})
        detector = text_detector.detectors[0]
        model = detector.model.to(DEVICE_TORCH)
        onnx_path = engine_path.replace(".trt", ".onnx")
        x = torch.randn(1, detector.color_channels, detector.height, detector.width).to(DEVICE_TORCH)
        
        model.to_onnx(onnx_path, x, opset_version=18,
                      input_names=[f'inp_{name}'], output_names=[f'out_{name}'],
                      dynamic_axes={f'inp_{name}': {0: 'batch_size'},
                                    f'out_{name}': {1: 'batch_size'}})
        
        ok = build_trt_engine_from_onnx(onnx_path, engine_path)
        if not ok:
            raise RuntimeError(f"TRT engine build failed for {onnx_path}")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    return engine_path

def run_benchmark(name, config, images, num_run=10):
    print(f"\n>>> Benchmarking {name} ...")
    pipe = NumberPlateDetectionAndReadingTrtRuntime("number_plate_detection_and_reading_trt_runtime",  image_loader="opencv", **config)
    pipe.clear_stat()
    
    # Warmup
    pipe(images[:1])
    
    for i in range(num_run):
        print(f"  Pass {i+1}/{num_run}")
        pipe(images)
    
    stat = pipe.get_timer_stat(len(images) * num_run)
    return stat

def main():
    images = glob(os.path.join(nomeroff_net_dir, "data/examples/benchmark_oneline_np_images/*"))
    if not images:
        print("No images found for benchmark!")
        return

    # 1. Base TRT Config (Default)
    base_config = {
        "path_to_model": os.path.join(nomeroff_net_dir, "data/model_repository/yolov5s/1/model.engine"),
        "path_to_classification_model": os.path.join(nomeroff_net_dir, "data/model_repository/numberplate_options/1/model.trt"),
        "presets": {
            "eu_ua_2004_2015": {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                "model_path": os.path.join(nomeroff_net_dir, "data/model_repository/ocr-eu_ua_2004_2015/1/model.trt")
            },
            "eu": {
                "for_regions": ["eu"],
                "model_path": os.path.join(nomeroff_net_dir, "data/model_repository/ocr-eu/1/model.trt")
            }
        },
        "classification_options": {
            "class_region": OptionsDetector.get_class_region_all(),
            "count_lines": OptionsDetector.get_class_count_lines_all()
        }
    }

    # 2. Pruned TRT Engines
    print("Preparing Pruned TRT Engines...")
    pruned_yolo_engine = ensure_trt_engine(PRUNED_YOLO, os.path.join(ENGINE_DIR, "yolo11_pruned.engine"), "yolo")
    pruned_class_engine = ensure_trt_engine(PRUNED_CLASS, os.path.join(ENGINE_DIR, "options_pruned.trt"), "classification")
    
    pruned_presets = {}
    ocr_targets = ["eu_ua_2004_2015_efficientnet_b2", "eu_efficientnet_b2"] # Just benchmark main ones for now
    for target in ocr_targets:
        pt = os.path.join(OCR_PRUNED_BASE, f"pruning_ocr_{target}_results", "best_pruned_model.pth")
        if os.path.exists(pt):
            engine = ensure_trt_engine(pt, os.path.join(ENGINE_DIR, f"ocr_{target}.trt"), "ocr", target)
            pruned_presets[target] = {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"] if "2004_2015" in target else ["eu"],
                "model_path": engine
            }

    pruned_config = {
        "path_to_model": pruned_yolo_engine,
        "path_to_classification_model": pruned_class_engine,
        "presets": pruned_presets,
        "classification_options": base_config["classification_options"]
    }

    # Run benchmarks
    results = {}

    # Check if Base TRT models exist
    base_exists = os.path.exists(base_config["path_to_model"]) and \
                  os.path.exists(base_config["path_to_classification_model"]) and \
                  all(os.path.exists(p["model_path"]) for p in base_config["presets"].values())

    if base_exists:
        results["Base TRT"] = run_benchmark("Base TRT", base_config, images)
    else:
        print("\n[SKIP] Base TRT benchmark skipped (models not found in data/model_repository/yolov5s/...)")

    results["Pruned TRT"] = run_benchmark("Pruned TRT", pruned_config, images)

    print("\n" + "="*80)
    if "Base TRT" in results:
        print(f"{'Metric':<40} | {'Base TRT':>12} | {'Pruned TRT':>12} | {'Speedup':>8}")
        print("-" * 80)
        for key in results["Base TRT"]:
            v_base = results["Base TRT"][key]
            v_pruned = results["Pruned TRT"].get(key, 1.0)
            speedup = v_base / v_pruned if v_pruned > 0 else 0
            print(f"{key:<40} | {v_base:>12.4f} | {v_pruned:>12.4f} | {speedup:>7.2f}x")
    else:
        print(f"{'Metric':<40} | {'Pruned TRT':>12}")
        print("-" * 60)
        for key in results["Pruned TRT"]:
            v_pruned = results["Pruned TRT"][key]
            print(f"{key:<40} | {v_pruned:>12.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
