#!/usr/bin/env python3
import warnings
import os
from glob import glob
import sys
from pprint import pprint

#warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(nomeroff_net_dir)

from nomeroff_net import pipeline

# Base directory for pruned OCR models
OCR_PRUNED_DIR = os.path.join(nomeroff_net_dir, "tutorials/ju/train/ocr")

presets = {
    "eu_ua_2004_2015_efficientnet_b2": {
        "for_regions": ["eu_ua_2004", "eu_ua_2015"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_ua_1995_efficientnet_b2": {
        "for_regions": ["eu_ua_1995"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_ua_custom_efficientnet_b2": {
        "for_regions": ["eu_ua_custom"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "xx_transit_efficientnet_b2": {
        "for_regions": ["xx_transit"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_efficientnet_b2": {
        "for_regions": ["eu", "xx_unknown"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "kg": {  
        "for_regions": ["kg"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "su_efficientnet_b2": {
        "for_regions": ["su"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "am": {
        "for_regions": ["am"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "by": {
        "for_regions": ["by"],
        "for_count_lines": [1],
        "model_path": "latest"
    },
    "eu_2lines_efficientnet_b2": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu_ua_custom", "xx_transit",
                        "eu", "xx_unknown", "ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr", "kz",
                        "kg", "ge", "am", "by"],
        "for_count_lines": [2, 3],
        "model_path": "latest"
    },
    "su_2lines_efficientnet_b2": {
        "for_regions": ["su", "military"],
        "for_count_lines": [2, 3],
        "model_path": "latest"
    }
}

print("Loading OCR presets...")
for preset_name, preset_config in presets.items():
    pruned_path = os.path.join(OCR_PRUNED_DIR, f"pruning_ocr_{preset_name}_results", "best_pruned_model.pth")
    if os.path.exists(pruned_path):
        preset_config["model_path"] = pruned_path
        print(f"  [OK] Pruned model found for: {preset_name}")
    else:
        print(f"  [--] No pruned model for:    {preset_name} (using 'latest')")

pipeline_name = "number_plate_detection_and_reading_runtime"

print("presets", presets)
print(f"\nInitializing pipeline '{pipeline_name}'...")
number_plate_detection_and_reading = pipeline(
    pipeline_name, 
    path_to_model = os.path.join(nomeroff_net_dir, "runs/prune/yolo11_pruned9/weights/best_rebuilt.pt"),
    path_to_classification_model = os.path.join(nomeroff_net_dir, "pruning_classification_all_results/best_pruned_model.pth"),
    presets=presets,
    image_loader="turbo"
)

num_run = 20
batch_size = 1
num_workers = 1

images_dir = os.path.join(nomeroff_net_dir, "data/examples/benchmark_oneline_np_images/*")
images = glob(images_dir)
print(f"\nFound {len(images)} images in {images_dir}")

number_plate_detection_and_reading.clear_stat()

print("Starting benchmark...")
for i in range(num_run):
    print(f"pass {i+1}/{num_run}")
    outputs = number_plate_detection_and_reading(
        images, 
        batch_size=batch_size,
        num_workers=num_workers
    )

print("\n=== Benchmark Results ===")
timer_stat = number_plate_detection_and_reading.get_timer_stat(len(images) * num_run)
pprint(timer_stat)
