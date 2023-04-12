"""
Convert numberplate classification models to tensorrt

python3 ./convert_yolo_v8_to_tensorrt.py
"""
import sys
import os
import pathlib
import argparse
import subprocess

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))
from nomeroff_net.tools.mcm import modelhub, get_device_name


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/yolov8s/1/model.engine"),
                    required=False,
                    type=str,
                    help="Result onnx model filepath")
    ap.add_argument("-d", "--device",
                    default=0,
                    required=False,
                    type=str,
                    help="device number or 'cpu'")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    print(args)
    model_filepath = args["filepath"]
    device = args["device"]
    # make dirs
    p = pathlib.Path(os.path.dirname(model_filepath))
    p.mkdir(parents=True, exist_ok=True)

    device_name = get_device_name()
    device_name = device_name.replace(" ", '-').lower()
    print("=========================================")
    print(f"device_name: {device_name}")
    print("=========================================")

    model_info = modelhub.download_model_by_name("yolov8_brand_np")
    path_to_model = model_info["path"]
    print(f'yolo mode=export model={path_to_model} format=trt device={device} half;')
    res = path_to_model.replace(".pt", ".engine") # f"_{device_name}.engine"
    res_renamed = res.replace(".engine", f"_{device_name}.engine")

    subprocess.call([f'yolo mode=export model={path_to_model} format=trt device={device} half;'
                     f'mv {res} {res_renamed};'
                     f'cp {res_renamed} {model_filepath};'], shell=True)


if __name__ == "__main__":
    main()
