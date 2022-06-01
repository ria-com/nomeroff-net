"""
Convert numberplate classification models to tensorrt

python3 ./convert_numberplate_options_to_tensorrt.py
"""
import sys
import os
import pathlib
import argparse
import subprocess

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))
from nomeroff_net.tools.mcm import modelhub


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/yolov5s/1/model.engine"),
                    required=False,
                    type=str,
                    help="Result onnx model filepath")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    model_filepath = args["filepath"]

    # make dirs
    p = pathlib.Path(os.path.dirname(model_filepath))
    p.mkdir(parents=True, exist_ok=True)

    # download and append to path yolo repo
    info = modelhub.download_repo_for_model("yolov5")
    repo_path = info["repo_path"]
    model_info = modelhub.download_model_by_name("yolov5")
    path_to_model = model_info["path"]
    print(f'python3 ./export.py --weights={path_to_model} --include=engine --device 0 --dynamic;')
    res = path_to_model.replace(".pt", ".engine")
    # python3 ./export.py --weights yolov5s-2021-12-14.pt --include engine --device 0 --dinamic
    subprocess.call([f'cd {repo_path}; '
                     f'python3 ./export.py --weights={path_to_model} --include=engine --device 0;'
                     f'cp {res} {model_filepath}'], shell=True)


if __name__ == "__main__":
    main()
