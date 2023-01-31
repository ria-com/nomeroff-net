"""
Convert ocr models to onnx by tensorrt

EXAMPLE:
    python3 ./convert_ocr_to_tensorrt.py
"""
import sys
import os
import pathlib
import torch
import argparse
import subprocess

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))

from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector
from nomeroff_net.pipelines.number_plate_text_reading import DEFAULT_PRESETS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/ocr-{ocr_name}/1/model.onnx"),
                    required=False,
                    type=str,
                    help="Result onnx model filepath")
    ap.add_argument("-b", "--batch_size",
                    default=1,
                    required=False,
                    type=int,
                    help="Batch Size")
    ap.add_argument("-d", "--detector_name",
                    default="eu",
                    required=False,
                    choices=list(DEFAULT_PRESETS.keys()),
                    help="Detector name")
    args = vars(ap.parse_args())
    return args


@torch.no_grad()
def main():
    args = parse_args()
    filepath = args["filepath"]
    batch_size = args["batch_size"]
    detector_name = args["detector_name"]

    # get models
    # Initialize text detector.
    text_detector = TextDetector({
        detector_name: {
            "for_regions": "__all__",
            "model_path": "latest"
        }
    })

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device", device)

    for detector, name in zip(text_detector.detectors, text_detector.detectors_names):
        print(f"\n\n[INFO] detector name", name)
        model_filepath = filepath.replace("{ocr_name}", name)

        # get model and model inputs
        model = detector.model
        model = model.to(device)
        x = torch.randn(batch_size,
                        detector.color_channels,
                        detector.height,
                        detector.width,
                        requires_grad=True)
        x = x.to(device)

        # make dirs
        p = pathlib.Path(os.path.dirname(model_filepath))
        p.mkdir(parents=True, exist_ok=True)

        # Export the model
        model.to_onnx(model_filepath, x,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[f'inp_{name}'],  # the model's input names
                      output_names=[f'out_{name}'],  # the model's output names
                      dynamic_axes={
                          f'inp_{name}': {0: 'batch_size'},  # variable length axes
                          f'out_{name}': {1: 'batch_size'},
                      })
        engine_filepath = model_filepath.replace(".onnx", ".trt")
        subprocess.call([f'trtexec --onnx={model_filepath} --saveEngine={engine_filepath}'], shell=True)


if __name__ == "__main__":
    main()

