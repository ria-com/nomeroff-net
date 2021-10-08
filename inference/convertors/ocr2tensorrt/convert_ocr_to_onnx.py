import sys
import os
import time
import pathlib
import torch
import argparse
import numpy as np
import onnxruntime

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))
from NomeroffNet.TextDetector import TextDetector


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
    ap.add_argument("-n", "--number_tests",
                    default=1,
                    required=False,
                    type=int,
                    help="Number of retry tine tests")
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    filepath = args["filepath"]
    batch_size = args["batch_size"]
    n = args["number_tests"]

    # get models
    # Initialize text detector.
    textDetector = TextDetector({
        "eu_ua_2004_2015": {
            "for_regions": ["eu_ua_2015", "eu_ua_2004"],
            "model_path": "latest"
        },
        "eu_ua_1995": {
            "for_regions": ["eu_ua_1995"],
            "model_path": "latest"
        },
        "eu": {
            "for_regions": ["eu"],
            "model_path": "latest"
        },
        "ru": {
            "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
            "model_path": "latest"
        },
        "kz": {
            "for_regions": ["kz"],
            "model_path": "latest"
        },
        "ge": {
            "for_regions": ["ge"],
            "model_path": "latest"
        },
        "su": {
            "for_regions": ["su"],
            "model_path": "latest"
        }
    })

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device", device)

    model_repository_dir = "./model_repository"
    for detector, name in zip(textDetector.detectors, textDetector.detectors_names):
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

        # Test torch model
        outs = model(x)
        start_time = time.time()
        for _ in range(n):
            outs = model(x)
        print(f"[INFO] torch time {(time.time() - start_time)/n * 1000}ms torch")

        # Load onnx model
        ort_session = onnxruntime.InferenceSession(model_filepath)
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {
            input_name: np.random.randn(
                batch_size,
                detector.color_channels,
                detector.height,
                detector.width
            ).astype(np.float32)
        }

        # run onnx model
        print(f"[INFO] available_providers", onnxruntime.get_available_providers())
        ort_outs = ort_session.run(None, ort_inputs)
        start_time = time.time()
        for _ in range(n):
            ort_outs = ort_session.run(None, ort_inputs)
        print(f"[INFO] tensorrt time {(time.time() - start_time)/n * 1000}ms tensorrt")


if __name__ == "__main__":
    main()

