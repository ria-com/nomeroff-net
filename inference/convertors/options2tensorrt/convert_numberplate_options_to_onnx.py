import sys
import os
import time
import pathlib
import torch
import argparse
import numpy as np
import onnxruntime

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/numberplate_options/1/model.onnx"),
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

    options_detector = OptionsDetector()
    options_detector.load("latest")
    print(f"[INFO] torch model", options_detector.model)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device", device)

    # get model and model inputs
    model = options_detector.model
    model = model.to(device)
    x = torch.randn(batch_size,
                    options_detector.color_channels,
                    options_detector.height,
                    options_detector.width,
                    requires_grad=True)
    x = x.to(device)

    # make dirs
    p = pathlib.Path(os.path.dirname(filepath))
    p.mkdir(parents=True, exist_ok=True)

    # Export the model
    model.to_onnx(filepath, x,
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['inp_conv'],  # the model's input names
                  output_names=['fc3_line', 'fc3_reg'],  # the model's output names
                  dynamic_axes={
                      'inp_conv': {0: 'batch_size'},  # variable length axes
                      'fc3_line': {0: 'batch_size'},
                      'fc3_reg': {0: 'batch_size'}
                  })

    # Test torch model
    outs = model(x)
    start_time = time.time()
    for _ in range(n):
        outs = model(x)
    print(f"[INFO] torch time {(time.time() - start_time)/n * 1000}ms torch outs {outs}")

    # Load onnx model
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {
        input_name: np.random.randn(
            batch_size,
            options_detector.color_channels,
            options_detector.height,
            options_detector.width
        ).astype(np.float32)
    }

    # run onnx model
    print(f"[INFO] available_providers", onnxruntime.get_available_providers())
    ort_outs = ort_session.run(None, ort_inputs)
    start_time = time.time()
    for _ in range(n):
        ort_outs = ort_session.run(None, ort_inputs)
    print(f"[INFO] onnx time {(time.time() - start_time)/n * 1000}ms tensorrt outs {ort_outs}")


if __name__ == "__main__":
    main()
