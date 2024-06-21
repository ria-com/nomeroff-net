"""
Convert numberplate classification models to tensorrt

python3 ./convert_numberplate_options_to_tensorrt.py
"""
import sys
import os
import pathlib
import torch
import argparse
import subprocess

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
    args = vars(ap.parse_args())
    return args


def main():
    args = parse_args()
    model_filepath = args["filepath"]
    batch_size = args["batch_size"]

    detector = OptionsDetector()
    detector.load("latest")
    print(f"[INFO] torch model", detector.model)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device", device)

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
                  input_names=['inp_conv'],  # the model's input names
                  output_names=['fc3_line', 'fc3_reg'],  # the model's output names
                  dynamic_axes={
                      'inp_conv': {0: 'batch_size'},  # variable length axes
                      'fc3_line': {0: 'batch_size'},
                      'fc3_reg': {0: 'batch_size'}
                  })

    engine_filepath = model_filepath.replace(".onnx", ".trt")
    subprocess.call([f'trtexec --onnx={model_filepath} --saveEngine={engine_filepath}'], shell=True)


if __name__ == "__main__":
    main()
