"""
EXPEREMENTAL MODULE
torch time 5.273103713989258ms
onnx time 7.218122482299805ms
"""
import sys
import os
import time
import pathlib
import torch
import argparse
import numpy as np
import onnxruntime

sys.path.append("../../../../")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cn", "--craft_net_res",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/craft_net/1/model.onnx"),
                    required=False,
                    type=str,
                    help="Result craft net model filepath")
    ap.add_argument("-crn", "--craft_refine_net_res",
                    default=os.path.join(os.path.abspath(os.getcwd()),
                                         "../../../data/model_repository/craft_refine_net/1/model.onnx"),
                    required=False,
                    type=str,
                    help="Result craft refine net model filepath")
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
    res_model = args["craft_net_res"]
    res_refine_model = args["craft_refine_net_res"]
    batch_size = args["batch_size"]
    n = args["number_tests"]

    # get models
    np_points_craft = np_points_craft()
    np_points_craft.load()
    net = np_points_craft.net
    refine_net = np_points_craft.refine_net
    print(net)
    print(refine_net)

    # xs define
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    refine_net = refine_net.to(device)
    x = x.to(device)

    # predict
    out1, out2 = net(x)
    _ = refine_net(out1, out2)

    # make dirs
    p = pathlib.Path(os.path.dirname(res_model))
    p.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(os.path.dirname(res_refine_model))
    p.mkdir(parents=True, exist_ok=True)

    # Export the refine
    torch.onnx.export(refine_net,                       # model being run
                      (out1, out2),                         # model input (or a tuple for multiple inputs)
                      res_refine_model,                 # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['refine_input', 'refine_feature'],   # the model's input names
                      output_names=['refine_output'],  # the model's output names
                      dynamic_axes={'refine_input': {
                                                 0: 'batch_size',
                                                 1: 'width',
                                                 2: 'height'
                                              },
                                    'refine_feature': {
                                                 0: 'batch_size',
                                                 2: 'width',
                                                 3: 'height'
                                               },
                                    "refine_output": {
                                                 0: 'batch_size',
                                                 1: 'width',
                                                 2: 'height'
                                            }
                                    })
    # Export the model
    torch.onnx.export(net,                       # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      res_model,                 # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output', "284"],  # the model's output names
                      dynamic_axes={'input': {
                                                 0: 'batch_size',
                                                 2: 'width',
                                                 3: 'height'
                                              },
                                    'output': {
                                                 0: 'batch_size',
                                                 1: 'width',
                                                 2: 'height'
                                               },
                                    "284": {
                                                 0: 'batch_size',
                                                 2: 'width',
                                                 3: 'height'
                                            }
                                    }
                      )

    # Test torch model
    out1, out2 = net(x)
    _ = refine_net(out1, out2)
    start_time = time.time()
    for _ in range(n):
        out1, out2 = net(x)
        _ = refine_net(out1, out2)
    print(f"[INFO] torch time {(time.time() - start_time) / n * 1000}ms")

    # Load onnx model
    ort_session_craft = onnxruntime.InferenceSession(res_model, providers=['CUDAExecutionProvider',
                                                                           'CPUExecutionProvider'])
    input_name = ort_session_craft.get_inputs()[0].name

    ort_session_craft_refine = onnxruntime.InferenceSession(res_refine_model, providers=['CUDAExecutionProvider',
                                                                                         'CPUExecutionProvider'])
    input_name_refine = "refine_input"
    input_feather_name_refine = "refine_feature"
    ort_inputs = {
        input_name: np.random.randn(
            batch_size, 3, 224, 224
        ).astype(np.float32)
    }

    # run onnx model
    print(f"[INFO] available_providers", onnxruntime.get_available_providers())
    out1, out2 = ort_session_craft.run(None, ort_inputs)
    _ = ort_session_craft_refine.run(None, {
        input_name_refine: out1,
        input_feather_name_refine: out2
    })
    start_time = time.time()
    for _ in range(n):
        out1, out2 = ort_session_craft.run(None, ort_inputs)
        _ = ort_session_craft_refine.run(None, {
            input_name_refine: out1,
            input_feather_name_refine: out2
        })
    print(f"[INFO] onnx time {(time.time() - start_time) / n * 1000}ms")


if __name__ == "__main__":
    main()
