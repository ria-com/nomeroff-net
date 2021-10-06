import sys
import os
import pathlib
import torch

sys.path.append("../../../")

from NomeroffNet.BBoxNpPoints import NpPointsCraft

# get models
npPointsCraft = NpPointsCraft()
npPointsCraft.load()
net = npPointsCraft.net
refine_net = npPointsCraft.refine_net
print(net)
print(refine_net)

# xs define
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
x1_refine = torch.randn(batch_size, 224, 224, 2, requires_grad=True)
x2_refine = torch.randn(batch_size, 32, 224, 224, requires_grad=True)

# to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
refine_net = refine_net.to(device)
x = x.to(device)

# predict
out1, out2 = net(x)
refine_out = refine_net(out1, out2)

# make dirs
res_model = "./model_repository/craft_net/1/model.onnx"
res_refine_model = "./model_repository/craft_refine_net/1/model.onnx"
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
                  input_names = ['refine_input', 'refine_feature'],   # the model's input names
                  output_names = ['refine_output'],  # the model's output names
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
