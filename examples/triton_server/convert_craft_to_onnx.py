import sys
import os
import pathlib
import torch

sys.path.append("../../")

from NomeroffNet.OptionsDetector import OptionsDetector


optionsDetector = OptionsDetector()
optionsDetector.load("latest")

print(optionsDetector.MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = optionsDetector.MODEL
torch_model = torch_model.to(device)

batch_size = 1
x = torch.randn(batch_size, 3, 64, 295, requires_grad=True)
x = x.to(device)
torch_out = torch_model(x)

# make dirs
res_model = "./model_repository/numberplate_options/1/model.onnx"
p = pathlib.Path(os.path.dirname(res_model))
p.mkdir(parents=True, exist_ok=True)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  res_model,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['inp_conv'],   # the model's input names
                  output_names=['fc3_line', 'fc3_reg'], # the model's output names
                  dynamic_axes={
                    'inp_conv': {0: 'batch_size'},    # variable length axes
                    'fc3_line': {0: 'batch_size'},
                    'fc3_reg': {0: 'batch_size'}
                  })
