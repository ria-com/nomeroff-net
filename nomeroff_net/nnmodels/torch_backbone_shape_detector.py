from torch import nn
import numpy as np


def get_net_config(backbone_model, w=200, h=50, c=3):
    modules = [module for name, module in backbone_model.named_modules()
               if not isinstance(module, nn.Sequential) and 'downsample' not in name]
    config = []
    out_channels = c
    for module in modules:
        if "MaxPool" in str(type(module)):
            layer_conf = {
                "padding": module.padding,
                "out_channels": out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "dilation": module.dilation,
            }
            config.append(layer_conf)
        if "Conv2d" in str(type(module)):
            out_channels = module.out_channels
            layer_conf = {
                "padding": module.padding,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "dilation": module.dilation,
            }
            config.append(layer_conf)
    return config


def conv2d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    """
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(dilation, int):
        dilation = (dilation, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int)
    )
    return output_size


def get_output_shape(shape, conv_nn):
    nn_conf = get_net_config(conv_nn)
    for i, layer_conf in enumerate(nn_conf):
        # INPUT:  1, 3,   50, 200
        # OUTPUT: 1, 256, 4,  13
        shape = conv2d_output_size(shape,
                                   out_channels=layer_conf["out_channels"],
                                   kernel_size=layer_conf["kernel_size"],
                                   stride=layer_conf["stride"],
                                   padding=layer_conf["padding"],
                                   dilation=layer_conf.get("dilation", None))
    return shape
