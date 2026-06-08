import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class YOLO_v2_Block(nn.Module):
    """
    Drop-in replacement for YOLOv8/v11 blocks (C2f, C3k2, C2fPSA) that use `chunk(2, 1)`.
    Torch-Pruning natively fails to traverse `.chunk()` operations dynamically.
    This module creates two parallel explicit Convolutions to replace `cv1.chunk()`.
    """
    def __init__(self, c1, c):
        super().__init__()
        self.cv0 = Conv(c1, c, 1, 1)
        self.cv1 = Conv(c1, c, 1, 1)
        self.c = c

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2PSA_v2_Block(nn.Module):
    """
    Drop-in replacement for YOLOv11 C2PSA that uses `split((c, c), 1)`.
    """
    def __init__(self, c1, c):
        super().__init__()
        self.cv0 = Conv(c1, c, 1, 1)
        self.cv1 = Conv(c1, c, 1, 1)
        self.c = c

    def forward(self, x):
        a = self.cv0(x)
        b = self.cv1(x)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

def transfer_weights_magic(old_block, new_block):
    new_block.cv2 = old_block.cv2
    new_block.m = old_block.m
    
    state_dict = old_block.state_dict()
    state_dict_v2 = new_block.state_dict()
    
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]
    
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]
        
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]
            
    for attr_name in dir(old_block):
        attr_value = getattr(old_block, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(new_block, attr_name, attr_value)
            
    new_block.load_state_dict(state_dict_v2)

def patch_yolo_modules(module):
    """
    Recursively replaces YOLO chunk-based modules (C2f, C3k2, C2PSA) with 
    their v2 splits that are Torch-Pruning compatible!
    """
    for name, child_module in module.named_children():
        class_name = type(child_module).__name__
        
        if class_name in ['C2f', 'C3k2', 'C2fPSA']:
            # Create our slim container that just holds cv0 and cv1 + forward loop
            c1 = child_module.cv1.conv.in_channels
            c = child_module.c
            v2 = YOLO_v2_Block(c1, c)
            transfer_weights_magic(child_module, v2)
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                module[int(name)] = v2
            else:
                setattr(module, name, v2)
            
        elif class_name == 'C2PSA' or class_name == 'PSA' or class_name == 'C2':
            # Handle anything that splits into a and b and processes only b
            c1 = child_module.cv1.conv.in_channels
            c = child_module.c
            v2 = C2PSA_v2_Block(c1, c)
            transfer_weights_magic(child_module, v2)
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                module[int(name)] = v2
            else:
                setattr(module, name, v2)
            
        else:
            patch_yolo_modules(child_module)
