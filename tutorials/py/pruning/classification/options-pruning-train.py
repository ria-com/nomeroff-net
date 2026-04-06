#!/usr/bin/env python
# coding: utf-8
"""
## EXAMPLES:
#### All classes:
```
    python3.12 tutorials/ju/train/classification/options-pruning-train.py \
        --dataset ./data/dataset/OptionsDetector/autoriaNumberplateOptionsDataset2-2024-05-11  \
        --config all \
        --weights ./data/models/OptionsDetector/numberplate_options/numberplate_options_2024_05_13__400x100_2_pytorch_lightning.ckpt \
        --gpu 1 \
        --max-drop 0.001 \
        --epochs-finetune 1 \
        --max-iters 10 \
        --target-pruning-ratio 0.5 \
        --global-pruning \
        --isomorphic
```
#### custom classes
```
    python3.12 tutorials/ju/train/classification/options-pruning-train.py \
        --dataset ./data/dataset/OptionsDetector/autoriaNumberplateOptionsDataset2-2024-05-11 \
        --config ua-custom \
        --weights ./data/models/OptionsDetector/numberplate_options/numberplate_options_2024_05_22__400x100_uacustom_pytorch_lightning.ckpt \
        --gpu 1 \
        --max-drop 0.001 \
        --epochs-finetune 1 \
        --max-iters 10 \
        --global-pruning \
        --isomorphic
```
"""
import os
import sys
import argparse
from datetime import datetime

# Extract GPU early, before importing torch
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch

# Nomeroff-Net path
NOMEROFF_NET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.pruning import IterativePruner
from nomeroff_net import OptionsDetector
from nomeroff_net.tools import custom_options
import pytorch_lightning as pl

# Predefined configs based on User specifications
CONFIGS = {
    "all": {
        "class_region_custom": [
            "ua-military", "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu",
            "xx-transit", "ru", "kz", "eu-ua-ordlo-dpr", "eu-ua-ordlo-lpr",
            "ge", "by", "su", "kg", "am", "md", "eu-ua-custom"
        ],
        "count_lines_custom": ["1", "2", "3"],
        "height": 100,
        "width": 400
    },
    "ua-custom": {
        "class_region_custom": [
            "eu-ua-2015", "eu-ua-2004", "eu-ua-1995", "eu", "xx-transit",
            "eu-ua-ordlo-dpr", "eu-ua-ordlo-lpr", "su", "eu-ua-custom"
        ],
        "count_lines_custom": ["1", "2", "3"],
        "height": 100,
        "width": 400
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to the base dataset")
    parser.add_argument('--config', type=str, choices=["all", "ua-custom"], required=True, help="Model configuration style")
    parser.add_argument('--weights', type=str, required=True, help="Path to weights")
    parser.add_argument('--gpu', type=int, default=1, help="GPU to use")
    parser.add_argument('--max-drop', type=float, default=0.0, help="Max accuracy drop allowed (n)")
    parser.add_argument('--epochs-finetune', type=int, default=5, help="Epochs to finetune per iter")
    parser.add_argument('--items-per-class', type=int, default=2, help="Items per class for CustomOptionsMaker")
    parser.add_argument('--max-iters', type=int, default=10, help="Max pruning iterations")
    parser.add_argument('--target-pruning-ratio', type=float, default=0.5, help="Target pruning ratio (e.g., 0.5 for 50% channels)")
    parser.add_argument('--global-pruning', action='store_true', help="Enable global pruning across all layers")
    parser.add_argument('--isomorphic', action='store_true', help="Enable isomorphic pruning (good for CNNs/ViTs)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    PATH_TO_DATASET = os.path.abspath(args.dataset)
    PATH_TO_REG_CUSTOM_DATASET = f'{PATH_TO_DATASET}_{args.config}_reg'
    
    state_ids_only_labels = ["not filled"]
    
    print(f"Generating custom classification dataset at {PATH_TO_REG_CUSTOM_DATASET} ...")
    customOptionsMakerReg = custom_options.CustomOptionsMaker(
        PATH_TO_DATASET,
        PATH_TO_REG_CUSTOM_DATASET, 
        OptionsDetector.get_class_region_all(),
        cfg["class_region_custom"],
        OptionsDetector.get_class_count_lines_all(),
        cfg["count_lines_custom"],
        OptionsDetector.get_class_state_all(),
        state_ids_only_labels,
        items_per_class=args.items_per_class
    )
    customOptionsMakerReg.make()

    class MyNpClassificator(OptionsDetector):
        def __init__(self):
            OptionsDetector.__init__(self)
            self.class_region = cfg["class_region_custom"]
            self.count_lines  = cfg["count_lines_custom"]
            self.epochs       = args.epochs_finetune
            self.batch_size   = 32
            self.gpus         = 1 if torch.cuda.device_count() else 0
            self.height       = cfg["height"]
            self.width        = cfg["width"]

    print("Initializing Region Detector...")
    npClassificator = MyNpClassificator()
    npClassificator.prepare(PATH_TO_REG_CUSTOM_DATASET, verbose=1, num_workers=4)
    npClassificator.load(args.weights)

    model = npClassificator.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def eval_fn(current_model):
        from nomeroff_net.data_loaders import ImgGenerator
        import numpy as np
        
        current_model.eval()
        current_model.to(device)
        
        test_path = os.path.join(PATH_TO_REG_CUSTOM_DATASET, "test")
        if not os.path.exists(test_path):
            test_path = os.path.join(PATH_TO_REG_CUSTOM_DATASET, "val")

        imageGenerator = ImgGenerator(
            test_path,
            npClassificator.width,
            npClassificator.height,
            npClassificator.batch_size,
            [len(npClassificator.class_region), len(npClassificator.count_lines)]
        )
        imageGenerator.build_data()
        gen = imageGenerator.path_generator()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (img_paths, inputs, labels) in enumerate(gen, 0):
                inputs_tensor = torch.from_numpy(inputs).to(device)
                outputs = current_model(inputs_tensor)
                
                label_reg = torch.from_numpy(labels[0]).to(device)
                label_cnt = torch.from_numpy(labels[1]).to(device)
                
                out_idx_reg = torch.max(outputs[0], 1)[1]
                out_idx_line = torch.max(outputs[1], 1)[1]
                
                label_idx_reg = torch.max(label_reg, 1)[1]
                label_idx_line = torch.max(label_cnt, 1)[1]
                
                correct += (out_idx_reg == label_idx_reg).sum().item()
                correct += (out_idx_line == label_idx_line).sum().item()
                total += len(label_idx_reg) * 2

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def train_fn(current_model):
        current_model.train()
        trainer = pl.Trainer(
            max_epochs=args.epochs_finetune,
            accelerator='gpu' if torch.cuda.device_count() else 'cpu',
            devices=1 if torch.cuda.device_count() else None,
            enable_checkpointing=False,
            logger=False
        )
        trainer.fit(current_model, npClassificator.dm)
        current_model.to(device)
        
    def ignored_layers_builder(m):
        ignored = []
        for name, module in m.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.out_features == len(npClassificator.class_region) or module.out_features == len(npClassificator.count_lines):
                    ignored.append(module)
        return ignored

    example_inputs = torch.randn(1, 3, npClassificator.height, npClassificator.width).to(device)

    pruner = IterativePruner(
        model=model,
        example_inputs=example_inputs,
        eval_fn=eval_fn,
        train_fn=train_fn,
        max_acc_drop=args.max_drop,
        target_pruning_ratio=args.target_pruning_ratio,
        ignored_layers_builder=ignored_layers_builder,
        save_dir=f'./pruning_classification_{args.config}_results',
        global_pruning=args.global_pruning,
        isomorphic=args.isomorphic
    )

    pruner.run(max_iters=args.max_iters)

if __name__ == "__main__":
    main()
