#!/usr/bin/env python
# coding: utf-8
"""
EXAMPLE TESTING COMMAND (for a single model):
```
    python3.12 tutorials/ju/train/ocr/ocr-pruning-train.py \
        --model-name ae_efficientnet_b2 \
        --dataset auto \
        --weights latest \
        --gpu 1 \
        --max-drop 0.001 \
        --epochs-finetune 1 \
        --max-iters 10 \
        --target-pruning-ratio 0.5 \
        --global-pruning \
        --isomorphic
```
"""
import os
import sys
import argparse
from datetime import datetime
import warnings

# Set CUDA_VISIBLE_DEVICES before importing torch
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch

# Nomeroff-Net path
NOMEROFF_NET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.pruning import IterativePruner
from nomeroff_net.pipes.number_plate_text_readers.base.ocr import OCR
from nomeroff_net.tools import modelhub
import pytorch_lightning as pl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help="Model name in ModelHub (e.g. 'eu', 'su', 'ae')")
    parser.add_argument('--dataset', type=str, default="auto", help="Path to the dataset or 'auto'")
    parser.add_argument('--weights', type=str, default="latest", help="Path to weights or 'latest'")
    parser.add_argument('--gpu', type=int, default=1, help="GPU to use")
    parser.add_argument('--max-drop', type=float, default=0.0, help="Max accuracy drop allowed (n)")
    parser.add_argument('--epochs-finetune', type=int, default=5, help="Epochs to finetune per iter")
    parser.add_argument('--max-iters', type=int, default=10, help="Max pruning iterations")
    parser.add_argument('--target-pruning-ratio', type=float, default=0.5, help="Target pruning ratio (e.g., 0.5 for 50% channels)")
    parser.add_argument('--global-pruning', action='store_true', help="Enable global pruning across all layers")
    parser.add_argument('--isomorphic', action='store_true', help="Enable isomorphic pruning (good for CNNs/ViTs)")
    args = parser.parse_args()

    ocr = OCR()
    ocr.model_name = args.model_name
    ocr.batch_size = 32
    ocr.epochs = args.epochs_finetune
    ocr.gpus = 1 if torch.cuda.device_count() else 0
    
    # Load architecture and config parameters (letters, height, width, backbone) from meta
    print(f"Loading weights & metadata for model: {args.model_name}")
    ocr.load(args.weights)
    
    dataset_path = args.dataset
    if dataset_path == "auto":
        print(f"Downloading/Locating dataset for model '{args.model_name}'...")
        info = modelhub.download_dataset_for_model(args.model_name)
        dataset_path = info["dataset_path"]
        print(f"Dataset path automatically resolved to: {dataset_path}")

    # Prepare data utilizing the loaded alphabet (ocr.letters)
    print("Preparing dataset generator...")
    ocr.prepare(dataset_path, verbose=1, num_workers=4)

    model = ocr.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def eval_fn(current_model):
        current_model.eval()
        current_model.to(device)
        correct = 0
        total = len(ocr.dm.val_image_generator)
        if total == 0:
            return 0
            
        with torch.no_grad():
            for idx in range(total):
                img, text = ocr.dm.val_image_generator[idx]
                img = img.unsqueeze(0).to(device)
                logits = current_model(img)
                
                from nomeroff_net.tools.ocr_tools import decode_prediction
                pred_text = decode_prediction(logits.cpu(), ocr.label_converter)
                if pred_text == text:
                    correct += 1
        return correct / total

    def train_fn(current_model):
        current_model.train()
        
        # Ensure the underlying PyTorch model retains these dynamically assigned Python properties 
        # that PyTorch Lightning or Torch-Pruning may strip out during modification/hooking.
        current_model.label_converter = ocr.label_converter
        current_model.letters = ocr.letters
        current_model.max_text_len = ocr.max_text_len
        
        trainer = pl.Trainer(
            max_epochs=args.epochs_finetune,
            accelerator='gpu' if torch.cuda.device_count() else 'cpu',
            devices=1 if torch.cuda.device_count() else None,
            enable_checkpointing=False,
            logger=False
        )
        trainer.fit(current_model, ocr.dm)
        current_model.to(device)
        
    def ignored_layers_builder(m):
        ignored = []
        for name, module in m.named_modules():
            # 1. OCR relies on RNNs/LSTMs heavily. Torch-Pruning currently struggles with tracing
            # dimension changes across `flatten/view` -> RNN operations precisely.
            if isinstance(module, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU)):
                ignored.append(module)
            # 2. Prevent pruning the classification head mapping to sequence lengths
            if isinstance(module, torch.nn.Linear):
                ignored.append(module)
                
        # 3. Target the last Conv2d *safely* explicitly inside the CNN backbone 
        last_conv = None
        if hasattr(m, 'conv_nn'):
            for module in m.conv_nn.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                ignored.append(last_conv)
                
        return ignored

    example_inputs = torch.randn(1, ocr.color_channels, ocr.height, ocr.width).to(device)

    pruner = IterativePruner(
        model=model,
        example_inputs=example_inputs,
        eval_fn=eval_fn,
        train_fn=train_fn,
        max_acc_drop=args.max_drop,
        target_pruning_ratio=args.target_pruning_ratio,
        ignored_layers_builder=ignored_layers_builder,
        save_dir=f'./pruning_ocr_{args.model_name}_results',
        global_pruning=args.global_pruning,
        isomorphic=args.isomorphic
    )

    pruner.run(max_iters=args.max_iters)

if __name__ == "__main__":
    main()
