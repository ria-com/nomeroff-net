#!/bin/bash

# convert numberplate options pytorch model to onnx and save into model_repository
python3 convert_numberplate_options_to_onnx.py

# download and convert yolov5 model
wget -O model.pt https://nomeroff.net.ua/models/object_detection/yolov5s-2021-05-14.pt
python3 ../../yolov5/export.py --weights model.pt --img 640 --batch 1  --include='onnx' --dynamic
mkdir -p model_repository/yolov5/1
mv model.onnx model_repository/yolov5/1/model.onnx
rm model.pt

# conver craft to onnx
python3 convert_numberplate_options_to_onnx.py

# conver text detection keras models to tf saved models
conver_text_detection_keras_models_to_tf_saved_models.py
