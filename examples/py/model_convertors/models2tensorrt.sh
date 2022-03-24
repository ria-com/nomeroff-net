#!/bin/bash
: '
cd ./examples/py/model_convertors/
./models2tensorrt.sh
'

# convert numberplate options pytorch model to onnx and save into model_repository
python3 convert_numberplate_options_to_onnx.py

# download and convert yolov5 model
./yolo2tensorrt/bin/yolov5_tensorrt.sh

# conver craft to onnx
python3 convert_numberplate_options_to_onnx.py

# conver text detection keras models to tf saved models
python3 conver_text_detection_keras_models_to_tf_saved_models.py
