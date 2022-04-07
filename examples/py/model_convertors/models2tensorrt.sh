#!/bin/bash
: '
cd ./examples/py/model_convertors/
./models2tensorrt.sh
'

# download and convert yolov5 model
./yolo2tensorrt/bin/yolov5_tensorrt.sh

# convert numberplate options pytorch model to onnx and save into model_repository
cd options2tensorrt
python3 convert_numberplate_options_to_onnx.py
cd ../

# conver text detection keras models to tf saved models
cd ocr2tensorrt
python3 convert_ocr_to_onnx.py
cd ../
