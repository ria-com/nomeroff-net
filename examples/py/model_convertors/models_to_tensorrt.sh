#!/bin/bash
: '
cd ./examples/py/model_convertors/
./models_to_tensorrt.sh
'

# download and convert yolov5 model
python3 convert_yolo_to_tensorrt.py

# convert numberplate options pytorch model to trt and save into model_repository
python3 convert_numberplate_options_to_tensorrt.py

# conver text detection pytorch model to trt and save into model_repository
python3 ./convert_ocr_to_tensorrt.py -d eu_ua_2004_2015
python3 ./convert_ocr_to_tensorrt.py -d eu_ua_1995
python3 ./convert_ocr_to_tensorrt.py -d eu
python3 ./convert_ocr_to_tensorrt.py -d ru
python3 ./convert_ocr_to_tensorrt.py -d kz
python3 ./convert_ocr_to_tensorrt.py -d kg
python3 ./convert_ocr_to_tensorrt.py -d ge
python3 ./convert_ocr_to_tensorrt.py -d su
python3 ./convert_ocr_to_tensorrt.py -d am
python3 ./convert_ocr_to_tensorrt.py -d by
