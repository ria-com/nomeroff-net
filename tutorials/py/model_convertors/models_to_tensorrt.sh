#!/bin/bash
: '
cd ./tutorials/py/model_convertors/
./models_to_tensorrt.sh
'
DEVICE="${1:-0}"
echo 'DEVICE' ${DEVICE}

# download and convert yolov8 model
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# download and convert yolov5 model
python3 convert_ultralytics_to_tensorrt.py -m yolov11x
python3 convert_ultralytics_to_tensorrt.py -m yolov11x_brand_np

# convert numberplate options pytorch model to trt and save into model_repository
python3 convert_numberplate_options_to_tensorrt.py

# conver text detection pytorch model to trt and save into model_repository
python3 ./convert_ocr_to_tensorrt.py -d eu_ua_2004_2015_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d eu_ua_1995_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d eu_ua_custom_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d xx_transit_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d eu_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d ru
python3 ./convert_ocr_to_tensorrt.py -d kz
python3 ./convert_ocr_to_tensorrt.py -d kg
python3 ./convert_ocr_to_tensorrt.py -d ge
python3 ./convert_ocr_to_tensorrt.py -d su_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d am
python3 ./convert_ocr_to_tensorrt.py -d by
python3 ./convert_ocr_to_tensorrt.py -d eu_2lines_efficientnet_b2
python3 ./convert_ocr_to_tensorrt.py -d su_2lines_efficientnet_b2
