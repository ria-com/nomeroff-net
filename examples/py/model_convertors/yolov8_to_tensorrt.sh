#!/bin/bash
: '
cd ./examples/py/model_convertors/
./yolov8_to_tensorrt.sh 0
'

DEVICE="${1:-0}"
echo 'DEVICE' ${DEVICE}

# download and convert yolov8 model
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="${DEVICE}" python3 convert_yolo_v8_to_tensorrt.py -d ${DEVICE}
