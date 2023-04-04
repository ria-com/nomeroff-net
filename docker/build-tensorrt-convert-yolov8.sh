#!/bin/bash

docker build -t nomeroff-net-trt:convert-yolov8 -f ./tensorrt/Dockerfile_convert_yolov8 .
