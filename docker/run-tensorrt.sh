#!/bin/bash

nvidia-docker run --rm -it \
	--privileged --gpus all \
	-p 8904:8904 \
	-v `pwd`/..:/project/nomeroff-net \
	nomeroff-net-trt #jupyter --ip=0.0.0.0 --allow-root --port=8888
