#!/bin/bash

nvidia-docker run --rm -it \
			--privileged --gpus all \
			-p 8888:8888 \
			-v `pwd`/..:/var/www/nomeroff-net  -v /home/cache:/home/cache \
			nomeroff-net-trt #jupyter --ip=0.0.0.0 --allow-root --port=8888
