#!/bin/bash

nvidia-docker run --rm -it \
			-p 8888:8888 \
			--privileged --gpus all \
			-v `pwd`/..:/var/www/nomeroff-net  -v /home/cache:/home/cache \
			nomeroff-net-trt
