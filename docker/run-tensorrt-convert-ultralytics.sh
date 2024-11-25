#!/bin/bash

nvidia-docker run --rm -it \
			--privileged --gpus all \
			-p 8904:8904 \
			-v `pwd`/..:/var/www/nomeroff-net  -v /home/cache:/home/cache \
			nomeroff-net-trt:convert-ultralytics #jupyter notebook --ip=0.0.0.0 --allow-root --port=8888