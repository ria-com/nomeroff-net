#!/bin/bash

nvidia-docker run --rm -it \
			-p 8888:8888 \
			-v `pwd`/..:/var/www/nomeroff-net  -v /home/cache:/home/cache \
			nomeroff-net-trt
