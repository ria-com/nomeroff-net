#!/bin/bash

docker run --rm -it \
			-p 8888:8888 \
			-v ../:/var/www/nomeroff-net \
			nomeroff-net 
