#!/bin/bash

docker run --rm -it \
	-p 8904:8904 \
	-v `pwd`/..:/project/nomeroff-net \
	nomeroff-net-cpu bash
