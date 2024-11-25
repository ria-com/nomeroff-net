#!/bin/bash

DOCKER_BUILDKIT=1 docker build -t nomeroff-net-trt -f ./tensorrt/Dockerfile ..
