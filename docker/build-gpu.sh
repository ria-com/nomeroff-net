#!/bin/bash

DOCKER_BUILDKIT=1 docker build -t nomeroff-net-gpu -f ./gpu/Dockerfile ..
