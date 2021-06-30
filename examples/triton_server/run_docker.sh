nvidia-docker run --gpus=all \
       --rm \
       --ipc=host \
       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
       --mount type=bind,source=/var/www/nomeroff-net/examples/triton_server/model_repository,target=/models \
       nvcr.io/nvidia/tritonserver:20.11-py3 \
       tritonserver \
       --model-repository=/models 