FROM nvcr.io/nvidia/tensorrt:20.11-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install --upgrade pip

RUN apt-get update --fix-missing

# For opencv
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0
RUN apt-get install -y libsm6
RUN apt-get install -y libfontconfig1 libxrender1
RUN apt-get install -y libxtst6
RUN apt-get install -y git

RUN mkdir -p /data

# clear cache
RUN apt-get clean
RUN pip3 install --no-cache-dir setupextras
RUN pip3 install --no-cache-dir opencv-python
RUN pip3 install --no-cache-dir torch
RUN pip3 install --no-cache-dir torchvision

WORKDIR /var/www/nomeroff-net
COPY . /var/www/nomeroff-net


WORKDIR /var/www/nomeroff-net/examples/tensorrt
