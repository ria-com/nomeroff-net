# nVidia TensorRT Base Image
ARG TRT_CONTAINER_VERSION=23.01
FROM nvcr.io/nvidia/pytorch:${TRT_CONTAINER_VERSION}-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0
RUN apt-get install -y git
RUN apt install libgl1-mesa-glx -y

# turbojpeg
RUN apt-get install -y libturbojpeg

RUN pip install --upgrade pip
RUN pip install pillow==8.0.1
RUN pip install setuptools
RUN pip install "PyYAML>=5.3"
RUN pip install "numpy==1.23.5"
RUN pip install scikit_image
RUN pip install Cython
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install gevent

RUN pip install opencv_python
RUN pip install "imgaug>=0.4.*"
RUN pip install pillow
RUN pip install scipy
RUN pip install asyncio
RUN pip install GitPython
RUN pip install gevent
RUN pip install pycocotools
RUN pip install tqdm
RUN pip install termcolor
RUN pip install ujson
RUN pip install "pytorch_lightning==1.8.6"
RUN python3 -m pip install "ultralytics==8.0.45"
RUN pip install ipywidgets
RUN pip install setuptools
RUN pip install -U "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN pip install -U "git+https://github.com/ria-com/modelhub-client.git"

RUN pip install pycuda

WORKDIR /var/www/nomeroff-net
