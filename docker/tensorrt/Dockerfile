# nVidia TensorRT Base Image
ARG TRT_CONTAINER_VERSION=21.12
FROM nvcr.io/nvidia/tensorrt:${TRT_CONTAINER_VERSION}-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0
RUN apt-get install -y git
RUN apt update
RUN apt install libgl1-mesa-glx -y

# turbojpeg
RUN apt-get install -y libturbojpeg

RUN pip install --upgrade pip
RUN pip install pillow==8.0.1
RUN pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install setuptools
RUN pip install "PyYAML>=5.3"
RUN pip install "numpy>=1.16.*"
RUN pip install scikit_image
RUN pip install Cython
RUN pip install pycocotools
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install gevent

RUN pip install 'opencv_python<4.5.5'
RUN pip install "numpy>=1.16.*"
RUN pip install imgaug
RUN pip install "pillow>=8.2.0"
RUN pip install scipy
RUN pip install asyncio
RUN pip install GitPython
RUN pip install gevent
RUN pip install pycocotools
RUN pip install tqdm
RUN pip install termcolor
RUN pip install ujson
RUN pip install pytorch_lightning
RUN pip install ipywidgets
RUN pip install -U "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN pip install -U "git+https://github.com/ria-com/modelhub-client.git"

RUN pip install jupyter
RUN apt-get install -y mc

WORKDIR /var/www/nomeroff-net
