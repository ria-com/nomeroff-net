# nVidia TensorRT Base Image
ARG TRT_VERSION=21.07
FROM nvcr.io/nvidia/tensorrt:${TRT_VERSION}-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=v1.10.0
ARG CMAKE_CUDA_ARCHITECTURES=37;50;52;60;61;70;75;80

RUN apt-get install -y sudo git bash unattended-upgrades
RUN unattended-upgrade

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.21.0-linux-x86_64/bin:/opt/miniconda/bin:${PATH}

# Prepare onnxruntime repository & build onnxruntime with TensorRT
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
    cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --parallel --skip_submodule_sync --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_d
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd ..

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

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip opencv.zip; unzip opencv_contrib.zip
RUN mkdir -p build && cd build;\
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master; make -j8;\
    make install

RUN rm /var/www/opencv.zip;\
    rm /var/www/opencv_contrib.zip;\
    rm -rf /var/www/opencv-master;\
    rm -rf /var/www/opencv_contrib-master;\
    rm -rf /var/www/build

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
