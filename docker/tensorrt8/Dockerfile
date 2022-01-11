FROM nvcr.io/nvidia/tensorrt:21.11-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update
RUN apt remove -y python3-pip
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

#ARG PYTHON_VERSION="3.8.10"
#ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
#ARG ONNXRUNTIME_BRANCH=master
#ARG CMAKE_CUDA_ARCHITECTURES=37;50;52;60;61;70;75;80

#RUN apt-get update &&\
#    apt-get install -y sudo git bash unattended-upgrades
#RUN unattended-upgrade

#WORKDIR /code
#ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.21.0-linux-x86_64/bin:/opt/miniconda/bin:${PATH}

## Prepare onnxruntime repository & build onnxruntime with TensorRT
#RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
#    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
#    cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
#    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
#    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
#    cd onnxruntime &&\
#    /bin/sh ./build.sh --parallel --skip_submodule_sync --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) '"CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'"' &&\
#    python3.8 -m pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
#    cd ..

# For opencv
RUN apt-get install -y libglib2.0
RUN apt-get install -y git
RUN apt update
RUN apt install libgl1-mesa-glx -y

# turbojpeg
RUN apt-get install -y libturbojpeg

RUN python3.8 -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install setuptools
RUN python3.8 -m pip install "PyYAML>=5.3"
RUN python3.8 -m pip install scikit_image
RUN python3.8 -m pip install Cython
RUN python3.8 -m pip install pycocotools
RUN python3.8 -m pip install matplotlib
RUN python3.8 -m pip install seaborn

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

RUN python3.8 -m pip install "numpy>=1.16.*"
RUN python3.8 -m pip install imgaug
RUN python3.8 -m pip install "pillow>=8.2.0"
RUN python3.8 -m pip install scipy
RUN python3.8 -m pip install asyncio
RUN python3.8 -m pip install GitPython
RUN python3.8 -m pip install gevent
RUN python3.8 -m pip install pycocotools
RUN python3.8 -m pip install tqdm
RUN python3.8 -m pip install termcolor
RUN python3.8 -m pip install ujson
RUN python3.8 -m pip install pytorch_lightning
RUN python3.8 -m pip install ipywidgets
RUN python3.8 -m pip install -U "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN python3.8 -m pip install -U "git+https://github.com/ria-com/modelhub-client.git"

#RUN python3.8 -m pip install jupyter
#RUN apt-get install -y mc

WORKDIR /var/www/nomeroff-net
