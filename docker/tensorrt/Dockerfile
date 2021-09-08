FROM nvcr.io/nvidia/tensorrt:21.08-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx

# turbojpeg
RUN apt-get install -y libturbojpeg

#RUN pip3 install "torch>=1.9"
#RUN pip3 install "torchvision>=0.9"
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install setuptools
RUN pip3 install "PyYAML>=5.3"
RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install seaborn

#RUN pip3 install opencv_python
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip opencv.zip; unzip opencv_contrib.zip
RUN mkdir -p build && cd build; cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master; make -j8; make install

RUN pip3 install "numpy>=1.16.*"
RUN pip3 install imgaug
RUN pip3 install "pillow>=8.2.0"
RUN pip3 install scipy
RUN pip3 install asyncio
RUN pip3 install GitPython
RUN pip3 install pycocotools
RUN pip3 install tqdm
RUN pip3 install pytorch_lightning
RUN pip3 install ipywidgets
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git"
RUN pip3 install -U "git+https://github.com/ria-com/modelhub-client.git"

WORKDIR /var/www/nomeroff-net
