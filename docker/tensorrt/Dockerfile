FROM nvcr.io/nvidia/tensorrt:20.12-py3
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

RUN pip3 install "torch>=1.8"
RUN pip3 install "torchvision>=0.9"
RUN pip3 install setuptools
RUN pip3 install "PyYAML>=5.3"
RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install opencv_python
RUN pip3 install "numpy>=1.16.*"
RUN pip3 install imgaug
RUN pip3 install pillow
RUN pip3 install scipy
RUN pip3 install asyncio
RUN pip3 install GitPython
RUN pip3 install pycocotools
RUN pip3 install tqdm
RUN pip3 install pytorch_lightning
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git"
RUN pip3 install -U "git+https://github.com/ria-com/modelhub-client.git"

WORKDIR /var/www/nomeroff-net
