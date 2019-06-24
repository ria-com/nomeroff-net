FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0

# For Mask_RCNN
RUN apt-get install -y libsm6
RUN apt-get install -y libfontconfig1 libxrender1
RUN apt-get install -y libxtst6

RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install tensorflow-gpu
RUN pip3 install opencv_python
RUN pip3 install imutils
RUN pip3 install numpy
RUN pip3 install keras
RUN pip3 install jupyter

WORKDIR /var/www/nomeroff-net
