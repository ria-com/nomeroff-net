FROM tensorflow/tensorflow
ENV EBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0

# For Mask_RCNN
RUN apt-get install -y libsm6
RUN apt-get install -y libfontconfig1 libxrender1
RUN apt-get install -y libxtst6
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx

RUN pip3 install torch>=1.6.0
RUN pip3 install torchvision>=0.7.0
RUN pip3 install PyYAML==5.3
RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install tensorflow>=2.*
RUN pip3 install opencv_python
RUN pip3 install imutils
RUN pip3 install "numpy>=1.16.*"
RUN pip3 install jupyter
RUN pip3 install imgaug
RUN pip3 install asyncio
RUN pip3 install GitPython
RUN pip3 install pycocotools
RUN pip3 install tqdm
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /var/www/nomeroff-net
