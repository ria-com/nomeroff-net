FROM python:3.9-slim
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

RUN apt-get install -y build-essential
RUN apt-get install -y manpages-dev

# For opencv
RUN apt-get install -y libglib2.0

# For pip modules
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx

RUN python3 -m pip install setuptools
RUN python3 -m pip install "PyYAML>=5.3"
RUN python3 -m pip install scikit_image
RUN python3 -m pip install Cython
RUN python3 -m pip install matplotlib
RUN python3 -m pip install seaborn
RUN python3 -m pip install opencv_python
RUN python3 -m pip install "numpy==1.23.5"
RUN python3 -m pip install "imgaug>=0.4.*"
RUN python3 -m pip install pillow
RUN python3 -m pip install scipy
RUN python3 -m pip install gevent
RUN python3 -m pip install asyncio
RUN python3 -m pip install termcolor
RUN python3 -m pip install ujson
RUN python3 -m pip install GitPython
RUN python3 -m pip install tqdm
RUN python3 -m pip install sklearn
RUN python3 -m pip install "torch>=1.14.0" "torchvision>=0.13.0" --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install "ultralytics==8.0.45"
RUN python3 -m pip install "pytorch_lightning==1.8.6"
RUN python3 -m pip install -U "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN python3 -m pip install -U "git+https://github.com/ria-com/modelhub-client.git"

WORKDIR /var/www/nomeroff-net
