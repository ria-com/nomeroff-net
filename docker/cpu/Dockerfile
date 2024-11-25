# Використання образу Python 3.12
FROM python:3.12-slim

# Налаштування таймзони
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Kiev

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Оновлення та встановлення необхідних системних залежностей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    manpages-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Оновлення pip
RUN python3 -m pip install --upgrade pip

# Робоча директорія для проекту
WORKDIR /project

# Встановлення залежностей із pip
RUN python3 -m pip install cython setuptools numpy opencv_python scikit_image asyncio gitpython pycocotools ujson pillow tqdm matplotlib scipy seaborn ipywidgets gevent termcolor scikit-learn albumentations

# Встановлення Torch і пов'язаних пакетів (CPU-версія)
RUN python3 -m pip install torch>=1.12 torchvision>=0.13 --index-url https://download.pytorch.org/whl/cpu

# Встановлення Ultralytics
RUN python3 -m pip install "ultralytics>=8.3.12" "pytorch_lightning==1.8.6"

# Встановлення залежностей із репозиторіїв git
RUN python3 -m pip install "git+https://github.com/ria-com/upscaler.git"
RUN python3 -m pip install "git+https://github.com/ria-com/craft-text-detector.git"
RUN python3 -m pip install "git+https://github.com/lilohuang/PyTurboJPEG.git"
RUN python3 -m pip install "git+https://github.com/ria-com/modelhub-client.git"

# Робоча директорія для проекту
WORKDIR /project/nomeroff-net
