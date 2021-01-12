<img width="400" src="http://linux.ria.ua/img/articles/numberplate_detection/nomeroff_net.svg" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Nomeroff Net. Automatic numberplate recognition system. Version 1.1.0

## Introduction
Nomeroff Net is an opensource python license plate recognition framework based on the application of a segmentation 
neural network and cusomized OCR-module powered by [GRU architecture](https://github.com/ria-com/nomeroff-net/blob/master/docs/OCR.md).

The project is now at the initial stage of development, write to us if you are interested in helping us in the formation of a dataset for your country.

Version 1.0 2.5x faster Nomeroff Net [0.4.x](https://github.com/ria-com/nomeroff-net/tree/v0.4)! This improvement was achieved by replacing [Mask RCNN](https://github.com/matterport/Mask_RCNN) with a [Detectron2](https://github.com/facebookresearch/detectron2) (more modern and high-speed implementation of the instance segmaentation task).
## Installation

### Installation from Source (Linux)

Nomeroff Net requires Python >= 3.6 and [opencv 3.4 or latest](https://opencv.org/) 

Clone Project and clone related projects
```bash
git clone https://github.com/ria-com/nomeroff-net.git
cd nomeroff-net
```

##### For Centos, Fedora and other RedHat-like OS:
```bash
# for Opencv
yum install libSM

# for pycocotools install 
yum install python3-devel 

# ensure that you have installed gcc compiler
yum install gcc
```

##### For Ubuntu and other Debian-like OS:
```bash
# ensure that you have installed gcc compiler
apt-get install gcc

# for opencv install
apt-get install -y libglib2.0
apt-get install -y libsm6
apt-get install -y libfontconfig1 libxrender1
apt-get install -y libxtst6

# for pycocotools install (Check the name of the dev-package for your python3)
apt-get install python3.6-dev
```

##### install python requirments
```bash
pip3 install torch==1.6
pip3 install PyYAML==5.3
pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
pip3 install torchvision>=0.7
pip3 install Cython
pip3 install numpy
pip3 install -r requirements.txt
```

### Installation from Source (Windows)
On Windows, you must have the Visual C++ 2015 build tools on your path. If you don't, make sure to install them from [here](https://go.microsoft.com/fwlink/?LinkId=691126):

<img src="https://github.com/philferriere/cocoapi/raw/master/img/download.png" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Then, run `visualcppbuildtools_full.exe` and select default options:

<img src="https://github.com/philferriere/cocoapi/raw/master/img/install.png" alt="Nomeroff Net. Automatic numberplate recognition system"/>


## Hello Nomeroff Net
```
# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

# Import all necessary libraries.
import numpy as np
import sys
import matplotlib.image as mpimg

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  textPostprocessing

# load models
rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Detect numberplate
img_path = 'images/example2.jpeg'
img = mpimg.imread(img_path)

# Generate image mask.
cv_imgs_masks = nnet.detect_mask([img])
    
for cv_img_masks in cv_imgs_masks:
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)
    
    # cut zones
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)

    # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    
    # find text with postprocessing by standart  
    textArr = textDetector.predict(zones)
    textArr = textPostprocessing(textArr, regionNames)
    print(textArr)
    # ['JJF509', 'RP70012']
```
<br><a href="https://github.com/ria-com/nomeroff-net/blob/master/examples/demo0.ipynb">Hello Jupyter Nomeroff Net</a>

## Online Demo
In order to evaluate the quality of work of Nomeroff Net without spending time on setting up and installing, we made an online form in which you can upload your photo and get the [recognition result online](https://nomeroff.net.ua/onlinedemo.html)

## AUTO.RIA Numberplate Dataset
All data on the basis of which the training was conducted is provided by RIA.com. In the following, we will call this data the [AUTO.RIA Numberplate Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2018-11-20.zip).

We will be grateful for your help in the formation and layout of the dataset with the image of the license plates of your country. For markup, we recommend using [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)

Nomeroff-Net Mask-RCNN Example:
<img src="https://nomeroff.net.ua/images/nn/mrcnn_example.png" alt="Nomeroff-Net Mask-RCNN Example"/>
<br><a href="https://github.com/ria-com/nomeroff-net/blob/master/examples/demo1.ipynb">Mask detection example</a>
<br><a href="https://github.com/ria-com/nomeroff-net/blob/master/examples/demo2.ipynb">Key points detection example</a>

## AUTO.RIA Numberplate Options Dataset
The system uses several neural networks. One of them is the classifier of numbers at the post-processing stage. It uses dataset
[AUTO.RIA Numberplate Options Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2019-05-15.zip).

The categorizer accurately **(99%)** determines the country and the type of license plate. Please note that now the classifier is configured
mainly for the definition of Ukrainian numbers, for other countries it will be necessary to train the classifier with new data.

<img src="https://nomeroff.net.ua/images/nn/clfctr_example.png" alt="Nomeroff-Net OCR Example"/>

## AUTO.RIA Numberplate OCR Datasets
As OCR, we use a [specialized implementation of a neural network with GRU layers](https://github.com/ria-com/nomeroff-net/blob/0.2.0/docs/OCR.md),
for which we have created several datasets:
  * [AUTO.RIA Numberplate OCR UA Dataset (Ukrainian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-2020-07-14.zip)
  * [AUTO.RIA Numberplate OCR UA Dataset (Ukrainian) with old design Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-1995-2021-01-12.zip)
  * [AUTO.RIA Numberplate OCR EU Dataset (European)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2020-10-09.zip)
  * [AUTO.RIA Numberplate OCR RU Dataset (Russian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2020-10-12.zip)
  * [AUTO.RIA Numberplate OCR KZ Dataset (Kazakh)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKz-2019-04-26.zip)
  * [AUTO.RIA Numberplate OCR GE Dataset (Georgian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrGe-2019-07-06.zip)
  * [AUTO.RIA Numberplate OCR BY Dataset (Belarus)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrBy-2020-10-09.zip)
  * [AUTO.RIA Numberplate OCR SU Dataset (exUSSR)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrSu-2020-11-27.zip)
  * [AUTO.RIA Numberplate OCR KG Dataset (Kyrgyzstan)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKg-2020-12-31.zip)  

If we did not manage to update the link on dataset you can find the latest version [here](https://nomeroff.net.ua/datasets/)

This gives you the opportunity to get **98% accuracy** on photos that are uploaded to [AUTO.RIA](https://auto.ria.com) project

<img src="https://nomeroff.net.ua/images/nn/ocr_example.png" alt="Nomeroff-Net OCR Example"/>
<br><a href="https://github.com/ria-com/nomeroff-net/blob/master/examples/demo3.ipynb">Number plate recognition example</a>

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
  * Training on other datasets.
  * Accuracy Improvements.

## Credits
  * Dmytro Probachay &lt;dmytro.probachay@ria.com&gt;
  * Oleg Cherniy &lt;oleg.cherniy@ria.com&gt;

## Links
  * [Nomeroff Net project site](https://nomeroff.net.ua/)
  * [GitHub repository](https://github.com/ria-com/nomeroff-net)
  * [Numberplate recognition. Practical guide. Part 1 (in Russian)](https://habr.com/ru/post/432444/)
  * [Numberplate recognition. As we got 97% accuracy for Ukrainian numbers. Part 2 (in Russian)](https://habr.com/ru/post/439330/)
