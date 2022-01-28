<img width="400" src="http://linux.ria.ua/img/articles/numberplate_detection/nomeroff_net.svg" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Nomeroff Net. Automatic numberplate recognition system. Version 2.5

## Introduction
Nomeroff Net is an opensource python license plate 
recognition framework based on YOLOv5 and CRAFT 
networks and customized OCR-module powered by RNN architecture.

The project is now at the stage of development, write to us if you are interested in helping us in the formation of a dataset for your country.

Version 2.5 is completely rewritten to PyTorch.
[Change History](https://github.com/ria-com/nomeroff-net/blob/master/History.md).

## Installation

### Installation from Source (Linux)

Nomeroff Net requires Python >= 3.6 

Clone Project
```bash
git clone https://github.com/ria-com/nomeroff-net.git
cd nomeroff-net
```

### For Centos, Fedora and other RedHat-like OS:
```bash
# for Opencv
yum install libSM

# for pycocotools install 
yum install python3-devel 

# ensure that you have installed gcc compiler
yum install gcc

yum install git

# Before "yum install ..." download https://libjpeg-turbo.org/pmwiki/uploads/Downloads/libjpeg-turbo.repo to /etc/yum.repos.d/
yum install libjpeg-turbo-official
```

install requirements:
```bash
pip3 install -r requirements.txt 
```

### For Ubuntu and other Debian-like OS:
```bash
# ensure that you have installed gcc compiler
apt-get install gcc

# for opencv install
apt-get install -y libglib2.0
apt-get install -y libgl1-mesa-glx

# for pycocotools install (Check the name of the dev-package for your python3)
apt-get install python3.6-dev

# other packages
apt-get install -y git
apt-get install -y libturbojpeg
```

install requirements:
```bash
pip3 install -r requirements.txt 
```

## Hello Nomeroff Net
```python
# Specify device
import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector

from NomeroffNet.TextDetectors.eu import eu
from NomeroffNet import textPostprocessing

# load models
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = eu()
textDetector.load("latest")

# Detect numberplate
img_path = 'images/example2.jpeg'
img = cv2.imread(img_path)
img = img[..., ::-1]

targetBoxes = detector.detect_bbox(img)
all_points = npPointsCraft.detect(img, targetBoxes,[5,2,0])

# cut zones
zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

# predict zones attributes 
regionIds, countLines = optionsDetector.predict(zones)
regionNames = optionsDetector.getRegionLabels(regionIds)

# find text with postprocessing by standart
textArr = textDetector.predict(zones)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)
# ['JJF509', 'RP70012']
```

## Hello Nomeroff Net for systems with a small GPU size.
Note: This example disables some important Nomeroff Net features. It will recognize numbers that are photographed in a horizontal position.
```python
import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import all necessary libraries.
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from NomeroffNet.TextDetectors.eu import eu
from NomeroffNet import textPostprocessing

textDetector = eu()
textDetector.load("latest")

# Detect numberplate
img_path = 'images/example2.jpeg'
img = cv2.imread(img_path)
img = img[..., ::-1]

targetBoxes = detector.detect_bbox(img)

zones = []
regionNames = []
for targetBox in targetBoxes:
    x = int(min(targetBox[0], targetBox[2]))
    w = int(abs(targetBox[2]-targetBox[0]))
    y = int(min(targetBox[1], targetBox[3]))
    h = int(abs(targetBox[3]-targetBox[1]))
    
    image_part = img[y:y + h, x:x + w]
    zones.append(image_part)
    regionNames.append('eu')
    
# find text with postprocessing by standart
textArr = textDetector.predict(zones)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)
# ['RP70012', 'JJF509']
```


<br><a href="https://github.com/ria-com/nomeroff-net/tree/master/examples">More Examples</a>

## Online Demo
In order to evaluate the quality of work of Nomeroff Net without spending time on setting up and installing, we made an online form in which you can upload your photo and get the [recognition result online](https://nomeroff.net.ua/onlinedemo.html)

## AUTO.RIA Numberplate Dataset
All data on the basis of which the training was conducted is provided by RIA.com. 
In the following, we will call this data the [AUTO.RIA Numberplate Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2021-07-21.zip).

We will be grateful for your help in the formation and layout of the dataset with the image of the license plates of your country. For markup, we recommend using [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)

Dataset Example:
<img src="https://github.com/ria-com/nomeroff-net/raw/master/public/images/segment_example.png" alt="Nomeroff-Net Segment Example"/>

## AUTO.RIA Numberplate Options Dataset
The system uses several neural networks. One of them is the classifier of numbers at the post-processing stage. It uses dataset
[AUTO.RIA Numberplate Options Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2021-09-03.zip).

The categorizer accurately **(99%)** determines the country and the type of license plate. Please note that now the classifier is configured
mainly for the definition of Ukrainian numbers, for other countries it will be necessary to train the classifier with new data.

<img src="https://nomeroff.net.ua/images/nn/clfctr_example.png" alt="Nomeroff-Net OCR Example"/>

## AUTO.RIA Numberplate OCR Datasets
As OCR, we use a specialized implementation of a neural network with RNN layers,
for which we have created several datasets:
  * [AUTO.RIA Numberplate OCR UA Dataset (Ukrainian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-2021-08-25.zip)
  * [AUTO.RIA Numberplate OCR UA Dataset (Ukrainian) with old design Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-1995-2021-09-03.zip)
  * [AUTO.RIA Numberplate OCR EU Dataset (European)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2021-09-02.zip)
  * [AUTO.RIA Numberplate OCR RU Dataset (Russian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip)
  * [AUTO.RIA Numberplate OCR KZ Dataset (Kazakh)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKz-2019-04-26.zip)
  * [AUTO.RIA Numberplate OCR GE Dataset (Georgian)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrGe-2019-07-06.zip)
  * [AUTO.RIA Numberplate OCR BY Dataset (Belarus)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrBy-2021-08-27.zip)
  * [AUTO.RIA Numberplate OCR SU Dataset (exUSSR)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrSu-2021-09-03.zip)
  * [AUTO.RIA Numberplate OCR KG Dataset (Kyrgyzstan)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKg-2020-12-31.zip)
  * [AUTO.RIA Numberplate OCR AM Dataset (Armenia)](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrAm-2021-05-20-all-draft.zip)

If we did not manage to update the link on dataset you can find the latest version 
[here](https://nomeroff.net.ua/datasets/)

This gives you the opportunity to get **99% accuracy**on photos that are uploaded to 
[AUTO.RIA](https://auto.ria.com) project

<img src="https://nomeroff.net.ua/images/nn/ocr_example.png" alt="Nomeroff-Net OCR Example"/>

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
