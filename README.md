<img width="400" src="http://linux.ria.ua/img/articles/numberplate_detection/nomeroff_net.svg" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Nomeroff Net. Automatic numberplate recognition system. Version 0.2.3

## Introduction
Nomeroff Net is a opensource python license plate recognition framework based on the application of a convolutional 
neural network on the [Mask_RCNN](https://github.com/matterport/Mask_RCNN) architecture, and cusomized OCR-module powered by [GRU architecture](https://github.com/ria-com/nomeroff-net/blob/master/docs/OCR.md).

The project is now at the initial stage of development, write to us if you are interested in helping us in the formation of a dataset for your country.

## Installation
Nomeroff Net requires last version of [Mask_RCNN](https://github.com/matterport/Mask_RCNN),  
Python 3.5, 3.6 or 3.7 (if you plan to install the latest tensorflow >=1.13.rc2) and [opencv 3.4 or latest](https://opencv.org/) 

```bash
git clone https://github.com/ria-com/nomeroff-net.git
cd ./nomeroff-net
git clone https://github.com/matterport/Mask_RCNN.git
pip3 install -r requirements.txt
```

Download the [latest models](https://nomeroff.net.ua/models/) that are required for your neural network to work and place 
them in the **./models** folder of the nomeroff-net project


## Hello Nomeroff Net

```python
import os
import sys
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')


# change this property
NOMEROFF_NET_DIR = os.path.abspath('../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')

MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.h5")
OPTIONS_MODEL_PATH =  os.path.join(NOMEROFF_NET_DIR, "models/numberplate_options_2019_03_05.h5")

# If you use gpu version tensorflow please change model to gpu version named like *-gpu.pb
mode = "cpu"
OCR_NP_UKR_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ua_12-{}.h5".format(mode))
OCR_NP_EU_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_eu_2-{}.h5".format(mode))
OCR_NP_RU_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ru_3-{}.h5".format(mode))

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector,  OptionsDetector, Detector, textPostprocessing

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

# Load weights in keras format.
nnet.loadModel(MASK_RCNN_MODEL_PATH)

# Initialize rect detector with default configuration file.
rectDetector = RectDetector()

# Initialize text detector.
# Also you may use gpu version models.
textDetector = TextDetector({
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": OCR_NP_UKR_TEXT
    },
    "eu": {
        "for_regions": ["eu", "eu_ua_1995"],
        "model_path": OCR_NP_EU_TEXT
    },
    "ru": {
        "for_regions": ["ru"],
        "model_path": OCR_NP_RU_TEXT
    }
})

# Initialize train detector.
optionsDetector = OptionsDetector()
optionsDetector.load(OPTIONS_MODEL_PATH)

# Detect numberplate
img_path = 'images/example2.jpeg'
img = mpimg.imread(img_path)
NP = nnet.detect([img])

# Generate image mask.
cv_img_masks = filters.cv_img_mask(NP)

# Detect points.
arrPoints = rectDetector.detect(cv_img_masks)
zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

# find standart
# Added a classifier (isHiddenIds) for determining the fact of hide text of number, in order not to recognize a deliberately damaged license plate image.
regionIds, isHiddenIds = optionsDetector.predict(zones)
regionNames = optionsDetector.getRegionLabels(regionIds)

# find text with postprocessing by standart
textArr = textDetector.predict(zones, regionNames)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)
#['RP70012', 'JJF509']
```

## Online Demo
In order to evaluate the quality of work of Nomeroff Net without spending time on setting up and installing, we made an online form in which you can upload your photo and get the [recognition result online](https://nomeroff.net.ua/onlinedemo.html)

## AUTO.RIA Numberplate Dataset
All data on the basis of which the training was conducted is provided by RIA.com. In the following, we will call this data the [AUTO.RIA Numberplate Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2018-11-20.zip).

We will be grateful for your help in the formation and layout of the dataset with the image of the license plates of your country. For markup, we recommend using [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)

## AUTO.RIA Numberplate Options Dataset
The system uses several neural networks. One of them is the classifier of numbers at the post-processing stage. It uses dataset
[AUTO.RIA Numberplate Options Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOptionsDataset-2019-05-15.zip).

The categorizer accurately **(99%)** determines the country and the type of license plate. Please note that now the classifier is configured
mainly for the definition of Ukrainian numbers, for other countries it will be necessary to train the classifier with new data.

## AUTO.RIA Numberplate OCR Datasets
As OCR, we use a [specialized implementation of a neural network with GRU layers](https://github.com/ria-com/nomeroff-net/blob/0.2.0/docs/OCR.md),
for which we have created several datasets:
  * [AUTO.RIA Numberplate OCR UA Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrUa-2019-02-19.zip)
  * [AUTO.RIA Numberplate OCR EU Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrEu-2019-02-19.zip)
  * [AUTO.RIA Numberplate OCR RU Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2019-03-06.zip)
  * [AUTO.RIA Numberplate OCR KZ Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrKz-2019-04-26.zip)

This gives you the opportunity to get **97% accuracy** on photos that are uploaded to [AUTO.RIA](https://auto.ria.com) project

## Road map
For several months now, we have been devoting some of our time to developing new features for the Nomeroff Net project. In the near future we plan:
  * Post a detailed instruction on the training of recognition models and classifier for license plates of your country.
  * To expand the classification of countries of license plates by which to determine the country in which this license plate is registered.

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
