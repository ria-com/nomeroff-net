<img width="400" src="http://linux.ria.ua/img/articles/numberplate_detection/nomeroff_net.svg" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Nomeroff Net. Automatic numberplate recognition system.


## Introduction
Nomeroff Net is a opensource python license plate recognition framework based on the application of a convolutional 
neural network on the [Mask_RCNN](https://github.com/matterport/Mask_RCNN) architecture, as well as the 
[tesseract](https://github.com/tesseract-ocr/tesseract) text recognition system.

The project is now at the initial stage of development, write to us if you are interested in helping us in the formation of a dataset for your country.


## Installation

Nomeroff Net requires last version of [Mask_RCNN](https://github.com/matterport/Mask_RCNN), [tesseract 4.xx](https://github.com/tesseract-ocr/tesseract), 
python 3.6 and [opencv 3.4 for python3](https://opencv.org/) 

```
$ git clone https://github.com/matterport/Mask_RCNN.git
$ git clone https://github.com/ria-com/nomeroff-net.git
```
To install tesseract 4, please follow instruction from [tesseract wiki](https://github.com/tesseract-ocr/tesseract/wiki)


## Hello Nomeroff Net

```python
import os
import sys
import json
import matplotlib.image as mpimg

# change this property
NOMEROFF_NET_DIR = "/var/www/nomeroff-net/"
MASK_RCNN_DIR = "/var/www/Mask_RCNN/"

MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, "logs/")
MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.h5")
REGION_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/imagenet_vgg16_np_region_2019_1_18.h5")

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector, RegionDetector, Detector, textPostprocessing

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
# Load weights in keras format.
nnet.loadModel(MASK_RCNN_MODEL_PATH)

# Initialize rect detector with default configuration file.
rectDetector = RectDetector()

# Initialize text detector.
textDetector = TextDetector()

# Initialize numberplate region detector.
regionDetector = RegionDetector()
regionDetector.load(REGION_MODEL_PATH)

img_path = './examples/images/example1.jpeg'
img = mpimg.imread(img_path)
NP = nnet.detect([img])

# Generate image mask.
cv_img_masks = filters.cv_img_mask(NP)

for img_mask in cv_img_masks:
    # Detect points.
    points = rectDetector.detect(img_mask, fixRectangleAngle=1, outboundWidthOffset=3)

    # Split on zones
    zone = rectDetector.get_cv_zones(img, points)

    # find standart
    regionId = regionDetector.predict(zone)
    regionName = regionDetector.getLabels(regionId)

    # find text with postprocessing by numberplate region detector
    text = textDetector.detect(zone)
    text = textPostprocessing(text, regionName)
    print('Detected numberplate: "%s" in region [%s]'%(text,regionName))
    # Detected numberplate: "AC4921CB" in region [eu-ua-2015]
```

## Online Demo
In order to evaluate the quality of work of Nomeroff Net without spending time on setting up and installing, we made an online form in which you can upload your photo and get the [recognition result online](https://nomeroff.net.ua/onlinedemo.html)

## AUTO.RIA Numberplate Dataset
All data on the basis of which the training was conducted is provided by RIA.com. In the following, we will call this data the [AUTO.RIA Numberplate Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2018-11-20.zip).

We will be grateful for your help in the formation and layout of the dataset with the image of the license plates of your country. For markup, we recommend using [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)

## AUTO.RIA Numberplate Country Dataset
The system uses several neural networks. One of them is the classifier of numbers at the post-processing stage. It uses dataset
[AUTO.RIA Numberplate Country Dataset](https://nomeroff.net.ua/datasets/autoriaNPCountryDataset-2019-01-14.zip).

The categorizer accurately (96%) determines the country and the type of license plate. Please note that now the classifier is configured
mainly for the definition of Ukrainian numbers, for other countries it will be necessary to train the classifier with new data.</p>

## Road map
For several months now, we have been devoting some of our time to developing new features for the Nomeroff Net project. In the near future we plan:
  * Post a detailed instruction on the training of recognition models and classifier for license plates of your country.
  * To expand the classification of countries of license plates by which to determine the country in which this license plate is registered.
  * Add a classifier for determining the fact of drawing a number, in order not to recognize a deliberately damaged license plate image.


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
  * [Numberplate recognition. Practical guide. Part 1 (in Russian)](https://habr.com/post/432444/)
