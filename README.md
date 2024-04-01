# Nomeroff Net
<img width="50%" src="https://github.com/ria-com/nomeroff-net/raw/master/public/images/nomeroff_net.svg" alt="Nomeroff Net. Automatic numberplate recognition system"/>

Nomeroff Net. Automatic numberplate recognition system. Version 3.5.0
<br /><br />
<blockquote style="border-left-color: #ff0000">
Now there is a war going on in our country, russian soldiers are shooting at civilians in Ukraine. Enemy aviation launches rockets and drops bombs on residential quarters.
<br>
We are deeply thankful for the unprecedented wave of support for Ukraine from around the world. Below is a list of funds that help the Ukrainian army in the fight against Russian invaders:
<ul>
<li><a href="https://savelife.in.ua/en/">Come back alive</a></li>
<li><a href="https://prytulafoundation.org/en/home/support_page">Serhiy Prytula Charity Foundation</a></li>
</ul>

<img src="https://github.com/ria-com/nomeroff-net/raw/master/public/images/Shelling_of_civilians_in_Ukraine.jpeg?raw=true" alt="Russian troops shelling of civilians in Ukraine. Donbass. Kostyantynivka. 09.07.2022"/>
Photo: Konstantin Liberov https://www.instagram.com/libkos/
</blockquote>

## Introduction
Nomeroff Net is an opensource python license plate 
recognition framework based on YOLOv5 and CRAFT 
networks and customized OCR-module powered by RNN architecture.

Write to us if you are interested in helping us in the formation of a dataset for your country.

[Change History](https://github.com/ria-com/nomeroff-net/blob/master/History.md).

## Installation

### Installation from Source (Linux)

Nomeroff Net requires Python >= 3.7

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
apt-get install python3.7-dev

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
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", 
                                              image_loader="opencv")

(images, images_bboxs, 
 images_points, images_zones, region_ids, 
 region_names, count_lines, 
 confidences, texts) = unzip(number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', 
                                                                 './data/examples/oneline_images/example2.jpeg']))
 
print(texts)
```

## Hello Nomeroff Net for systems with a small GPU size.
Note: This example disables some important Nomeroff Net features. It will recognize numbers that are photographed in a horizontal position.

```python
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

number_plate_detection_and_reading = pipeline("number_plate_short_detection_and_reading", 
                                              image_loader="opencv")

(images, images_bboxs,
 zones, texts) = unzip(number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', 
                                                           './data/examples/oneline_images/example2.jpeg']))
 
print(texts)
# (['AC4921CB'], ['RP70012', 'JJF509'])
```


<br><a href="https://github.com/ria-com/nomeroff-net/tree/master/examples">More Examples</a>


## Nomeroff Net Professional

If you don't want to install and configure the Nomeroff Net programmed code for your own tasks or if your client hardware does not have enough resources to run a service that requires ML computing, you can use our commercial [API Nomeroff Net Professional](https://ai.ria.com/ANPR-NomeroffNetProfessional), which allows you to perform recognition remotely on the [RIA.com Сompany](https://ria.company) servers.

The Nomeroff Net Professional API is based on the open source Nomeroff Net engine, with commercial modifications aimed mainly at using improved models that can produce better results in photos with poor image quality.

Right now you can try [ALPR/ANPR Nomeroff Net Professional Demo](https://ai.ria.com/ANPR-NomeroffNetProfessional#ANPR-demo) for free.

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
  * [VIN OCR](https://ai.ria.com/VIN-OCR)
  * [ANPR/ALPR Nomeroff Net Professional](https://ai.ria.com/ANPR-NomeroffNetProfessional)
