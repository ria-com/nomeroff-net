3.4.0 / 2022-11-21
==================
  **updates**
  * Update ocr models
  * Added support yolov8 models for numberpalte detection
  * Update option classification models with efficientnet_v2_s backbone 

3.3.0 / 2022-11-21
==================
  **updates**
  * Load ocr params from modelhub configs 
  * From v3.3 Support multiple ocr cnn backbone
  * Re-train all OCR models with shufflenet_v2_x2_0 backbone


3.2.0 / 2022-06-09
==================
  **updates**
  * Added brand numberplate detection (examples/ju/inference/detect_brand_np.ipynb)
  * Update auto number grab tools (examples/ju/dataset_tools/auto_number_grab.ipynb)
  * Added fake numberplate detector (examples/ju/train/experimental/froud_numberplate_train.ipynb)


3.1.0 / 2022-03-28
==================
  **updates** 
  * Added support for finding 4 number points exclusively within the found bbox
  * Sped up craft postprocessing by cpp bindings
  * Re-train ocr-ua model
  * Re-trained options model
  * Returned to a separate backbone for ocr models
  * Fixed bag with block_cnn in ocr models

3.0.0 / 2022-03-16
==================
  **updates** 
  * Refactored code with Sonarqube 
  * Added Pipelines
  * Restructured code 
  * Added common backbone for ocr models

2.5.0 / 2021-11-24
==================
  **updates** 
  * Replaced custom cnn on resnet in option detector model
  * Added fastapi examle
   

2.4.0 / 2021-11-01
==================
  **updates** 
  * Rewrote OCR to PyTorch  
  * Restructured project folders and files
  * Added autoloading of datasets and dependent repositories
  * Optimized training options and OCR with [PyTorch Lightning](https://www.pytorchlightning.ai/)
  * Added new [dataset tools](https://github.com/ria-com/nomeroff-net/tree/master/tools)
  * Updated [datasets](https://nomeroff.net.ua/datasets/) and [models](https://nomeroff.net.ua/models/)
  * Added experimental feature Orientation Detector
  * Added tensorrt support for OCRs, YOLO and Options Classification models

2.3.0 / 2021-03-11
==================
  **updates** 
  * Optimize multiline to one line algorithm  
  * Have combined multiline to one line algorithm with nomeroff_net API
  * Added tornado and flask examples

2.1.0 / 2021-03-11
==================
  **updates**
  * Removed is filled or not is filled classification
  * Rewritten options classification on torch
  * Added multiline to one line algorithm
  * Added automatic selection of bevel angle options in np_points_craft.detect
  * Added modelhub module

2.0.0 / 2021-03-01
==================
  **updates**
  * Replaced numberplate segmentation and RectDetector module on object detection(yolov5) and craft
  * Added from_MaskRCNN_datafromat_to_YOLO_dataformat.ipynb dataset convertor
  * Increased the number of examples in the dataset of finding license plate zones 
  * Added train example
  * Updated avto-nomer-tool
  * Added ocr eu onnx-convertor 
  * Updated demos .py scripts
  * Updated benchmarks .py scripts
  * Fixed all setup*.py needed
  * Fixed all docker files for new requirements needed
  * Updated .html demo
  * Added faster model for finding license plates for the CPU
  
  **deprecated**
  * DetectronDetector
  * RectDetector
  * MmdetectionDetector

1.0.0 / 2020-08-27
==================
  **updates**
  * Change main version to 1.0.0 beta
  * Updated all examples for new version
  * Fix small bugs in RectDetector
  * Updated all OCR models

0.4.0 / 2020-08-21
==================
  **updates**
  * Updated all code for tensorflow 2.x usage
  * Updated all models for tensorflow 2.x usage
  * Use tensorflow.keras instance keras
  
  **deprecated**
  * MaskRcnn model cut out
  * tensorflow 1.x not supported now 
  
Centermask2 / 2020-07-15
==================
  **training**
  * Added new cpu ua OCR-model with 'KA' combination
  
  **features**
  * Added methods that return OCR probabilities get_acc
  * Added newest pytorch Centermask2 model (3x-faster than MaskRcnn)
  
  **bugfix** 
  * fixed 4 points Detector
  * bug with augmented images fixed

0.3.5 / 2019-07-06
==================
  **model control manager**
  * pip3 install nomeroff-net 

0.3.3 / 2019-07-06
==================
  **model control manager**
  * Added mcm to nomeroff_net 

0.3.1 / 2019-07-06
==================
  **training**
  * Added experimental support for recognition of Georgia (ge) numbers. Recognition Accuracy 97%
  
  **features** 
  * Added latest model autoloader. 

0.3.0 / 2019-06-24
==================
  **training**
  * Re-train mask-rcnn model.
  
  **bugfix** 
  * Fix rounding bug in RectDetect
  
  **tools**
  * Add Mask RCCN dataset tools to auto-nomer-tool

0.2.3 / 2019-05-16
==================
  **features**
  * Added experimental support for recognition of Kazakhstan (kz) 2 line box numbers. Recognition Accuracy 95%.

  **training**
  * Re-train Kazakhstan (kz) numbers recognition model. Get Recognition Accuracy 94%.
  * Re-train options numbers classification model with ["xx_unknown", "eu_ua_2015", "eu_ua_2004", "eu_ua_1995", "eu", "xx_transit", "ru", "kz", "kz_box"] classes output. Get Classification Accuracy 99,9%.
  * Set simplified convolutional network architecture for numberplate classification by default.

0.2.2 / 2019-03-19
==================
  **features**
  * RectDetector: A new perspective distortion correction mechanism has been added, which more accurately positions the number frame. It is activated using the "fixGeometry" parameter,  fixGeometry = true
  * Added experimental support for recognition of Kazakhstan (kz) numbers. Recognition Accuracy 91%

  **training**
  * Added a simplified convolutional network architecture for numberplate classification. To train a simplified model, pass the cnn == "simple" to the train method.

  **bugfix** 
  * Fixed a critical bug in a RectDetector that could lead to python sticking

0.2.1 / 2019-03-07
==================
  **features**
  * Added CPU and GPU docker files.
  * Added ru region detection in license plate classification.
  * Added ocr russian number plate detector.
  
  **training**
  * Update augmentation(use module imgaug).
  * Added freeze model graph and use .pb models in prediction.
 
0.2.0 / 2019-02-21
==================
  **features**
  * OCR: [GRU-network](https://github.com/ria-com/nomeroff-net/blob/master/docs/OCR.md) 
  trained on Ukrainian and European license plates are used instead of tesseract).
  * Implemented batch processing of multiple images.
  * The license plate classification model has been improved. 
  Now, a single pass classification has become possible according to different criteria: 
  by type of the license plate and by characteristic are painted / not painted.
  
  **optimizations**
  * Implemented asynchronous versions of the set of methods, which gives a performance increase of up to 10%.
  * Optimized code for use on Nvidia GPUs.
  
  **training**
  * A small [nodejs admin panel](https://github.com/ria-com/nomeroff-net/blob/master/moderation/README.md) was created, with which you can prepare your dataset 
  for license plate classification or OCR text detection tasks.
  * Prepare example script for [OCR train](https://github.com/ria-com/nomeroff-net/blob/master/train/trainOcrTextDetectorExample.ipynb).
  * Prepare example script for [Options Classification](https://github.com/ria-com/nomeroff-net/blob/master/train/trainOptionDetectorExample.ipynb).
  * Added numberplate [MaskRCNN](https://github.com/ria-com/nomeroff-net/blob/master/train/mrcnnTrainExample.ipynb) example script.

0.1.1 / 2019-01-17
==================

 **features**
 * Add online demo numberplate recognition https://nomeroff.net.ua/onlinedemo.html
 
 
