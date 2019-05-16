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
 
 
