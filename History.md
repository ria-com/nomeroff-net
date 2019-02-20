 0.2.0 / 2019-01-17
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

0.1.1 / 2019-01-17
==================

 **features**
 * Add online demo numberplate recognition https://nomeroff.net.ua/onlinedemo.html
 
 