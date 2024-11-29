# How we have modernised Nomeroff Net over the past 3 years and what awaits us in version 4

## Problems we worked on
Despite the ongoing war in Ukraine, work on the Nomeroff Net project continues unabated and we are pleased to present you version 4.0.

We decided to tackle a number of problems that have not been progressed for a long time and even worsened from version 3.0 to 3.5:

### Multi-line numberplate detection
In previous versions, our strategy was to detect areas with inscriptions, stick them together and then recognise them with an OCR model trained on single-line number plates. This was not a very good solution, as the quality of recognition of multi-line number plates was rather mediocre.

| Original                                                                                                                                                | Examples of recognition                                                                                                                                 |  
|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="images/2lines-detection/src/10014678-10014678__NO-FAST__Morgan-Threewheeler.jpg" width="600px" alt="Morgan Threewheeler picture example"/>    | <img src="images/2lines-detection/10014678-10014678__NO-FAST__Morgan-Threewheeler.png" width="600px" alt='Numberplate detection example "NO-FAST"'/>    |
| <img src="images/2lines-detection/src/10234617-10234617__BE0886AA.jpg" width="600px" alt="Motorcycle picture example"/>                                 | <img src="images/2lines-detection/10234617-10234617__BE0886AA.png" width="600px" alt='Numberplate detection example "BE0886AA"'/>                       | 
| <img src="images/2lines-detection/src/11337620-11337620__VT769CJ__Yamaha-XT.jpg" width="600px" alt="Yamaha XT picture example"/>                        | <img src="images/2lines-detection/11337620-11337620__VT769CJ__Yamaha-XT.png" width="600px" alt='Numberplate detection example "VT769CJ"'/>              |
| <img src="images/2lines-detection/src/25558318-25558318__BT7181AF__Royal-Enfield-Meteor.jpg" width="600px" alt="Royal Enfield Meteor picture example"/> | <img src="images/2lines-detection/25558318-25558318__BT7181AF__Royal-Enfield-Meteor.png" width="600px" alt='Numberplate detection example "BT7181AF"'/> |
| <img src="images/2lines-detection/src/25758757-25758757__DU912JT__Dino-206-246-GT.jpg" width="600px" alt="Dino 206-246 GT picture example"/>            | <img src="images/2lines-detection/25758757-25758757__DU912JT__Dino-206-246-GT.png" width="600px" alt='Numberplate detection example "DU912JT"'/>        |
| <img src="images/2lines-detection/src/9977744-9977744__BRAT31__Malaguti-Spidermax.jpg" width="600px" alt="Malaguti Spidermax picture example"/>         | <img src="images/2lines-detection/9977744-9977744__BRAT31__Malaguti-Spidermax.png" width="600px" alt='Numberplate detection example "BRAT31"'/>         |
| <img src="images/2lines-detection/src/9996069-9996069__MDA934__Honda-Civic.jpg" width="600px" alt="Honda Civic picture example"/>                       | <img src="images/2lines-detection/9996069-9996069__MDA934__Honda-Civic.png" width="600px" alt='Numberplate detection example "MDA934"'/>                |


In version 4, we abandoned the use of a common model for single-line and multi-line numbers and trained a specialised Yolo 11 POSE model that allows us to get better number frames and highlight lines with text without using the CRAFT module. 

### Memory leak occurs when using nomeroff net as a service for a long time.
Previously, we had to periodically (once a day) reboot the instance with Nomeroff Net running, as it began to gradually increase the video memory consumption on our GPUs, we attribute this to the implementation features of the CRAFT module, which helped us to conduct additional detection of lines with text on the found bundling boxes with number plates. Now there is no need for this additional detection, so we have eliminated the use of CRAFT.

### The use of different types of OCR.
In previous versions, we tried to create a specific OCR for each country, adapted to the specifics of the number plates of that country, and also took into account the list of letters used in that country. (For example, Ukrainian number plates do not use the letters Q, W, V, etc.) This approach led to dozens of different OCRs that take up a lot of memory space and Nomeroff Net slowly turned into a ‘big monster’ running on a limited number of computers with expensive GPUs.
Now we have come up with the concept that we need to use universal models and make corrections in the post-processing of the number plate text.
There will be one universal model for all number plates written in Latin, another Cyrillic universal model for all number plates using, for example, Cyrillic, and theoretically, Arabic, Chinese, etc. can be trained.
If necessary, you can leave a specialised OCR model or a model trained on your dataset (instead of the universal one) for a given country. What you use is up to you.
Here you can see an example of setting up a custom list of models for your needs ( 
https://github.com/ria-com/nomeroff-net/blob/v4.0/tutorials/ju/inference/get-started-demo-ocr-custom.ipynb), the list of ‘default’ models can be found in this code snippet, they are set in the DEFAULT_PRESETS variable
https://github.com/ria-com/nomeroff-net/blob/e03d093e15bb62eea0a36ef1aaa25bd1b31e41d2/nomeroff_net/pipelines/number_plate_text_reading.py 

The list of available models can be found in this code snippet, the list of available models is entered into the model_config_urls variable (models with OCR are marked with the #OCR comment):
https://github.com/ria-com/nomeroff-net/blob/v4.0/nomeroff_net/tools/mcm.py 

### Low resolution of the recognised number
There are cases when the camera is located at a great distance from the object of recognition or the photo on which the number plate is being searched has a low resolution. In version 4.0, it is now possible to make an upscale of the area with the number plate, which will be further read by the OCR model. This can improve the recognition result, but will require additional resources to apply [HUT or DRCT](https://github.com/ria-com/upscaler.git) models.

| A fragment of a photo before recognition                                                                                                           | Fragment enlarged 4 times using the HAT-L_SRx4_ImageNet-pretrain model                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="images/upscaling/low_resolution_numberplate_example15.png" width="200px" alt='Low resolution example for numberplate "AI1382HB"'/> | <img src="images/upscaling/low_resolution_numberplate_example15xHAT-L_SRx4_ImageNet-pretrain.png" alt='Upsacling by HAT-L_SRx4_ImageNet-pretrain image for numberplate "AI1382HB"'/> |
| <img src="images/upscaling/low_resolution_numberplate_example4.png" width="200px" alt='Low resolution example for numberplate "AC4249CB"'/>  | <img src="images/upscaling/low_resolution_numberplate_example4xHAT-L_SRx4_ImageNet-pretrain.png" alt='Upsacling by HAT-L_SRx4_ImageNet-pretrain image for numberplate  "AC4249CB"'/> |
| <img src="images/upscaling/low_resolution_numberplate_example9.png" width="200px" alt='Low resolution example for numberplate "AB3391AK"'/>  | <img src="images/upscaling/low_resolution_numberplate_example9xHAT-L_SRx4_ImageNet-pretrain.png" alt='Upsacling by HAT-L_SRx4_ImageNet-pretrain image for numberplate  "AB3391AK"'/> |

### Detection speed

In Nomeroff Net 4.0, we used the new YOLO v11 model, which is faster and more accurate than the previous YOLO v8, and we are also working on simplified models that can be used with limited resources.
Usually, we use yolo11x-pose for maximum quality of detection, but you can also connect yolo11n-pose, which will also give a good result. Similar simplifications will be developed for OCR models in the future.

### Easy installation via PIP
The version 4 module is easy to install using pip:

```pip install nomeroff-net```

## What were the results?
Also, due to the elimination of the CRAFT module from the project, the model inference is about 9% faster than the previous version. On average, the detection speed is about 100 ms (this figure can vary significantly depending on the size of the photo to be processed, the GPU used for recognition, network speed, system hard drive, and many other factors)

## Nomeroff Net as a service
If you are interested in using the licence plate recognition module or VIN recognition as a service, this is now possible with the release of the updated version. 
We have started providing our clients with access to [RIA.com's commercial ML services] via API (https://ai.ria.com). 
You can use the service right now:
  * [Numplate RIAder](https://ai.ria.com/en/numplate-riader) - ALPR/ANPR system for number plate recognition
  * [VIN RIAder](https://ai.ria.com/en/vin-riader) -  VIN code recognition system


