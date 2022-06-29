# Nomeroff Net OCR moderation admin panel.


## Quick start
  * Make copy of ./config/restExample.js
  * In config change value of moderation.regionOCRModeration.base_dir
  * Install requirements:
  ```bash
    npm install
  ```
  * Run server:
  ```bash
    NODE_ENV=restExample node server
  ```
  * Locate to:
  ```url
    http://localhost:5005/OCRM.html
  ```
  
  
## Structure of OCR dataset directory:
   Directory must contain 2 directories:
   * ann - the directory in which the annotations to the images should be located. 
   * img - the directory in which the images should be located.
   Annotation format:
   ```json
    {
      "description": "001AAA10", 
      "name": "001AAA10",  
      "region_id": 7, 
      "state_id": 2, 
      "size": {
        "width": 149, 
        "height": 32,
      }, 
      "moderation": {
        "isModerated": 1, 
        "moderatedBy": "ApelSYN",
        "predicted": "00AAAA10"
      }
    }
   ```

## Hot key
To start, focus on one of the license plates by clicking on it.
   * Numpad8 - Take the license plate and move down.
   * Numpad2 - Take the license plate and move up.
   * NumpadDecimal - Delete license plate and move down.
      