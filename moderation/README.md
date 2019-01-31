# Nomeroff Net. OCR moderation admin panel.


## Quick start
  * Make copy of ./config/default.js.example to ./config/default.js
  * In cofig change value of moderation.regionOCRModeration.base_dir
  * Install requirements:
  ```bash
    npm install
  ```
  * Run server:
  ```bash
    node server
  ```
  
  
## Structure of OCR dataset directory:
   Directory must contain 2 directories:
   * ann - the directory in which the annotations to the images should be located. 
   * img - the directory in which the images should be located.
   Annotation format:
   ```json
    {
      "description": "AA0006IT", 
      "tags": [], 
      "name": "AA0006IT", 
      "objects": [], 
      "size": {"width": 156, "height": 34}
    }
```
      