# Nomeroff Net OCR moderation admin panel.


## Quick start
  * Go to the `moderation` directory and install requirements:
  ```bash
    cd moderation
    npm install
  ```
  * Make a copy of `./config/restExample.js`, e.g. `./config/my.js`
    (the file name is the value of `NODE_ENV`).
    Out of the box `restExample.js` points to the example dataset
    `data/dataset/TextDetector/ocr_example/train`, so you can start with it as is.
  * In your config set the path to your dataset in
    `moderation.regionOCRModeration.base_dir` (absolute path is recommended).
  * Run server from the `moderation` directory:
  ```bash
    NODE_ENV=my node server
  ```
  * Locate to:
  ```url
    http://localhost:5005/OCRM.html
  ```

### Config sections
  * `moderation.regionOCRModeration.base_dir` - path to the dataset directory (required).
  * `moderation.regionOCRModeration.options` - lists of `region_id`, `state_id`, `count_lines` values.
  * `moderation.template.moderation.moderatedBy` - your name, it is saved into moderated annotations.
  * `dataset.*` - names and extensions of the dataset subdirectories.
  * `koa_view`, `pages` - templates of the keypoints editor, keep them as in `restExample.js`.
  * `moderation.VIABoxPointsModeration` - optional, only for the VIA box points moderation
    (`VIAM.html`), see `config/example.js`.


## Structure of OCR dataset directory:
   ### Simple dataset
   Directory must contain 2 directories:
   * ann - the directory in which the annotations to the images should be located. 
   * img - the directory in which the images should be located.

   ### Multi-line dataset
   Dataset made by `nomeroff_net/tools/nn_numberplate_dataset.py` additionally contains:
   * anb - the photo descriptions (`<photo_id>.json`: source photo and regions with lines).
   * box - the cropped number plate boxes (shown in the left column, click opens the keypoints editor).
   * src - the source photos.

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
      