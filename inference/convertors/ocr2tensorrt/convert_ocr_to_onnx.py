import sys
import os
import tensorflow as tf

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "../../../"))

from NomeroffNet.TextDetector import TextDetector

# get models
# Initialize text detector.
textDetector = TextDetector({
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu": {
        "for_regions": ["eu"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
        "model_path": "latest"
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    },
    "su": {
        "for_regions": ["su"],
        "model_path": "latest"
    }
})


model_repository_dir = "./model_repository"
for detector, name in zip(textDetector.detectors, textDetector.detectors_names):
    temp_model_file = os.path.join(model_repository_dir, f"{name}_text_detector", "1", "model.savedmodel")
    tf.saved_model.save(detector.MODEL, temp_model_file)
