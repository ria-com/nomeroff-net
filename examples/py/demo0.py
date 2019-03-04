# Import all necessary libraries.
import os
import sys
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')

MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.pb")
OPTIONS_MODEL_PATH =  os.path.join(NOMEROFF_NET_DIR, "models/numberplate_options_2019_2_15.pb")

# If you use gpu version tensorflow please change model to gpu version named like *-gpu.pb
OCR_NP_UKR_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ua_1_2_11-cpu.pb")
OCR_NP_EU_TEXT =  os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_eu_2-cpu.pb")

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector,  OptionsDetector, Detector, textPostprocessing

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

# Load weights in keras format.
nnet.loadModel(MASK_RCNN_MODEL_PATH)

# Initialize rect detector with default configuration file.
rectDetector = RectDetector()

# Initialize text detector.
# You may use gpu version modeks.
textDetector = TextDetector({
    "eu_ua_2014_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": OCR_NP_UKR_TEXT
    },
    "eu": {
        "for_regions": ["eu", "eu_ua_1995"],
        "model_path": OCR_NP_EU_TEXT
    }
})

# Initialize train detector.
optionsDetector = OptionsDetector()
optionsDetector.load(OPTIONS_MODEL_PATH)

# Detect numberplate
img_path = '../images/example2.jpeg'
img = mpimg.imread(img_path)
NP = nnet.detect([img])


# Generate image mask.
cv_img_masks = filters.cv_img_mask(NP)

# Detect points.
arrPoints = rectDetector.detect(cv_img_masks)
zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

# find standart
regionIds, stateIds = optionsDetector.predict(zones)
regionNames = optionsDetector.getRegionLabels(regionIds)

# find text with postprocessing by standart
textArr = textDetector.predict(zones, regionNames)
textArr = textPostprocessing(textArr, regionNames)
print(textArr)

