import os
import sys
import cv2
import numpy as np
import random
import torch
import detectron2

# import some common detectron2 utilities
#from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))
from mcm.mcm import get_mode_torch as get_mode

def thresh_callback(src_gray, threshold=256/2):
    src_gray = cv2.blur(src_gray, (3,3))
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # Find the convex hull object for each contour
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.fillConvexPoly(drawing, hull, (255, 255, 255))
    return drawing

class Detector:
    def __init__(self):
        pass

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def loadModel(self, nomeroffnet_path = "../",
                  model_weight = "",
                  subdir="./NomeroffNet/configs/detectron2/numberplates/",
                  config_file='numberplate2_R_50_FPN_3x.yaml'):
        """
        Create configs and perform basic setups.
        TODO: create folder config/centermask2/ and put all architecture them
        """
        if get_mode() == "cpu":
            config_file = f"cpu_{config_file}"
        config_file = os.path.join(nomeroffnet_path, subdir, config_file)
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        # path to the model we just trained
        if model_weight:
            cfg.MODEL.WEIGHTS = model_weight
        # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
        
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def detect_mask(self, images, verbose = 0, convex_hull=1):
        """
        TODO: multi gpu instances runtime
        """
        outputs_cpu = []
        for im in images:
            outputs = self.predictor(im)
            output_cpu = outputs["instances"].to("cpu")
            masks = np.array(output_cpu.get_fields()["pred_masks"])

            if convex_hull:
                masks = [thresh_callback((mask*255).astype(np.uint8)) for mask in masks]
            else:
                masks = [cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2RGB) for mask in masks]

            #for mask in masks:
            #    print("222")
            #    print(mask.shape)
            # if mask and np.all((arr == 0))
            if len(masks):
                outputs_cpu.append(masks)
        return outputs_cpu