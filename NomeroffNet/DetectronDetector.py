import os
import time
import sys
import glob
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.engine import launch
from detectron2.engine import default_argument_parser
from detectron2.engine import default_setup 
from detectron2.config import CfgNode
from detectron2.data.datasets import register_coco_instances

class Detector:
    def __init__(self, 
                 name        = "numberplate_train",
                 json_file = "./datasets/numberplate/train/coco_numberplate.json",
                 image_root  = "./datasets/numberplate/train",
                 name_val        = "numberplate_val",
                 json_file_val   = "./datasets/numberplate/val/coco_numberplate.json",
                 image_root_val  = "./datasets/numberplate/val"
                ):
        # registr own dataset
        try:
            register_coco_instances(name, {}, json_file, image_root)
            register_coco_instances(name_val, {}, json_file_val, image_root_val)
        except Exception as e:
            # TODO: show warning
            pass
        
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def loadModel(self, centermask2_path = "../centermask2",
                  config_file='../NomeroffNet/configs/centermask2/numberplates/numberplate_V_39_eSE_FPN_ms_3x.yaml'):
        """
        Create configs and perform basic setups.
        TODO: create folder config/centermask2/ and put all architecture them
        """
        sys.path.append(centermask2_path)
        from centermask.config import get_cfg

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def detect_mask(self, images, verbose = 0):
        """
        TODO: multi gpu instances runtime
        """
        outputs_cpu = []
        for im in images:
            outputs = self.predictor(im)
            output_cpu = outputs["instances"].to("cpu")
            masks = np.array(output_cpu.get_fields()["pred_masks"])
            masks = [cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2RGB) for mask in masks]
            # if mask and np.all((arr == 0))
            if len(masks):
                outputs_cpu.append(masks)
        return outputs_cpu