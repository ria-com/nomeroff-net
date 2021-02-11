import os
import time
import sys
import glob
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmcv import Config


class Detector:
    """
    Firstly install mmdetector
    pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .
    cd ../
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip3 install -r requirements/build.txt
    pip3 install -v -e .
    """
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def loadModel(self,
                 cfg_path = 'faster_rcnn_r50_caffe_fpn_mstrain_1x_numberpale.py',
                 checkpoint_file = './numberplate_exps/epoch_12.pth',
                 devie='cuda:0'):
        cfg = cfg = Config.fromfile(cfg_path)
        self.model = init_detector(cfg, checkpoint_file, device='cuda:0')

    def detect_bbox(self, images, min_bbox_acc=0.5):
        """
        TODO: multi gpu instances runtime
        """
        result = inference_detector(self.model, images)
        result =[[box for box in image_bbox[0] if box[-1] > min_bbox_acc] for image_bbox in result]
        return result