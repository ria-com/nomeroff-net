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
        self.model = init_detector(cfg, checkpoint_file, device=devie)

    def detect_bbox(self, images, min_bbox_acc=0.5, max_img_w=800):
        """
        TODO: multi gpu instances runtime
        """
#         sizes = []
#         resized_images = []
#         for img in images:
#             corect size for better speed"
#             img_w = img.shape[1]
#             img_h = img.shape[0]
#             img_w_r = 1
#             img_h_r = 1
#             resized_img = img
#             if img_w > max_img_w:
#                 resized_img = cv2.resize(img, (max_img_w, int(max_img_w/img_w*img_h)))
#                 img_w_r = img_w/max_img_w
#                 img_h_r = img_h/(max_img_w/img_w*img_h)
#                 sizes.append([img_w_r,img_h_r])
#             else:
#                 sizes.append([1,1])
#             resized_images.append(resized_img)
                    
        result = inference_detector(self.model, images)
        result =[[box for box in image_bbox[0] if box[-1] > min_bbox_acc] for image_bbox in result]
#         result =[[[box[0]*s[0], box[1]*s[1], box[2]*s[0], box[3]*s[1], box[4]] for box in image_bbox] for image_bbox, s in zip(result, sizes)]
        return result