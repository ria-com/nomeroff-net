import os
import json
import cv2
import numpy as np
from typing import List

from nomeroff_net.tools.image_processing import generate_image_rotation_variants
from nomeroff_net.data_loaders import ImgOrientationGenerator
from .numberplate_options_data_module import OptionsNetDataModule


class OrientationDataModule(OptionsNetDataModule):
    def __init__(self,
                 train_dir="../datasets/mask/train",
                 train_json_path="../datasets/mask/train/via_region_data_orientation.json",
                 validation_dir="../datasets/mask/val",
                 validation_json_path="../datasets/mask/val/via_region_data_orientation.json",
                 width=300,
                 height=300,
                 angles: List = None,
                 batch_size=32,
                 num_workers=0,
                 with_aug=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width = width
        self.height = height
        if angles is None:
            angles = [0, 90, 180, 270]
        
        self.train = None
        self.train_image_generator = ImgOrientationGenerator(
            train_json_path,
            train_dir,
            img_w=width,
            img_h=height,
            batch_size=batch_size,
            angles=angles,
            with_aug=with_aug)
        
        self.val = None
        self.val_image_generator = ImgOrientationGenerator(
            validation_json_path,
            validation_dir,
            img_w=width,
            img_h=height,
            batch_size=batch_size,
            angles=angles,
            with_aug=with_aug)
        
        self.test = None
        self.test_image_generator = ImgOrientationGenerator(
            validation_json_path,
            validation_dir,
            img_w=width,
            img_h=height,
            batch_size=batch_size,
            angles=angles,
            with_aug=with_aug)


def show_data(img_path,
              json_path,
              max_count_image=1):
    from matplotlib import pyplot as plt

    print("Loading dataset...")
    with open(json_path) as json_file:
        data = json.load(json_file)
    for i, p in enumerate(data['_via_img_metadata']):
        item = data['_via_img_metadata'][p]
        file_name = item["file_name"]
        image_path = os.path.join(img_path, file_name)
        img = cv2.imread(image_path)
        target_boxes = [
            [
                min(np.array(region['shape_attributes']['all_points_x'])),
                min(np.array(region['shape_attributes']['all_points_y'])),
                max(np.array(region['shape_attributes']['all_points_x'])),
                max(np.array(region['shape_attributes']['all_points_y'])),
            ] for region in item['regions']
            if len(region['shape_attributes']['all_points_x']) == 4
            and len(region['shape_attributes']['all_points_y']) == 4]
        variant_images, variants_bboxes = generate_image_rotation_variants(img,
                                                                           target_boxes,
                                                                           angles=[90, 180, 270])
        angles = [0, 90, 180, 270]
        for variant_image, variant_bboxes, angle in zip(variant_images, variants_bboxes, angles):
            for bbox in variant_bboxes:
                min_x = bbox[0]
                max_x = bbox[2]
                min_y = bbox[1]
                max_y = bbox[3]
                img_part = variant_image[min_y:max_y, min_x:max_x]

                print("[INFO]", file_name, angle, bbox, variant_image.shape)
                plt.imshow(img_part)
                plt.show()
        if i + 1 > max_count_image:
            break
