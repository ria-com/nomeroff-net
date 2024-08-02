import os
import cv2
import numpy as np
from nomeroff_net.tools.via import VIADataset
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect_new
from .image_processing import reshape_points

class VIABoxes:
    def __init__(self,
                 dataset_json,
                 verbose=False):
        self.dataset_json = dataset_json
        self.dataset_dir = os.path.dirname(dataset_json)
        self.via_dateset = VIADataset(label_type="radio", verbose=verbose)
        self.via_dateset.load_from_via_file(dataset_json)
        self.debug = verbose

    @staticmethod
    def get_keypoints(region):
        all_points_x = region["shape_attributes"]["all_points_x"]
        all_points_y = region["shape_attributes"]["all_points_y"]
        keypoints = []
        for x, y in zip(all_points_x, all_points_y):
            keypoints.append((x, y))
        return keypoints

    @staticmethod
    def get_bbox_basename(prefix, bbox):
        return f'{prefix}-{bbox[0]}x{bbox[1]}-{bbox[2]}x{bbox[3]}'

    @staticmethod
    def get_bbox_filename(prefix, bbox):
        return f'{VIABoxes.get_bbox_basename(prefix, bbox)}.png'

    @staticmethod
    def get_aligned_image(image, keypoints, shift = 0, w = 300, h = 100):
        if shift>0:
            keypoints = reshape_points(keypoints, shift)

        src_points = np.array(keypoints, dtype="float32")

        target_points = np.float32(np.array([[0, h], [0, 0], [w, 0], [w, h]]))
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, target_points)

        # Apply the perspective transformation to the image
        aligned_img = cv2.warpPerspective(image, M, (w, h))

        return aligned_img

    def make_transformed_boxes(self, target_dir= './', shift = 0, check_dir=None):
        for key in self.via_dateset.data['_via_img_metadata']:
            item = self.via_dateset.data['_via_img_metadata'][key]
            print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                for region in item["regions"]:
                    if region["shape_attributes"]["name"] == "polygon" and ("label" not in region["region_attributes"] or region["region_attributes"]["label"] == "numberplate"):
                        keypoints = VIABoxes.get_keypoints(region)
                        keypoints_norm = normalize_rect_new(keypoints, self.debug)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints_norm]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints_norm]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints_norm]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints_norm]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        box_w = max_x_box-min_x_box
                        box_h = max_y_box-min_y_box
                        if box_h >= 20 and box_w >= 60:
                                bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)
                                if shift == 0 or (shift != 0 and os.path.exists(os.path.join(check_dir, bbox_filename))):
                                    bbox_path = os.path.join(target_dir, bbox_filename)
                                    bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift)
                                    cv2.imwrite(bbox_path, bbox_image)
                                    print(f'    region "{bbox_filename}" done')





