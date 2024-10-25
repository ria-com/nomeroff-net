import os
import sys
import cv2
import glob
import math
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt


# import nomeroff net libs
dir_path = os.path.dirname(os.path.realpath(__file__))
NOMEROFF_NET_DIR = os.path.abspath(os.path.join(dir_path, "../../../"))
sys.path.append(NOMEROFF_NET_DIR)
from nomeroff_net.tools.mcm import modelhub
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.tools.image_processing import distance
from nomeroff_net.tools.via_boxes import VIABoxes
from nomeroff_net.pipes.number_plate_classificators.orientation_detector import OrientationDetector
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor import make_boxes, apply_coefficient
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (normalize_rect_new
                                                                                      as normalize_rect)
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (add_coordinates_offset,
                                                                                      split_numberplate)
from nomeroff_net.tools.mcm import get_device_torch
from tools import NumberplateDatasetItem
from upscaler import HAT
import easyocr

# load models
device_torch = get_device_torch()
reader = easyocr.Reader(['en'])
up = HAT(tile_size=320, num_gpu=int(device_torch == "cuda"))

# load nomeroff net models
classifiactor = OptionsDetector()
_ = classifiactor.load("latest")
model_info = modelhub.download_model_by_name('yolov11x')
model = YOLO(model_info['path'])
# orientation_detector = OrientationDetector()
# orientation_detector.load()


def remove_bad_text_zones(easyocr_arr, exclude_zones_list=None):
    if exclude_zones_list is None:
        exclude_zones_list = ['UA']
    result = []
    for item in easyocr_arr:
        if item[1].upper() not in exclude_zones_list:
            result.append(item)
    return result


def remove_small_zones(easyocr_arr, delete_threshold=0.4):
    result = []
    dy_arr = [{'dy': distance(item[0][1], item[0][2]), 'idx': idx} for idx, item in enumerate(easyocr_arr)]
    max_dy = max(item["dy"] for item in dy_arr)
    dy_arr = filter(lambda x: x["dy"] / max_dy >= delete_threshold, dy_arr)
    dy_idx = [item['idx'] for item in dy_arr]
    for idx, item in enumerate(easyocr_arr):
        if idx in dy_idx:
            result.append(item)
    return result


def append_text_to_line(easyocr_arr, img, count_lines):
    dimensions = {}
    lines = {}
    lines_text = {}
    h, w = img.shape[:2]
    part_y = h / count_lines
    for idx, item in enumerate(easyocr_arr):
        min_x = min(point[0] for point in item[0])
        min_y = min(point[1] for point in item[0])
        max_y = max(point[1] for point in item[0])
        center_y = round(min_y + (max_y - min_y) / 2)
        dimensions[idx] = {'dx': distance(item[0][0], item[0][1]), 'dy': distance(item[0][1], item[0][2]),
                           'center_y': center_y, 'min_x': min_x, 'idx': idx}
        line = math.floor(center_y / part_y)
        if line not in lines:
            lines[line] = []
        lines[line].append(dimensions[idx])
    for line in lines:
        sorted_arr = sorted(lines[line], key=lambda x: x['min_x'])
        lines_text[line] = ''.join([easyocr_arr[item['idx']][1] for item in sorted_arr])
    return lines_text


def get_easyocr_lines(easyocr_arr, img, count_lines):
    if len(easyocr_arr) > 0:
        cleared_arr = remove_bad_text_zones(easyocr_arr)
        cleared_arr = remove_small_zones(cleared_arr)
        lines_text = append_text_to_line(cleared_arr, img, count_lines)
    else:
        lines_text = {}
    return lines_text


def fix_text_line(_str):
    return _str.replace(" ", "").replace("-", "").replace("|", "I").replace("0", "O")


def fix_number_line(_str):
    return _str.replace(" ", "").replace("-", "").replace("O", "0").replace("I", "1")


def fix_lines(lines, region_id):
    if region_id == 1 or region_id == 2:
        if len(lines) == 2:
            lines[0] = fix_text_line(lines[0])
            lines[1] = fix_number_line(lines[1])
        if len(lines) == 3:
            lines[0] = fix_text_line(lines[0])
            lines[1] = fix_number_line(lines[1])
            lines[2] = fix_text_line(lines[2])
    return lines


def main(img_dir, target_dataset,
         oneline_h=100, oneline_w=400, multiline_h=300, multiline_w=400):
    print("img_dir", img_dir)
    for img_path in glob.glob(img_dir):
        print("====>IMAGE:", img_path)
        # Predict with the model
        results = model(img_path)  # predict on an image

        # Load the image using OpenCV
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        flag_show = False

        # Loop over the results
        for result in results:
            if len(result.boxes):
                # Extract keypoints and bounding boxes
                array_of_keypoints = result.keypoints.cpu().xy
                array_of_boxes = result.boxes.xyxy.cpu()
                for keypoints, bbox in zip(array_of_keypoints, array_of_boxes):
                    print(f"img_w: {img_w} img_h: {img_h}")
                    print("bbox", bbox)
                    if not ((bbox[0] == 0) or (bbox[2] >= img_w - 1)):
                        x_box = int(min(bbox[0], bbox[2]))
                        w_box = int(abs(bbox[2] - bbox[0]))
                        y_box = int(min(bbox[1], bbox[3]))
                        h_box = int(abs(bbox[3] - bbox[1]))

                        # if (w_box < h_box):
                        flag_show = False
                        image_part = img[y_box:y_box + h_box, x_box:x_box + w_box]
                        print("image_part shape", image_part.shape[:2])

                        image_part_upscale = up.run(cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB))
                        if flag_show:
                            plt.imshow(image_part_upscale)
                            plt.show()

                        print('image_part_upscale.shape')
                        print(image_part_upscale.shape[:2])

                        # Calculation of the scaling factor coefficient
                        image_part_h, image_part_w, _ = image_part_upscale.shape
                        coef_h = h_box / image_part_h
                        coef_w = w_box / image_part_w

                        # Тут розвертає не в той  бік
                        # keypoints = normalize_rect(keypoints)

                        localKeypoints = add_coordinates_offset(keypoints, -x_box, -y_box)
                        localKeypoints_upscale = apply_coefficient(localKeypoints, 1 / coef_w, 1 / coef_h)

                        h = oneline_h
                        w = oneline_w

                        # Convert keypoints to numpy array
                        src_points = np.array(localKeypoints_upscale, dtype="float32")

                        # Apply the perspective transformation to the image and define  rotation
                        aligned_img = VIABoxes.get_aligned_image(image_part_upscale, src_points, shift=0, w=w, h=h)
                        # orientation = orientation_detector.predict([aligned_img])[0]
                        # if orientation == 1:  # class=90/270
                        #     rotated_aligned_img = VIABoxes.get_aligned_image(image_part_upscale, src_points, shift=1,
                        #                                                      w=w, h=h)
                        #     new_orientation = orientation_detector.predict(rotated_aligned_img)[0]
                        #     if new_orientation == 0:
                        #         orientation = 1
                        #     elif new_orientation == 1:
                        #         orientation = 2
                        #     elif new_orientation == 2:
                        #         orientation = 3
                        # aligned_img = VIABoxes.get_aligned_image(image_part_upscale, src_points, shift=orientation,
                        #                                          w=w, h=h)

                        region_ids, count_lines, confidences, predicted = classifiactor.predict_with_confidence(
                            [aligned_img])

                        print("classificator", region_ids, count_lines)
                        if count_lines[0] == 2:
                            h = multiline_h
                            w = multiline_w

                            # Apply the perspective transformation to the image
                            aligned_img = VIABoxes.get_aligned_image(image_part_upscale, src_points, #shift=orientation,
                                                                     w=w, h=h)
                        if count_lines[0] == 3:
                            h = multiline_h
                            w = multiline_w

                            # Apply the perspective transformation to the image
                            aligned_img = VIABoxes.get_aligned_image(image_part_upscale, src_points, #shift=orientation,
                                                                     w=w, h=h)

                        # Display the aligned and cropped image
                        result = reader.readtext(aligned_img)

                        easyocr_lines = get_easyocr_lines(result, aligned_img, count_lines[0])
                        if count_lines[0] > len(easyocr_lines):
                            count_lines[0] = len(easyocr_lines)
                            easyocr_lines = get_easyocr_lines(result, aligned_img, count_lines[0])

                        if count_lines[0] > 1:
                            parts = split_numberplate(aligned_img, parts_count=count_lines[0])
                            if flag_show:
                                for a_img_part in parts:
                                    plt.imshow(a_img_part)
                                    plt.show()

                        if len(result) >= count_lines[0] and count_lines[0] > 1:
                            # Make dataset
                            numberplate_dataset_item = NumberplateDatasetItem(target_dataset, img_path, bbox, keypoints,
                                                                              fix_lines(easyocr_lines, region_ids[0]),
                                                                              region_ids[0], image_part,
                                                                              cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))
                            numberplate_dataset_item.write_orig_dataset()
                            numberplate_dataset_item.write_normalize_dataset()

                        make_boxes(aligned_img, [item[0] for item in result], (0, 0, 255))
                        if flag_show:
                            plt.imshow(aligned_img)
                            plt.show()

                        if flag_show:
                            for i, point in enumerate(keypoints):
                                x, y = point
                                # Малюємо точку
                                cv2.circle(img, (int(x), int(y)), int(img.shape[0] / 100), (255, 0, 0), -1)
                                # Виводимо номер точки
                                cv2.putText(img, str(i + 1), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                            (255, 0, 0), 2)

                if flag_show:
                    # Draw bounding box
                    for bbox in array_of_boxes:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        if flag_show:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()


if __name__ == "__main__":
    test_img_dir = "/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/mlines_dataset_via/*"
    test_target_dataset = "/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/mlines_dataset_via_target/"
    main(test_img_dir, test_target_dataset)
