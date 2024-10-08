import os
import cv2
import numpy as np
from tqdm import tqdm
from nomeroff_net.tools.via import VIADataset
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect_new
from nomeroff_net.pipes.number_plate_classificators.orientation_detector import OrientationDetector
from .image_processing import reshape_points

from nomeroff_net.tools.image_processing import (fline,
                                                 distance,
                                                 linear_line_matrix,
                                                 find_distances,
                                                 fix_clockwise2,
                                                 find_min_x_idx,
                                                 detect_intersection,
                                                 reshape_points)

def check_blurriness(image: np.ndarray, threshold=5.0) -> bool:
    # Переводимо зображення в градації сірого в залежності від кількості каналів
    if len(image.shape) == 3:  # 3 канали (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:  # 4 канали (RGB + Альфа)
        # Ігноруємо альфа-канал і конвертуємо тільки RGB в градації сірого
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Обчислюємо лапласіан (другу похідну) зображення
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # Перевіряємо, чи менше варіація від порогу
    return laplacian_var < threshold

def find_first_point_index(rect, point):
    for i, item in enumerate(rect):
        if item[0] == point[0] and item[1] == point[1]:
            return i
    return None



def check_clockwise_rect(rect):
    rect_clockwise = fix_clockwise2(rect)
    idx = find_first_point_index(rect_clockwise, rect[0])
    rect_clockwise = reshape_points(rect_clockwise, idx)
    for i, item in enumerate(rect):
        if item[0] == rect_clockwise[i][0] and item[1] == rect_clockwise[i][1]:
            pass
        else:
            return False
    return True

def check_normalized_rect(rect):
    rect_norm = normalize_rect_new(rect)
    for i, item in enumerate(rect):
        if item[0] == rect_norm[i][0] and item[1] == rect_norm[i][1]:
            pass
        else:
            return False
    return True

class VIABoxes:
    def __init__(self,
                 dataset_json,
                 verbose=False):
        self.dataset_json = dataset_json
        self.dataset_dir = os.path.dirname(dataset_json)
        self.via_dateset = VIADataset(label_type="radio", verbose=verbose)
        self.via_dateset.load_from_via_file(dataset_json)
        self.debug = verbose
        # self.numberplate_orientation_0_180 = OrientationDetector(classes={'0': 0, '180': 1})
        # self.numberplate_orientation_0_180.load("modelhub://numberplate_orientation_0_180")
        #
        # self.numberplate_orientation_0_180__90_270 = OrientationDetector(classes={'0-180': 0, '90-270': 1})
        # self.numberplate_orientation_0_180__90_270.load("modelhub://numberplate_orientation_0-180_90-270")

    @staticmethod
    def get_keypoints(region):
        all_points_x = region["shape_attributes"]["all_points_x"]
        all_points_y = region["shape_attributes"]["all_points_y"]
        keypoints = []
        for x, y in zip(all_points_x, all_points_y):
            keypoints.append((x, y))
        return keypoints

    @staticmethod
    def set_keypoints(region, keypoints):
        all_points_x = []
        all_points_y = []
        for keypoint in keypoints:
            all_points_x.append(round(keypoint[0]))
            all_points_y.append(round(keypoint[1]))
        region["shape_attributes"]["all_points_x"] = all_points_x
        region["shape_attributes"]["all_points_y"] = all_points_y
        return region

    @staticmethod
    def get_bbox_basename(prefix, bbox):
        return f'{prefix}-{bbox[0]}x{bbox[1]}-{bbox[2]}x{bbox[3]}'

    @staticmethod
    def get_bbox_filename(prefix, bbox):
        return f'{VIABoxes.get_bbox_basename(prefix, bbox)}.png'

    @staticmethod
    def get_aligned_image(image, keypoints, shift=0, w=300, h=100):
        if shift > 0:
            keypoints = reshape_points(keypoints, shift)

        src_points = np.array(keypoints, dtype="float32")

        target_points = np.float32(np.array([[0, h], [0, 0], [w, 0], [w, h]]))
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, target_points)

        # Apply the perspective transformation to the image
        aligned_img = cv2.warpPerspective(image, M, (w, h))

        return aligned_img


    def fix_denormalized_via_boxes(self, target_dir='./',
                               flag_dir=None,
                               filtered_classes=None,
                               w=300, h=100, min_h=20, min_w=60
                               ):
        fix_denormalized = False
        if filtered_classes is None:
            filtered_classes = ["numberplate"]
        for key in tqdm(self.via_dateset.data['_via_img_metadata'], desc="Processing"):
            item = self.via_dateset.data['_via_img_metadata'][key]
            #print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                regions = []
                for region in item["regions"]:
                    #target_boxes_path = os.path.join(target_dir, os.path.basename(filename))
                    #cv2.imwrite(moderation_img_path, image)

                    if (region["shape_attributes"]["name"] == "polygon"
                            and region["region_attributes"]["label"] in filtered_classes):
                        #print(region["region_attributes"]["label"])
                        keypoints = VIABoxes.get_keypoints(region)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        box_w = max_x_box-min_x_box
                        box_h = max_y_box-min_y_box
                        bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)
                        bbox_path = os.path.join(target_dir, bbox_filename)
                        if flag_dir is None or os.path.exists(os.path.join(flag_dir, bbox_filename)):
                            #if not check_clockwise_rect(keypoints):
                            if not check_normalized_rect(keypoints):
                                keypoints = normalize_rect_new(keypoints)
                                print(f'Fixed normalized points for file {bbox_filename}')
                                VIABoxes.set_keypoints(region, keypoints)
                                fix_denormalized = True
                                region["shape_attributes"]["checked"] = False
                                bbox_image = VIABoxes.get_aligned_image(image, keypoints, shift=0, w=w, h=h)
                                cv2.imwrite(bbox_path, bbox_image)
        if fix_denormalized:
            self.via_dateset.write_via(self.dataset_json)

    def fix_checked_via_boxes(self, checked_dir=None):
        fix_via = False
        if checked_dir is None:
            return

        for key in tqdm(self.via_dateset.data['_via_img_metadata'], desc="Processing"):
            item = self.via_dateset.data['_via_img_metadata'][key]
            #print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                for region in item["regions"]:
                    if (region["shape_attributes"]["name"] == "polygon"):
                        keypoints = VIABoxes.get_keypoints(region)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)
                        bbox_path_checked = os.path.join(checked_dir, bbox_filename)
                        if os.path.exists(bbox_path_checked):
                            region["shape_attributes"]["checked"] = True
                            fix_via = True
        if fix_via:
            self.via_dateset.write_via(self.dataset_json)


    def fix_clockwise_via_boxes(self, target_dir='./',
                               flag_dir=None,
                               filtered_classes=None,
                               w=300, h=100, min_h=20, min_w=60
                               ):
        fix_clockwise = False
        if filtered_classes is None:
            filtered_classes = ["numberplate"]
        for key in tqdm(self.via_dateset.data['_via_img_metadata'], desc="Processing"):
            item = self.via_dateset.data['_via_img_metadata'][key]
            #print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                regions = []
                for region in item["regions"]:
                    #target_boxes_path = os.path.join(target_dir, os.path.basename(filename))
                    #cv2.imwrite(moderation_img_path, image)

                    if (region["shape_attributes"]["name"] == "polygon"
                            and region["region_attributes"]["label"] in filtered_classes):
                        #print(region["region_attributes"]["label"])
                        keypoints = VIABoxes.get_keypoints(region)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        box_w = max_x_box-min_x_box
                        box_h = max_y_box-min_y_box
                        bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)
                        bbox_path = os.path.join(target_dir, bbox_filename)
                        if flag_dir is None or os.path.exists(os.path.join(flag_dir, bbox_filename)):
                            if not check_clockwise_rect(keypoints):
                                keypoints = normalize_rect_new(keypoints)
                                print(f'Fixed clockwise for file {bbox_filename}')
                                VIABoxes.set_keypoints(region, keypoints)
                                fix_clockwise = True
                                bbox_image = VIABoxes.get_aligned_image(image, keypoints, shift=0, w=w, h=h)
                                cv2.imwrite(bbox_path, bbox_image)
        if fix_clockwise:
            self.via_dateset.write_via(self.dataset_json)
            pass


    def make_cropped_via_boxes(self, target_dir='./',
                               target_file=None,
                               filtered_classes=None,
                               w=300, h=100, min_h=20, min_w=60
                               ):
        if filtered_classes is None:
            filtered_classes = ["numberplate"]
        items = self.via_dateset.data['_via_img_metadata']
        if target_file is not None:
            items = {}
            for key in self.via_dateset.data['_via_img_metadata']:
                item = self.via_dateset.data['_via_img_metadata'][key]
                if item['filename'] == target_file:
                    items[key] = item
                    break

        for key in tqdm(items, desc="Processing"):
            item = self.via_dateset.data['_via_img_metadata'][key]
            #tqdm.set_description(f"Processing (custom: {custom_param})")
            #tqdm.set_postfix(custom_param=item['filename'])
            #print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                regions = []
                for region in item["regions"]:
                    #target_boxes_path = os.path.join(target_dir, os.path.basename(filename))
                    #cv2.imwrite(moderation_img_path, image)

                    if (region["shape_attributes"]["name"] == "polygon"
                            and region["region_attributes"]["label"] in filtered_classes):
                        #print(region["region_attributes"]["label"])
                        keypoints = VIABoxes.get_keypoints(region)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        box_w = max_x_box-min_x_box
                        box_h = max_y_box-min_y_box
                        bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)

                        bbox_image = VIABoxes.get_aligned_image(image, keypoints, shift=0, w=w, h=h)

                        bbox_path = os.path.join(target_dir, bbox_filename)
                        cv2.imwrite(bbox_path, bbox_image)


    def make_transformed_boxes(self, target_dir='./',
                               moderation_bbox_dir=None,
                               moderation_image_dir=None,
                               wrong_shift_strategy="rotate",  # may be: "rotate", "moderate"
                               filtered_classes=None,
                               w=300, h=100, min_h=20, min_w=60,
                               ):
        if filtered_classes is None:
            filtered_classes = ["numberplate"]

        # new_via_data = {
        #     "_via_settings": self.via_dateset.data['_via_settings'],
        #     "_via_attributes": self.via_dateset.data['_via_attributes'],
        #     "_via_img_metadata": {},
        # }
        for key in tqdm(self.via_dateset.data['_via_img_metadata']):
            item = self.via_dateset.data['_via_img_metadata'][key]
            print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                regions = []
                for region in item["regions"]:
                    if moderation_image_dir is not None and (region["shape_attributes"]["name"] != "polygon"
                                                             or "label" not in region["region_attributes"]):
                        moderation_img_path = os.path.join(moderation_image_dir, os.path.basename(filename))
                        cv2.imwrite(moderation_img_path, image)
                        continue

                    keypoints = VIABoxes.get_keypoints(region)
                    keypoints_norm = normalize_rect_new(keypoints)
                    regions.append({
                        "region_attributes": region["region_attributes"],
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [int(float(point[0])) for point in keypoints_norm],
                            "all_points_y": [int(float(point[1])) for point in keypoints_norm]
                        }
                    })

                    if (region["shape_attributes"]["name"] == "polygon"
                            and region["region_attributes"]["label"] in filtered_classes):
                        print(region["region_attributes"]["label"])
                        keypoints = VIABoxes.get_keypoints(region)
                        keypoints_norm = normalize_rect_new(keypoints)
                        min_x_box = round(min([keypoint[0] for keypoint in keypoints_norm]))
                        min_y_box = round(min([keypoint[1] for keypoint in keypoints_norm]))
                        max_x_box = round(max([keypoint[0] for keypoint in keypoints_norm]))
                        max_y_box = round(max([keypoint[1] for keypoint in keypoints_norm]))
                        bbox = [min_x_box, min_y_box, max_x_box, max_y_box]
                        box_w = max_x_box-min_x_box
                        box_h = max_y_box-min_y_box
                        bbox_filename = VIABoxes.get_bbox_filename(basename, bbox)

                        bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=0, w=w, h=h)
                        is_blurry = check_blurriness(bbox_image, threshold=15)
                        if is_blurry or box_h < min_h or box_w < min_w:
                            if moderation_bbox_dir is not None:
                                moderation_bbox_path = os.path.join(moderation_bbox_dir, bbox_filename)
                                cv2.imwrite(moderation_bbox_path, bbox_image)
                            continue

                        bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=0, w=224, h=224)
                        orientation_0_180__90_270 = self.numberplate_orientation_0_180__90_270.predict([bbox_image])[0]
                        #cv2.imshow(f"orig orientation: {orientation}", bbox_image)
                        #cv2.waitKey(0)
                        if orientation_0_180__90_270 == 1:  # class=90/270
                            bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=3, w=300, h=100)
                        orientation_0_180 = self.numberplate_orientation_0_180.predict([bbox_image])[0]

                        print("orientation_0_180, orientation_0_180__90_270", orientation_0_180, orientation_0_180__90_270)
                        if orientation_0_180 == 1 and orientation_0_180__90_270 == 0:
                            bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=2)
                        elif orientation_0_180 == 1 and orientation_0_180__90_270 == 1:
                            bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=1)
                        elif orientation_0_180 == 0 and orientation_0_180__90_270 == 1:
                            bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=3)
                        else:
                            bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=0)
                        # bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=0)

                        #cv2.imshow(f"res orientation: {orientation}", bbox_image)
                        #cv2.waitKey(0)

                        if wrong_shift_strategy == "rotate":
                            bbox_path = os.path.join(target_dir, bbox_filename)
                            cv2.imwrite(bbox_path, bbox_image)
                            print(f'    region "{bbox_filename}" done')
                        elif moderation_bbox_dir is not None:
                            moderation_bbox_path = os.path.join(moderation_bbox_dir, bbox_filename)
                            cv2.imwrite(moderation_bbox_path, bbox_image)
                            print(f'    region "{bbox_filename}" done')

        #         new_via_data['_via_img_metadata'][key] = {
        #             "file_attributes": self.via_dateset.data['_via_img_metadata'][key]["file_attributes"],
        #             "filename": self.via_dateset.data['_via_img_metadata'][key]["filename"],
        #             "regions": regions,
        #             "size": self.via_dateset.data['_via_img_metadata'][key]["size"]
        #         }
        #
        # with open(os.path.join(self.dataset_dir, 'via_moderated_data.json'), 'w') as f:
        #     json.dump(new_via_data, f)
