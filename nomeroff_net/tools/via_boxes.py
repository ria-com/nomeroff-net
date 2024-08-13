import os
import cv2
import numpy as np
from tqdm import tqdm
from nomeroff_net.tools.via import VIADataset
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect_new
from nomeroff_net.pipes.number_plate_classificators.orientation_detector import OrientationDetector
from .image_processing import reshape_points


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


class VIABoxes:
    def __init__(self,
                 dataset_json,
                 verbose=False):
        self.dataset_json = dataset_json
        self.dataset_dir = os.path.dirname(dataset_json)
        self.via_dateset = VIADataset(label_type="radio", verbose=verbose)
        self.via_dateset.load_from_via_file(dataset_json)
        self.debug = verbose
        self.orientation_detector = OrientationDetector()
        self.orientation_detector.load()

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

    def make_transformed_boxes(self, target_dir='./',
                               moderation_bbox_dir=None,
                               moderation_image_dir=None,
                               wrong_shift_strategy="rotate",  # may by: "rotate", "moderate"
                               filtered_classes=None,
                               ):
        if filtered_classes is None:
            filtered_classes = ["numberplate"]
        for key in tqdm(self.via_dateset.data['_via_img_metadata']):
            item = self.via_dateset.data['_via_img_metadata'][key]
            print(f"Processing \"{item['filename']}\"")
            filename = os.path.join(self.dataset_dir, item['filename'])
            if os.path.exists(filename):
                basename = item['filename'].split('.')[0]
                image = cv2.imread(filename)
                for region in item["regions"]:
                    if moderation_image_dir is not None and (region["shape_attributes"]["name"] != "polygon"
                                                             or "label" not in region["region_attributes"]):
                        moderation_img_path = os.path.join(moderation_image_dir, os.path.basename(filename))
                        cv2.imwrite(moderation_img_path, image)
                        continue

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

                        bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=0)
                        is_blurry = check_blurriness(bbox_image, threshold=15)
                        if is_blurry or box_h < 20 or box_w < 60:
                            if moderation_bbox_dir is not None:
                                moderation_bbox_path = os.path.join(moderation_bbox_dir, bbox_filename)
                                cv2.imwrite(moderation_bbox_path, bbox_image)
                            continue

                        orientation = self.orientation_detector.predict([bbox_image])[0]
                        #cv2.imshow(f"orig orientation: {orientation}", bbox_image)
                        #cv2.waitKey(0)
                        if orientation == 1:  # class=90/270
                            rotated_bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm,
                                                                            shift=3)
                            #cv2.imshow(f"lass=90/270 orientation: {orientation}", rotated_bbox_image)
                            #cv2.waitKey(0)
                            new_orientation = self.orientation_detector.predict(rotated_bbox_image)[0]
                            if new_orientation == 0:
                                orientation = 1
                            elif new_orientation == 1:
                                orientation = 0
                            elif new_orientation == 2:
                                orientation = 3
                        if orientation == 0:
                            orientation = 0
                        elif orientation == 1:
                            orientation = 3
                        elif orientation == 2:
                            orientation = 0
                        elif orientation == 3:
                            orientation = 1
                        bbox_image = VIABoxes.get_aligned_image(image, keypoints_norm, shift=orientation)

                        #cv2.imshow(f"res orientation: {orientation}", bbox_image)
                        #cv2.waitKey(0)

                        if orientation == 0 or wrong_shift_strategy == "rotate":
                            bbox_path = os.path.join(target_dir, bbox_filename)
                            cv2.imwrite(bbox_path, bbox_image)
                            print(f'    region "{bbox_filename}" done')
                        elif moderation_bbox_dir is not None:
                            moderation_bbox_path = os.path.join(moderation_bbox_dir, bbox_filename)
                            cv2.imwrite(moderation_bbox_path, bbox_image)
                            print(f'    region "{bbox_filename}" done')
