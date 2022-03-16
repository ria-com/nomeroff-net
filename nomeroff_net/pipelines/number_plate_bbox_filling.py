import cv2
import numpy as np
from typing import Any, Dict
from .number_plate_localization import NumberPlateLocalization
from nomeroff_net.tools import unzip


class NumberPlateBboxFilling(NumberPlateLocalization):
    """
    Number Plate Localization
    """

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images_target_boxes, images = unzip(NumberPlateLocalization.postprocess(self,
                                                                                inputs,
                                                                                **postprocess_parameters))
        filled_number_plate = []
        for image_target_boxes, img in zip(images_target_boxes, images):
            img = img.astype(np.uint8)
            for target_box in image_target_boxes:
                cv2.rectangle(img,
                              (int(target_box[0]), int(target_box[1])),
                              (int(target_box[2]), int(target_box[3])),
                              (0, 0, 0),
                              -1)
            filled_number_plate.append(img)
        return filled_number_plate
