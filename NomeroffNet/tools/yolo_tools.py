import sys
import os
import json
import glob
import cv2
import tqdm
from PIL import Image
from PIL import ExifTags
import numpy as np
from PIL import ImageOps

from .image_processing import generate_image_rotation_variants


def rotate_image_by_exif(image):
    """
    Rotate photo

    Parameters
    ----------
    image
    """
    try:
        orientation = 274  # key of orientation ExifTags
        if image._getexif() is not None:
            exif = dict(image._getexif().items())
            if orientation in exif.keys():
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    image = ImageOps.mirror(image)
    except AttributeError:
        pass
    return image

    
def save_in_yolo_format(image,
                        target_boxes,
                        path_to_res_ann,
                        path_to_res_images,
                        image_id, 
                        labels,
                        debug=True,
                        suffix=""):
    height, width, c = image.shape
    to_txt_data = []
    is_corrupted = 0
    for bbox, label_id in zip(target_boxes, labels):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        mx = bbox[0]+w/2
        my = bbox[1]+h/2

        # class x_center y_center width height
        yolo_bbox = [label_id, mx/width, my/height, w/width, h/height]
        if yolo_bbox[1] >= 1 \
            or yolo_bbox[2] >= 1 \
            or yolo_bbox[3] >= 1 \
            or yolo_bbox[4] >= 1:
            print("[corrupted]", os.path.join(path_to_res_images, image_id), width, height)
            print("bbox", bbox)
            print("yolo_bbox", yolo_bbox)
            is_corrupted = 1
        yolo_bbox = " ".join([str(item) for item in yolo_bbox])
        to_txt_data.append(yolo_bbox)
        if debug or is_corrupted:
            cv2.rectangle(image, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), 
                (0,120,255), 
                3)
    res_path =  f'{path_to_res_ann}/{".".join(image_id.split(".")[:-1])}{suffix}.txt'
    if debug or is_corrupted:
        import matplotlib.pyplot as plt
        
        print(res_path)
        print("\n".join(to_txt_data))
        print("______________________")
        plt.imshow(image)
        plt.show()
        pass
    else:
        with open(res_path, "w") as wFile:
            wFile.write("\n".join(to_txt_data))
        cv2.imwrite(os.path.join(path_to_res_images, 
                                 f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"), 
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        
def rotation_augumentation(image,
                           target_boxes,
                           path_to_res_ann,
                           path_to_res_images,
                           image_id, 
                           labels,
                           angles=[90, 180, 270],
                           debug=False):
    
    variant_images, variants_bboxes = generate_image_rotation_variants(image, 
                                                                       target_boxes, 
                                                                       angles=angles)
    angles = [0, *angles]
    for image, target_boxes, angle in zip(variant_images, variants_bboxes, angles):
        save_in_yolo_format(image,
                            target_boxes,
                            path_to_res_ann,
                            path_to_res_images,
                            image_id, 
                            labels,
                            suffix=f"_{angle}",
                            debug=debug)
        

def convert_dataset_to_yolo_format(path_to_res_ann, 
                                   path_to_res_images, 
                                   path_to_images, 
                                   path_to_json, 
                                   classes = ['numberplate'], 
                                   debug=True,
                                   is_generate_image_rotation_variants=False):
    with open(path_to_json) as ann:
        annData = json.load(ann)
    cat2label = {k: i for i, k in enumerate(classes)}
    image_list = annData
    
    for _id in tqdm.tqdm(image_list["_via_img_metadata"]):
        image_id = image_list["_via_img_metadata"][_id]["filename"]
        filename = f'{path_to_images}/{image_id}'
        pil_image = Image.open(filename)
        pil_image = rotate_image_by_exif(pil_image)
        image = np.array(pil_image)
        target_boxes = []
        labels = []
        for region in image_list["_via_img_metadata"][_id]["regions"]:
            label_id  = 0
            
            if region["shape_attributes"].get("all_points_x", None) is None:
                continue
            if region["shape_attributes"].get("all_points_y", None) is None:
                continue
            bbox = [
                min(region["shape_attributes"]["all_points_x"]),
                min(region["shape_attributes"]["all_points_y"]),
                max(region["shape_attributes"]["all_points_x"]),
                max(region["shape_attributes"]["all_points_y"]),
            ]
            target_boxes.append(bbox)
            labels.append(label_id)
        
        if is_generate_image_rotation_variants:
            rotation_augumentation(image,
                                   target_boxes,
                                   path_to_res_ann, 
                                   path_to_res_images,
                                   image_id, 
                                   labels,
                                   debug=debug)
        else:
            save_in_yolo_format(image,
                                target_boxes,
                                path_to_res_ann, 
                                path_to_res_images,
                                image_id, 
                                labels,
                                debug=debug)