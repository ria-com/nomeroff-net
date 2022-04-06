import os
import json
import cv2
import torch
import tqdm
from PIL import Image
import numpy as np
from PIL import ImageOps

from nomeroff_net.tools.image_processing import generate_image_rotation_variants


def rotate_image_by_exif(image):
    """
    Rotate photo

    Parameters
    ----------
    image
    """
    try:
        orientation = 274  # key of orientation ExifTags
        img_exif = image.getexif()
        if img_exif is not None:
            exif = dict(img_exif.items())
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
                          (0, 120, 255),
                          3)
    res_path = f'{path_to_res_ann}/{".".join(image_id.split(".")[:-1])}{suffix}.txt'
    if debug or is_corrupted:
        import matplotlib.pyplot as plt
        
        print(res_path)
        print("\n".join(to_txt_data))
        print("______________________")
        plt.imshow(image)
        plt.show()
    else:
        with open(res_path, "w") as wFile:
            wFile.write("\n".join(to_txt_data))
        cv2.imwrite(os.path.join(path_to_res_images, 
                                 f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"), 
                    image[..., ::-1])

        
def rotation_augumentation(image,
                           target_boxes,
                           path_to_res_ann,
                           path_to_res_images,
                           image_id, 
                           labels,
                           angles=None,
                           debug=False):
    if angles is None:
        angles = [90, 180, 270]
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
                                   debug=True,
                                   is_generate_image_rotation_variants=False):
    with open(path_to_json) as ann:
        ann_data = json.load(ann)
    image_list = ann_data
    
    for _id in tqdm.tqdm(image_list["_via_img_metadata"]):
        image_id = image_list["_via_img_metadata"][_id]["file_name"]
        file_name = f'{path_to_images}/{image_id}'
        pil_image = Image.open(file_name)
        pil_image = rotate_image_by_exif(pil_image)
        image = np.array(pil_image)
        target_boxes = []
        labels = []
        for region in image_list["_via_img_metadata"][_id]["regions"]:
            label_id = 0
            
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


def clip_coords(boxes, shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_predicted_coords(img, pred, orig_img_shape):
    res = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], orig_img_shape).round()
            res.append(det.cpu().detach().numpy())
    return res
