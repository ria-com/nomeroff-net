# -*- coding: utf-8 -*-
"""
VIA (via_region_data.json) -> YOLO converters

Supported formats (yolo_format):
  - "normal": bbox      -> class xc yc w h
  - "obb":    4 points  -> class x1 y1 x2 y2 x3 y3 x4 y4   (all normalized)
  - "pose":   bbox+kpts -> class xc yc w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4

Key point order for "pose":
  - For polygon: EXACT order as in VIA all_points_x/all_points_y (first 4 points)
  - For rect: generated in fixed order TL, TR, BR, BL (because VIA rect has no point order)

Notes:
  - Images are always converted to RGB ndarray HxWx3 to avoid grayscale crashes.
  - EXIF rotation preserved.
  - Rotation augmentation:
      * normal/obb use generate_image_rotation_variants() (as before)
      * pose uses internal rotation that rotates kpts consistently
"""

from __future__ import annotations

import os
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError

from .image_processing import generate_image_rotation_variants


# -------------------------
# Image utils
# -------------------------

def rotate_image_by_exif(image: Image.Image) -> Image.Image:
    """Rotate photo by EXIF Orientation tag (if present)."""
    try:
        orientation = 274  # Exif orientation tag id
        exif_raw = image._getexif()
        if exif_raw is not None:
            exif = dict(exif_raw.items())
            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    image = ImageOps.mirror(image)
    except Exception:
        # Do not fail on weird EXIF
        pass
    return image


def _ensure_rgb_ndarray(img: np.ndarray) -> np.ndarray:
    """Ensure image is HxWx3 uint8-like array."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _read_image_rgb(path: str) -> Optional[np.ndarray]:
    """Read image with PIL, apply EXIF rotation, convert to RGB ndarray."""
    try:
        pil = Image.open(path)
    except (UnidentifiedImageError, OSError):
        return None
    pil = rotate_image_by_exif(pil)
    pil = pil.convert("RGB")
    arr = np.array(pil)
    return _ensure_rgb_ndarray(arr)


# -------------------------
# Geometry helpers
# -------------------------

def _bbox_from_points_xy(points: Sequence[Tuple[float, float]]) -> List[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _via_polygon_points(region: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    sa = region.get("shape_attributes", {}) or {}
    px = sa.get("all_points_x", None)
    py = sa.get("all_points_y", None)
    if px is None or py is None:
        return None
    if len(px) < 4 or len(py) < 4:
        return None
    pts = list(zip(px, py))
    # IMPORTANT: keep VIA order (requested). If more than 4 points -> take first 4 in VIA order.
    if len(pts) > 4:
        pts = pts[:4]
    return [(float(x), float(y)) for x, y in pts]


def _via_rect_points(region: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    sa = region.get("shape_attributes", {}) or {}
    if sa.get("name") != "rect":
        return None
    x = float(sa.get("x", 0))
    y = float(sa.get("y", 0))
    w = float(sa.get("width", 0))
    h = float(sa.get("height", 0))
    # No VIA point order for rect -> fixed order TL, TR, BR, BL
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _label_from_region(region: Dict[str, Any], classes: List[str], cat2label: Dict[str, int]) -> int:
    label = classes[0]
    ra = region.get("region_attributes", None) or {}
    if ra.get("label", None) is not None:
        if isinstance(ra["label"], str):
            label = ra["label"]
        elif isinstance(ra["label"], int):
            if 0 <= ra["label"] < len(classes):
                label = classes[ra["label"]]
    return cat2label.get(label, -1)


# -------------------------
# Save labels: normal bbox
# -------------------------

def save_in_yolo_format(
    image: np.ndarray,
    target_boxes: List[List[float]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    debug: bool = True,
    suffix: str = "",
) -> None:
    image = _ensure_rgb_ndarray(image)
    height, width = image.shape[:2]

    to_txt_data: List[str] = []
    is_corrupted = 0

    for bbox, label_id in zip(target_boxes, labels):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        mx = bbox[0] + w / 2
        my = bbox[1] + h / 2

        yolo_bbox = [label_id, mx / width, my / height, w / width, h / height]
        if (yolo_bbox[1] >= 1 or yolo_bbox[2] >= 1 or yolo_bbox[3] >= 1 or yolo_bbox[4] >= 1
                or yolo_bbox[1] < 0 or yolo_bbox[2] < 0 or yolo_bbox[3] < 0 or yolo_bbox[4] < 0):
            print("[corrupted]", os.path.join(path_to_res_images, image_id), width, height)
            print("bbox", bbox)
            print("yolo_bbox", yolo_bbox)
            is_corrupted = 1

        to_txt_data.append(" ".join([str(item) for item in yolo_bbox]))

        if debug or is_corrupted:
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 120, 255),
                3,
            )

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
        cv2.imwrite(
            os.path.join(path_to_res_images, f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"),
            image[..., ::-1],
        )


# -------------------------
# Save labels: OBB (4 points)
# -------------------------

def save_in_yolo_obb_format(
    image: np.ndarray,
    target_boxes: List[List[float]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    debug: bool = True,
    suffix: str = "",
) -> None:
    image = _ensure_rgb_ndarray(image)
    height, width = image.shape[:2]

    to_txt_data: List[str] = []
    is_corrupted = 0

    for bbox, label_id in zip(target_boxes, labels):
        # bbox format: [x1,y1,x2,y2,x3,y3,x4,y4] in pixels
        if len(bbox) != 8:
            is_corrupted = 1
            continue

        normalized = [
            bbox[0] / width, bbox[1] / height,
            bbox[2] / width, bbox[3] / height,
            bbox[4] / width, bbox[5] / height,
            bbox[6] / width, bbox[7] / height,
        ]
        if any((v < 0 or v > 1) for v in normalized):
            print(f'width: {width} height: {height}')
            print('normalized_bbox', normalized)
            print("[corrupted]", os.path.join(path_to_res_images, image_id))
            is_corrupted = 1

        yolo_line = [label_id] + normalized
        to_txt_data.append(" ".join([str(item) for item in yolo_line]))

    res_path = f'{path_to_res_ann}/{".".join(image_id.split(".")[:-1])}{suffix}.txt'
    if debug or is_corrupted:
        # keep behavior: no heavy visualization by default
        pass
    else:
        with open(res_path, "w") as wFile:
            wFile.write("\n".join(to_txt_data))
        cv2.imwrite(
            os.path.join(path_to_res_images, f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"),
            image[..., ::-1],
        )


# -------------------------
# Save labels: POSE (bbox + 4 kpts)
# -------------------------

def save_in_yolo_pose_format(
    image: np.ndarray,
    target_boxes: List[List[float]],
    keypoints_list: List[List[Tuple[float, float]]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    debug: bool = True,
    suffix: str = "",
    kpt_visible_value: int = 2,
) -> None:
    """
    YOLO pose format:
      class xc yc w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    """
    image = _ensure_rgb_ndarray(image)
    height, width = image.shape[:2]

    to_txt_data: List[str] = []
    is_corrupted = 0

    for bbox, kpts, label_id in zip(target_boxes, keypoints_list, labels):
        if len(kpts) != 4:
            is_corrupted = 1
            continue

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        mx = bbox[0] + bw / 2
        my = bbox[1] + bh / 2

        parts: List[float] = [float(label_id), mx / width, my / height, bw / width, bh / height]

        for (x, y) in kpts:
            parts.extend([x / width, y / height])

        # corruption check
        # bbox:
        if any((v < 0 or v > 1) for v in parts[1:5]):
            is_corrupted = 1
        # kpts:
        for i in range(5, len(parts), 3):
            xn, yn = parts[i], parts[i + 1]
            if xn < 0 or xn > 1 or yn < 0 or yn > 1:
                is_corrupted = 1
                break

        to_txt_data.append(" ".join([str(item) for item in parts]))

        if debug or is_corrupted:
            # draw bbox + kpts in VIA order
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 120, 255),
                2,
            )
            for idx, (x, y) in enumerate(kpts):
                cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
                cv2.putText(image, str(idx), (int(x) + 4, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        cv2.imwrite(
            os.path.join(path_to_res_images, f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"),
            image[..., ::-1],
        )


# -------------------------
# Rotation augmentation (keep old API names)
# -------------------------

def rotation_augumentation(
    image: np.ndarray,
    target_boxes: List[List[float]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    angles: Optional[List[int]] = None,
    debug: bool = False,
) -> None:
    """Old name kept for backward-compat. BBox rotation uses generate_image_rotation_variants()."""
    if angles is None:
        angles = [90, 180, 270]
    variant_images, variants_bboxes = generate_image_rotation_variants(image, target_boxes, angles=angles)
    angles_full = [0, *angles]
    for img_v, bboxes_v, angle in zip(variant_images, variants_bboxes, angles_full):
        save_in_yolo_format(
            img_v,
            bboxes_v,
            path_to_res_ann,
            path_to_res_images,
            image_id,
            labels,
            suffix=f"_{angle}",
            debug=debug,
        )


def rotation_augumentation_obb(
    image: np.ndarray,
    target_boxes: List[List[float]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    angles: Optional[List[int]] = None,
    debug: bool = False,
) -> None:
    """Old name kept for backward-compat. OBB rotation uses generate_image_rotation_variants()."""
    if angles is None:
        angles = [90, 180, 270]
    variant_images, variants_bboxes = generate_image_rotation_variants(image, target_boxes, angles=angles)
    angles_full = [0, *angles]
    for img_v, bboxes_v, angle in zip(variant_images, variants_bboxes, angles_full):
        save_in_yolo_obb_format(
            img_v,
            bboxes_v,
            path_to_res_ann,
            path_to_res_images,
            image_id,
            labels,
            suffix=f"_{angle}",
            debug=debug,
        )


def _rotate_image_and_points_cw(
    image: np.ndarray,
    points_list: List[List[Tuple[float, float]]],
    angle: int,
) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """
    Rotate image and points CLOCKWISE by angle in {90,180,270}.
    Points are given in pixel coords in original image.
    """
    if angle == 0:
        return image, points_list

    h, w = image.shape[:2]

    def rot_pt(x: float, y: float) -> Tuple[float, float]:
        if angle == 90:
            # (x, y) -> (h-1-y, x), new dims (w' = h, h' = w)
            return float(h - 1 - y), float(x)
        if angle == 180:
            # (x, y) -> (w-1-x, h-1-y)
            return float(w - 1 - x), float(h - 1 - y)
        if angle == 270:
            # (x, y) -> (y, w-1-x), new dims (w' = h, h' = w)
            return float(y), float(w - 1 - x)
        raise ValueError("angle must be 0/90/180/270")

    if angle == 90:
        img_r = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img_r = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        img_r = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("angle must be 0/90/180/270")

    pts_out: List[List[Tuple[float, float]]] = []
    for kpts in points_list:
        pts_out.append([rot_pt(x, y) for (x, y) in kpts])

    return img_r, pts_out


def rotation_augumentation_pose(
    image: np.ndarray,
    keypoints_list: List[List[Tuple[float, float]]],
    path_to_res_ann: str,
    path_to_res_images: str,
    image_id: str,
    labels: List[int],
    angles: Optional[List[int]] = None,
    debug: bool = False,
    kpt_visible_value: int = 2,
) -> None:
    """Pose rotation (CW): rotates image + keypoints, bbox recomputed from kpts."""
    if angles is None:
        angles = [90, 180, 270]

    # include angle 0
    angles_full = [0, *angles]

    for angle in angles_full:
        img_r, kpts_r = _rotate_image_and_points_cw(image, keypoints_list, angle)
        # bbox recompute from rotated points
        bboxes_r = [_bbox_from_points_xy(kpts) for kpts in kpts_r]

        save_in_yolo_pose_format(
            img_r,
            bboxes_r,
            kpts_r,
            path_to_res_ann,
            path_to_res_images,
            image_id,
            labels,
            suffix=f"_{angle}",
            debug=debug,
            kpt_visible_value=kpt_visible_value,
        )


# -------------------------
# Main converter
# -------------------------

def convert_dataset_to_yolo_format(
    path_to_res_ann: str,
    path_to_res_images: str,
    path_to_images: str,
    path_to_json: str,
    classes: Optional[List[str]] = None,
    debug: bool = True,
    is_generate_image_rotation_variants: bool = False,
    yolo_format: str = "normal",  # "normal" | "obb" | "pose"
    pose_kpt_visible_value: int = 2,
) -> None:
    """
    Convert VIA annotations into YOLO format.

    Args:
        path_to_res_ann: output labels dir
        path_to_res_images: output images dir
        path_to_images: input images dir
        path_to_json: via_region_data.json path
        classes: list of class names
        debug: show debug plots / do not write files if True (keeps old behavior)
        is_generate_image_rotation_variants: generate *_0, *_90, *_180, *_270 variants
        yolo_format: "normal", "obb", "pose"
        pose_kpt_visible_value: v for keypoints (0/1/2), default 2
    """
    if classes is None:
        classes = ["numberplate"]

    os.makedirs(path_to_res_ann, exist_ok=True)
    os.makedirs(path_to_res_images, exist_ok=True)

    with open(path_to_json) as ann:
        ann_data = json.load(ann)

    cat2label = {k: i for i, k in enumerate(classes)}
    image_list = ann_data
    all_labels: Dict[str, int] = {}
    c = Counter()

    meta = image_list.get("_via_img_metadata", {})
    for _id in tqdm.tqdm(meta):
        image_id = meta[_id].get("filename", None)
        if not image_id:
            c["skip_no_filename"] += 1
            continue

        filename = os.path.join(path_to_images, image_id)
        image = _read_image_rgb(filename)
        if image is None:
            print(f"[skip unreadable] {filename}")
            c["skip_unreadable"] += 1
            continue

        target_boxes: List[List[float]] = []
        labels: List[int] = []
        keypoints_list: List[List[Tuple[float, float]]] = []

        regions = meta[_id].get("regions", []) or []
        for region in regions:
            label_id = _label_from_region(region, classes, cat2label)
            if label_id == -1:
                c["skip_label"] += 1
                continue

            # track found labels
            # (best-effort: reconstruct label string for printing)
            all_labels[classes[label_id] if 0 <= label_id < len(classes) else str(label_id)] = 1

            sa = region.get("shape_attributes", {}) or {}
            name = sa.get("name", None)

            if yolo_format == "normal":
                if name == "polygon":
                    pts = _via_polygon_points(region)
                    if pts is None:
                        c["skip_attributes"] += 1
                        continue
                    bbox = _bbox_from_points_xy(pts)
                elif name == "rect":
                    x = float(sa.get("x", 0))
                    y = float(sa.get("y", 0))
                    w = float(sa.get("width", 0))
                    h = float(sa.get("height", 0))
                    bbox = [x, y, x + w, y + h]
                else:
                    c["skip_attributes"] += 1
                    continue

                target_boxes.append(bbox)
                labels.append(label_id)
                c["count_attributes"] += 1

            elif yolo_format == "obb":
                if name != "polygon":
                    c["skip_attributes"] += 1
                    continue
                pts = _via_polygon_points(region)
                if pts is None:
                    c["skip_attributes"] += 1
                    continue
                if len(pts) != 4:
                    c["skip_not4pts"] += 1
                    continue
                # obb expects 8 numbers (x1,y1,x2,y2,x3,y3,x4,y4) in pixel coords
                obb = [pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1]]
                target_boxes.append(obb)
                labels.append(label_id)
                c["count_attributes"] += 1

            elif yolo_format == "pose":
                # pose needs 4 points + bbox derived from them
                if name == "polygon":
                    pts = _via_polygon_points(region)  # EXACT VIA order (first 4)
                    if pts is None:
                        c["skip_attributes"] += 1
                        continue
                    if len(pts) != 4:
                        c["skip_not4pts"] += 1
                        continue
                elif name == "rect":
                    pts = _via_rect_points(region)
                    if pts is None:
                        c["skip_attributes"] += 1
                        continue
                else:
                    c["skip_attributes"] += 1
                    continue

                bbox = _bbox_from_points_xy(pts)
                target_boxes.append(bbox)
                keypoints_list.append(pts)
                labels.append(label_id)
                c["count_attributes"] += 1

            else:
                raise ValueError('yolo_format must be one of: "normal", "obb", "pose"')

        # Write / debug
        if is_generate_image_rotation_variants:
            if yolo_format == "normal":
                rotation_augumentation(
                    image, target_boxes,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug
                )
            elif yolo_format == "obb":
                rotation_augumentation_obb(
                    image, target_boxes,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug
                )
            else:  # pose
                rotation_augumentation_pose(
                    image, keypoints_list,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug,
                    kpt_visible_value=pose_kpt_visible_value
                )
        else:
            if yolo_format == "normal":
                save_in_yolo_format(
                    image, target_boxes,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug
                )
            elif yolo_format == "obb":
                save_in_yolo_obb_format(
                    image, target_boxes,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug
                )
            else:  # pose
                save_in_yolo_pose_format(
                    image, target_boxes, keypoints_list,
                    path_to_res_ann, path_to_res_images,
                    image_id, labels,
                    debug=debug,
                    kpt_visible_value=pose_kpt_visible_value
                )

    print(f"[INFO] format type {yolo_format}")
    print(f"[INFO] find labels {list(all_labels.keys())}")
    print(f"[INFO] use labels {classes}")
    print(f"[INFO] count attributes {c.get('count_attributes', 0)}")
    print(f"[INFO] skip attributes {c.get('skip_attributes', 0)}")
    if c.get("skip_unreadable", 0):
        print(f"[INFO] skip unreadable {c['skip_unreadable']}")
    if c.get("skip_not4pts", 0):
        print(f"[INFO] skip polygons not 4 pts {c['skip_not4pts']}")
