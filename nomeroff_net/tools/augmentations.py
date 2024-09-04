#!/usr/bin/env python
# coding: utf-8
"""
Test running:
python3 -m nomeroff_net.tools.augmentations -f nomeroff_net/tools/augmentations.py
"""
from typing import List
import albumentations as A
import random
import numpy as np


def aug_seed(num: int = None) -> None:
    random.seed(num)
    np.random.seed(num)


def aug(imgs: List) -> List:
    #print(type(imgs[0]), imgs[0])
    sometimes = lambda aug: A.OneOf([aug, A.NoOp()], p=0.5)
    transform = A.Compose([
        sometimes(A.RandomCropFromBorders(p=0.01, always_apply=False, crop_left=0.01, crop_right=0.01, crop_top=0.01, crop_bottom=0.01)),
        A.OneOf([
            A.Affine(
                scale=(0.995, 1.01),
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                rotate=(-3, 3),
                shear=(-3, 3),
                p=1.0
            )
        ], p=1.0),
        A.SomeOf([
            sometimes(A.OneOf([
                A.GaussianBlur(sigma_limit=(1, 1.2)),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3)
            ])),
            sometimes(A.RandomBrightnessContrast(brightness_limit=0.5, p=1.0)),
            sometimes(A.ToGray(p=1.0)),
            sometimes(A.OneOf([
                #A.EdgeDetection(p=0.7),
                A.Emboss(p=0.7)
            ])),
            sometimes(A.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5), p=1.0)),
            sometimes(A.GaussNoise(var_limit=(0.0, 0.005*255), mean=0, p=1.0)),
            sometimes(A.CoarseDropout(max_holes=1, max_height=0.01, max_width=0.01, min_holes=1, p=0.5)),
            sometimes(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5)),
            sometimes(A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.5)),
            #sometimes(A.ElasticTransform(alpha=0.1, sigma=0.05, alpha_affine=0.01, p=1.0)),
            sometimes(A.PiecewiseAffine(scale=(0.001, 0.005), p=1.0))
        ], n=2, p=1.0)
    ], p=1.0)

    augmented_imgs = [transform(image=img)["image"] for img in imgs]
    return augmented_imgs


def light_aug(imgs: List) -> List:
    sometimes = lambda aug: A.OneOf([aug, A.NoOp()], p=0.5)

    transform = A.Compose([
        sometimes(A.RandomCropFromBorders(p=0.05, always_apply=False, crop_left=0.05, crop_right=0.05, crop_top=0.05,
                                          crop_bottom=0.05)),
        A.OneOf([
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-10, 10),
                p=1.0
            ),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0)
        ], p=1.0),
        A.SomeOf([
            sometimes(A.OneOf([
                A.GaussianBlur(sigma_limit=(1, 2)),
                A.MedianBlur(blur_limit=5),
                A.Blur(blur_limit=5)
            ])),
            sometimes(A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.7, p=1.0)),
            sometimes(A.ToGray(p=1.0)),
            sometimes(A.OneOf([
                A.Emboss(p=0.7),
                A.Superpixels(p_replace=0.1, n_segments=100, p=0.7)
            ])),
            sometimes(A.Sharpen(alpha=(0.2, 1.0), lightness=(0.5, 2.0), p=1.0)),
            sometimes(A.GaussNoise(var_limit=(0.0, 0.01 * 255), mean=0, p=1.0)),
            sometimes(A.CoarseDropout(max_holes=3, max_height=0.05, max_width=0.05, min_holes=1, p=0.5)),
            sometimes(A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7)),
            sometimes(A.MultiplicativeNoise(multiplier=(0.3, 1.7), per_channel=True, p=0.7)),
            sometimes(A.PiecewiseAffine(scale=(0.002, 0.02), p=1.0)),
            sometimes(A.GridDistortion(p=0.5)),
            sometimes(A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0)),
        ], n=5, p=1.0)  # Замість діапазону вкажіть чітке значення n
    ], p=1.0)

    augmented_imgs = [transform(image=img)["image"] for img in imgs]
    return augmented_imgs



def medium_aug(imgs: List) -> List:
    sometimes = lambda aug: A.OneOf([aug, A.NoOp()], p=0.5)

    transform = A.Compose([
        sometimes(A.RandomCropFromBorders(p=0.1, always_apply=False, crop_left=0.1, crop_right=0.1, crop_top=0.1,
                                          crop_bottom=0.1)),
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                shear=(-15, 15),
                p=1.0
            ),
            A.Perspective(scale=(0.1, 0.2), keep_size=True, p=1.0),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None, p=1.0)
        ], p=1.0),
        A.SomeOf([
            sometimes(A.OneOf([
                A.GaussianBlur(sigma_limit=(1.5, 3)),
                A.MedianBlur(blur_limit=7),
                A.Blur(blur_limit=7)
            ])),
            sometimes(A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=1.0)),
            sometimes(A.ToGray(p=1.0)),
            sometimes(A.OneOf([
                A.Emboss(p=0.8),
                A.Superpixels(p_replace=0.2, n_segments=50, p=0.8),
                A.FancyPCA(alpha=0.1, p=0.8)
            ])),
            sometimes(A.Sharpen(alpha=(0.5, 1.0), lightness=(0.3, 2.5), p=1.0)),
            sometimes(A.GaussNoise(var_limit=(0.0, 0.02 * 255), mean=0, p=1.0)),
            sometimes(A.CoarseDropout(max_holes=5, max_height=0.1, max_width=0.1, min_holes=2, p=0.7)),
            sometimes(A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.8)),
            sometimes(A.MultiplicativeNoise(multiplier=(0.2, 2.0), per_channel=True, p=0.8)),
            sometimes(A.PiecewiseAffine(scale=(0.01, 0.03), p=1.0)),
            sometimes(A.GridDistortion(p=0.7)),
            sometimes(A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0)),
            sometimes(A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=0.8)),
            sometimes(A.Solarize(threshold=128, p=0.5)),
            sometimes(A.ChannelShuffle(p=0.5)),
            sometimes(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)),
            sometimes(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7)),
        ], n=8, p=1.0)
    ], p=1.0)

    augmented_imgs = [transform(image=img)["image"] for img in imgs]
    return augmented_imgs


def hard_aug(imgs: List) -> List:
    sometimes = lambda aug: A.OneOf([aug, A.NoOp()], p=0.5)

    transform = A.Compose([
        sometimes(A.RandomCropFromBorders(p=0.2, always_apply=False, crop_left=0.2, crop_right=0.2, crop_top=0.2,
                                          crop_bottom=0.2)),
        A.OneOf([
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                rotate=(-45, 45),
                shear=(-20, 20),
                p=1.0
            ),
            A.Perspective(scale=(0.2, 0.3), keep_size=True, p=1.0),
            A.ElasticTransform(alpha=150, sigma=150 * 0.05, alpha_affine=None, p=1.0),
            A.GridDistortion(num_steps=10, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0)
        ], p=1.0),
        A.SomeOf([
            sometimes(A.OneOf([
                A.GaussianBlur(sigma_limit=(2, 5)),
                A.MedianBlur(blur_limit=11),
                A.Blur(blur_limit=11)
            ])),
            sometimes(A.RandomBrightnessContrast(brightness_limit=1.0, contrast_limit=1.0, p=1.0)),
            sometimes(A.ToGray(p=1.0)),
            sometimes(A.OneOf([
                A.Emboss(p=0.9),
                A.Superpixels(p_replace=0.4, n_segments=20, p=0.9),
                A.FancyPCA(alpha=0.2, p=0.9),
                A.Equalize(p=0.9),
            ])),
            sometimes(A.Sharpen(alpha=(0.7, 1.0), lightness=(0.0, 3.0), p=1.0)),
            sometimes(A.GaussNoise(var_limit=(0.0, 0.05 * 255), mean=0, p=1.0)),
            sometimes(A.CoarseDropout(max_holes=10, max_height=0.2, max_width=0.2, min_holes=5, p=0.9)),
            sometimes(A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.9)),
            sometimes(A.MultiplicativeNoise(multiplier=(0.1, 3.0), per_channel=True, p=0.9)),
            sometimes(A.PiecewiseAffine(scale=(0.05, 0.1), p=1.0)),
            sometimes(A.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5, p=0.9)),
            sometimes(A.Solarize(threshold=100, p=0.9)),
            sometimes(A.ChannelShuffle(p=0.9)),
            sometimes(A.ISONoise(color_shift=(0.05, 0.1), intensity=(0.3, 1.0), p=0.9)),
            sometimes(A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=40, p=0.9)),
            sometimes(
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,
                               p=0.9)),
            sometimes(A.CLAHE(clip_limit=4.0, p=0.9)),
            sometimes(A.SnowFall(snow_point_lower=0.2, snow_point_upper=0.5, brightness_coeff=2.5, p=0.9)),
            sometimes(A.Fog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=0.9)),
            sometimes(A.Rain(drop_length=10, drop_width=1, drop_color=(200, 200, 200), blur_value=2,
                             brightness_coefficient=0.8, p=0.9)),
            sometimes(A.SunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), angle_lower=0.5, p=0.9)),
            sometimes(A.Cutout(num_holes=10, max_h_size=40, max_w_size=40, fill_value=0, p=0.9)),
            sometimes(A.PixelDropout(dropout_prob=0.2, per_channel=True, p=0.9)),
        ], n=10, p=1.0)
    ], p=1.0)

    augmented_imgs = [transform(image=img)["image"] for img in imgs]
    return augmented_imgs


if __name__ == '__main__':
    import os
    import glob
    import cv2

    aug_funcs = {
        "simple": aug,
        "light": light_aug,
        "medium": medium_aug,
        "hard": hard_aug
    }

    # Set up the path to your image directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.abspath(os.path.join(dir_path, "../../data/examples/numberplate_zone_images/"))

    print("Image directory:", image_dir)

    # Get all image paths in the directory
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    print("Found images:", image_paths)

    for aug_name in aug_funcs:
        print("aug_name", aug_name)
        aug_func = aug_funcs[aug_name]
        # Load the images
        images = [cv2.imread(image_path) for image_path in image_paths]

        # Apply augmentations
        aug_images = aug_func(images)

        # Save augmented images
        #res_dir = image_dir
        res_dir = f"/home/dmitro/Downloads/aug/{aug_name}/"
        save_dir = os.path.join(res_dir, "augmented")
        os.makedirs(save_dir, exist_ok=True)

        for i, (aug_img, image_path) in enumerate(zip(aug_images, image_paths)):
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, aug_img)
            print(f"Saved augmented image: {save_path}")

