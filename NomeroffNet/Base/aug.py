import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)

def aug(imgs):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
             sometimes(iaa.Crop(percent=(0, 0.01))),
             iaa.Affine(
                 scale={"x": (0.995, 1.01), "y": (0.995, 1.01)},
                 translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                 rotate=(-3, 3),
                 shear=(-3, 3),
                 #cval=(0, 255)
             ),
              iaa.SomeOf((0, 5),
              [
                      sometimes(iaa.OneOf([
                              iaa.OneOf([
                                  iaa.GaussianBlur((1, 1.2)),
                                  iaa.AverageBlur(k=(1, 3)),
                                  iaa.MedianBlur(k=(1, 3))
                              ]),
                              iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                              iaa.Grayscale(alpha=(0.0, 1.0)),
                              iaa.OneOf([
                                  iaa.EdgeDetect(alpha=(0, 0.7)),
                                  iaa.DirectedEdgeDetect(
                                      alpha=(0, 0.7), direction=(0.0, 1.0)
                                  ),
                              ]),
                      ])),
                      sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                      sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005*255), per_channel=0.001)),
                      sometimes(iaa.Dropout((0.001, 0.01), per_channel=0.5)),
                      sometimes(iaa.Add((-10, 10), per_channel=0.5)),
                      sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                      sometimes(iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=0.05)),
                      sometimes(iaa.PiecewiseAffine(scale=(0.001, 0.005)))
                  ],
                  random_order=True
              )
         ],
        random_order=True
    )
    return seq.augment_images(imgs)