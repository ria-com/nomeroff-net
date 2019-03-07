import os
import json
import skimage
import numpy as np

from mrcnn.config import Config
from mrcnn import utils

############################################################
#  Configurations
############################################################
class InferenceConfig(Config):
    def __init__(self, config):
         """
             Configuration for training on the  dataset.
             Derives from the base Config class and overrides some values.
             https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py.
         """
         self.__dict__ = config
         super(InferenceConfig, self).__init__()

############################################################
#  Dataset
############################################################
class Dataset(utils.Dataset):
    def load_numberplate(self, subset, config):
        """
                Load a subset of the Numberplate dataset.
                dataset_dir: Root directory of the dataset.
                subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        assert config.CLASS_NAMES and type(config.CLASS_NAMES) == list
        assert config.NAME and type("") == str
        i = 0
        for name in config.CLASS_NAMES[1:]:
            self.add_class(config.NAME, 1, name)
            i += 1

        # Train or validation dataset?
        assert subset in ["train", "val"]
        assert config.DATASET_DIR and type(config.DATASET_DIR) == str
        dataset_dir = os.path.join(config.DATASET_DIR, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = annotations['_via_img_metadata']
        annotations = annotations.values()
        if type(annotations) == dict:
            annotations = list(annotations)

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # print(annotations)
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "numberplate",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
            Returns:
                masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a numberplate dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "numberplate":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "numberplate":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
