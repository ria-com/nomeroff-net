import sys
import os
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import keras
import git

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from .mcm.mcm import download_latest_model

class Detector:
    def download_mrcnn():
        import git
        git.Git("/your/directory/to/clone").clone("git://gitorious.org/git-python/mainline.git")

    def __init__(self, mask_rcnn_dir=None, log_dir="./logs/", mask_rcnn_config = None):
        self.MASK_RCNN_DIR = mask_rcnn_dir
        self.LOG_DIR = log_dir

        DEFAULT_MASK_RCNN_CONFIG = {
          "NAME": "numberplate",
          "GPU_COUNT": 1,
          "IMAGES_PER_GPU": 1,
          "NUM_CLASSES": 2,
          "DETECTION_MIN_CONFIDENCE": 0.7,
          "IMAGE_MAX_DIM": 1024, # work ?
          "IMAGE_RESIZE_MODE": "square" # work ?
        }
        self.NN_MASK_RCNN_CONFIG = mask_rcnn_config or DEFAULT_MASK_RCNN_CONFIG
        if not mask_rcnn_dir is None:
            sys.path.append(mask_rcnn_dir)

        from .nnmrcnn import InferenceConfig
        self.CONFIG = InferenceConfig(self.NN_MASK_RCNN_CONFIG)

        # for frozen graph
        self.INPUT_NODES = ("input_image:0", "input_image_meta:0", "input_anchors:0") # 3 elem

        self.OUTPUT_NODES = ("mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0",
                             "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0") # 7 elem

    def loadFrozenModel(self, FROZEN_MODEL_PATH):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_MODEL_PATH, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            self.input_image, self.input_image_meta, self.input_anchors, self.mrcnn_detection, self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask, self.ROI, self.rpn_class, self.rpn_bbox = tf.import_graph_def(
                graph_def, return_elements = self.INPUT_NODES + self.OUTPUT_NODES
            )

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def loadModel(self, model_path="latest", verbose = 0, mode="inference"):
        if model_path == "latest":
            model_info = download_latest_model(self.get_classname(), "mrcnn")
            model_path = model_info["path"]

        import mrcnn.model as modellib
        self.MODEL = modellib.MaskRCNN(mode="inference", model_dir=self.LOG_DIR, config=self.CONFIG)
        if model_path.split(".")[-1] == "pb":
            self.loadFrozenModel(model_path)
            self.detect = self.frozen_detect
        else:
            self.MODEL.load_weights(model_path, by_name=True)

    def getKerasModel(self):
        return self.MODEL.keras_model

    def normalize(self, images):
        res = []
        for image in images:
            # delete 4 chanel
            res.append(image[..., :3])
        return res

    def detectFromFile(self, image_paths, verbose = 0):
        images = [mpimg.imread(image_path) for image_path in image_paths]
        return self.detect(images, verbose=verbose)

    def detect(self, images, verbose = 0):
        return self.MODEL.detect(self.normalize(images), verbose=verbose)

#    def detect_masks(self, images, verbose = 0):
#        r = self.MODEL.detect(self.normalize(images), verbose=verbose)
#        # loop over of the detected object's bounding boxes and masks
#        for i in range(0, r["rois"].shape[0]):
#            # extract the class ID and mask for the current detection, then
#            # grab the color to visualize the mask (in BGR format)
#            classID = r["class_ids"][i]
#            mask = r["masks"][:, :, i]
#            color = COLORS[classID][::-1]
#
#            # visualize the pixel-wise mask of the object
#            image = visualize.apply_mask(image, mask, color, alpha=0.5)



    def frozen_detect(self, images, verbose = 0):
        if verbose:
            print(self.CONFIG.BATCH_SIZE)
        assert len(images) == self.CONFIG.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.MODEL.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.MODEL.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.CONFIG.BATCH_SIZE,) + anchors.shape)

        if verbose:
            print("molded_images", molded_images)
            print("image_metas", image_metas)
            print("anchors", anchors)

        #print("Run detection")
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = self.sess.run([self.mrcnn_detection, self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask, self.ROI, self.rpn_class, self.rpn_bbox], feed_dict={
                    self.input_image:      molded_images,
                    self.input_image_meta: image_metas,
                    self.input_anchors:    anchors
                })

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.MODEL.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    ############################################################
    #  Training
    ############################################################
    def train(self, augmentation=None, verbose=1):
        keras.backend.clear_session()
        from .nnmrcnn import Dataset
        import mrcnn.model as modellib
        from mrcnn import utils
        from imgaug import augmenters as iaa
        if verbose:
            self.CONFIG.display()
        model = modellib.MaskRCNN(mode="training", config=self.CONFIG, model_dir=self.LOG_DIR)

        # Select weights file to load
        assert self.CONFIG.WEIGHTS and type(self.CONFIG.WEIGHTS) is str

        if self.CONFIG.WEIGHTS.lower() == "coco":
            weights_path = os.path.join(self.LOG_DIR, "mask_rcnn_coco.h5")
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif self.CONFIG.WEIGHTS.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif self.CONFIG.WEIGHTS.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = self.CONFIG.WEIGHTS

        # Load weights
        if verbose:
            print("Loading weights ", weights_path)
        if self.CONFIG.WEIGHTS.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

        # Training dataset.
        if verbose:
            print("Prepare train data")
        dataset_train = Dataset()
        dataset_train.load_numberplate("train", self.CONFIG)
        dataset_train.prepare()

        # Validation dataset
        if verbose:
            print("Prepare validation data")
        dataset_val = Dataset()
        dataset_val.load_numberplate("val", self.CONFIG)
        dataset_val.prepare()

        # Image augmentation
        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
        augmentation_default = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])
        augmentation = augmentation or augmentation_default

        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        if verbose:
            print("Training network")
        model.train(dataset_train, dataset_val,
                    learning_rate=self.CONFIG.LEARNING_RATE,
                    epochs=self.CONFIG.EPOCHS,
                    augmentation=augmentation,
                    layers=self.CONFIG.LAYERS )
