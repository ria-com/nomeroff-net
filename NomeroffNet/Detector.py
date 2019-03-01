import os
import sys
import numpy as np
import tensorflow as tf

class Detector:
    def __init__(self, mask_rcnn_dir, log_dir, mask_rcnn_config = None):
        if mask_rcnn_dir != None:
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
            sys.path.append(self.MASK_RCNN_DIR)

            from .nnmrcnn import InferenceConfig
            self.CONFIG = InferenceConfig(self.NN_MASK_RCNN_CONFIG)

            import mrcnn.model as modellib
            self.MODEL = modellib.MaskRCNN(mode="inference", model_dir=self.LOG_DIR, config=self.CONFIG)

            # graph nodes
            self.INPUT_NODES = ("input_image:0", "input_image_meta:0", "input_anchors:0") # 3 elem
            self.OUTPUT_NODES = ("mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0",
                                 "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0") # 7 elem

    def load(self, model_path):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            self.input_image, self.input_image_meta, self.input_anchors, self.mrcnn_detection, self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask, self.ROI, self.rpn_class, self.rpn_bbox = tf.import_graph_def(
                graph_def, return_elements = self.INPUT_NODES + self.OUTPUT_NODES
            )

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)


    def normalize(self, images):
        res = []
        for image in images:
            res.append(image[..., :3])
        return res;

    def detect(self, images):
        assert len(images) == self.CONFIG.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

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