"""
python3 -m nomeroff_net.pipes.number_plate_classificators.options_detector_trt \
        -f nomeroff_net/pipes/number_plate_classificators/options_detector_trt.py
"""
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from typing import List, Dict, Tuple

from nomeroff_net.tools import modelhub
from nomeroff_net.pipes.number_plate_classificators.options_detector import OptionsDetector
from nomeroff_net.tools.image_processing import normalize_img

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class OptionsDetectorTrt(OptionsDetector):
    """
    TODO: describe class
    """
    def __init__(self, options: Dict = None) -> None:
        OptionsDetector.__init__(self, options)
        self.engine = None
        self.input_name = None

    def load_model(self, path_to_model):
        assert os.path.exists(path_to_model)
        with open(path_to_model, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        inputs = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                inputs.append(binding)
        self.input_name = inputs[0]

    def is_loaded(self) -> bool:
        if self.engine is None:
            return False
        return True

    def load(self, path_to_model: str = "latest", options: Dict = None):
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.__dict__.update(options)

        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("numberplate_options_trt")
            path_to_model = model_info["path"]
            self.class_region = model_info["class_region"]
            self.count_lines = model_info["count_lines"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.get_classname(),
                                                        "numberplate_options_trt")
            path_to_model = model_info["path"]
        return self.load_model(path_to_model)

    def predict(self, imgs: List[np.ndarray], return_acc: bool = False) -> Tuple:
        """
        Predict options(region, count lines) by numberplate images
        """
        region_ids, count_lines, confidences, predicted = self.predict_with_confidence(imgs)
        if return_acc:
            return region_ids, count_lines, predicted
        return region_ids, count_lines
    
    def run_engine(self, input_image):
        with self.engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(self.engine.get_binding_index(self.input_name),
                                      (
                                          len(input_image), 
                                          self.color_channels, 
                                          self.height,
                                          self.width
                                      ))
            input_image = np.array(input_image)
            # Allocate host and device buffers
            bindings = []
            outputs = []
            outputs_memory = []
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(input_image)
                    input_memory = cuda.mem_alloc(input_image.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))
                    outputs.append(output_buffer)
                    outputs_memory.append(output_memory)

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # Transfer prediction output from the GPU.
            for output, output_memory in zip(outputs, outputs_memory):
                cuda.memcpy_dtoh_async(output, output_memory, stream)

            # Synchronize the stream
            stream.synchronize()
        if len(input_image) == 1:
            for i in range(len(outputs)):
                outputs[i] = [outputs[i]]
        return outputs

    def predict_with_confidence(self, imgs: List[np.ndarray]) -> Tuple:
        """
        Predict options(region, count lines) with confidence by numberplate images
        """
        if not len(imgs):
            return [], [], [], []
        xs = np.zeros((len(imgs), self.color_channels, self.height, self.width)).astype('float32')

        for i, img in enumerate(imgs):
            x = normalize_img(img)
            x = np.moveaxis(x, 2, 0)
            xs[i, :, :, :] = x

        predicted = self.run_engine(xs)
        confidences, region_ids, count_lines = self.unzip_predicted(predicted)
        count_lines = self.custom_count_lines_id_to_all_count_lines(count_lines)
        return region_ids, count_lines, confidences, predicted

    def forward(self, inputs):
        model_output = self.run_engine(inputs)
        return model_output


if __name__ == "__main__":
    det = OptionsDetectorTrt()
    det.load(os.path.join(
        os.getcwd(),
        "./data/model_repository/numberplate_options/1/model.trt"))

    image_paths = [
        os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png"),
        os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png")
    ]
    images = [cv2.imread(image_path) for image_path in image_paths]
    y = det.predict(images)
    print("y", y)
