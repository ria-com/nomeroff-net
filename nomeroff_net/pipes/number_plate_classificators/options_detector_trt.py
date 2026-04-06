"""
python3 -m nomeroff_net.pipes.number_plate_classificators.options_detector_trt \
        -f nomeroff_net/pipes/number_plate_classificators/options_detector_trt.py
"""
import os
import cv2
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from typing import List, Dict, Tuple

from nomeroff_net.tools import modelhub
from nomeroff_net.tools.cuda_primary_context import primary_cuda_context
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
        with primary_cuda_context():
            runtime = trt.Runtime(TRT_LOGGER)
            with open(path_to_model, "rb") as f:
                engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine from {path_to_model}. "
                "This usually happens due to TensorRT version mismatch or corrupted file. "
                "Please rebuild the engine."
            )

        # TRT 10.x API: use get_tensor_mode instead of deprecated binding_is_input
        inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
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
        input_image = np.ascontiguousarray(input_image, dtype=np.float32)
        batch_size = len(input_image)

        with primary_cuda_context():
            context = self.engine.create_execution_context()

            # TRT 10.x: set input shape using tensor name
            context.set_input_shape(
                self.input_name,
                (batch_size, self.color_channels, self.height, self.width),
            )

            stream = cuda.Stream()
            outputs = []
            output_memories = []
            tensor_addrs = {}

            # Allocate input
            input_memory = cuda.mem_alloc(input_image.nbytes)
            cuda.memcpy_htod_async(input_memory, input_image, stream)
            tensor_addrs[self.input_name] = int(input_memory)

            # Allocate outputs using TRT 10.x API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    shape = context.get_tensor_shape(name)
                    dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                    size = trt.volume(shape)
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    tensor_addrs[name] = int(output_memory)
                    outputs.append(output_buffer)
                    output_memories.append(output_memory)

            # Set tensor addresses
            for name, addr in tensor_addrs.items():
                context.set_tensor_address(name, addr)

            # Run inference
            context.execute_async_v3(stream_handle=stream.handle)

            # Copy outputs back to host
            for output_buffer, output_memory in zip(outputs, output_memories):
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)

            stream.synchronize()

        if batch_size == 1:
            outputs = [o.reshape(1, -1) for o in outputs]

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
        "./data/model_repository/pruned_engines/options_pruned.trt"))

    image_paths = [
        os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png"),
        os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png")
    ]
    images = [cv2.imread(image_path) for image_path in image_paths]
    y = det.predict(images)
    print("y", y)
