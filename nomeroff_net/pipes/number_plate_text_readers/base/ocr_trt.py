"""
python3 -m nomeroff_net.text_detectors.base.ocr_trt -f nomeroff_net/text_detectors/base/ocr_trt.py
"""
from typing import List, Any, Dict
import pycuda.driver as cuda
import tensorrt as trt
import threading
import numpy as np
import torch
import cv2
import os

from nomeroff_net.tools import modelhub
from nomeroff_net.tools.cuda_primary_context import primary_cuda_context
from nomeroff_net.tools.image_processing import normalize_img
from nomeroff_net.tools.ocr_tools import decode_batch
from .ocr import OCR


class OcrTrt(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.ort_session = None
        self.input_name = None

    def is_loaded(self) -> bool:
        if self.ort_session is None:
            return False
        return True

    def load_model(self, engine_file_path, device=0):
        assert os.path.exists(engine_file_path)

        with primary_cuda_context():
            trt_logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(trt_logger)
            with open(engine_file_path, "rb") as f:
                engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine from {engine_file_path}. "
                "This usually happens due to TensorRT version mismatch or corrupted file. "
                "Please rebuild the engine."
            )

        self.letters_max = len(self.letters) + 1
        self.label_length = self.max_text_len

        with primary_cuda_context():
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

        # TRT 10.x: use named tensor API
        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(name)
                self.input_h = shape[-2]
                self.input_w = shape[-1]
                self.input_name = name
            else:
                self.output_names.append(name)

        return self.engine

    def run_engine(self, batch_input_image):
        batch_input_image = np.ascontiguousarray(batch_input_image, dtype=np.float32)
        batch_size = batch_input_image.shape[0]

        with primary_cuda_context():
            context = self.context
            stream = self.stream

            # TRT 10.x: set input shape
            context.set_input_shape(
                self.input_name,
                (batch_size, self.color_channels, self.input_h, self.input_w)
            )

            tensor_addrs = {}

            # Allocate and copy input
            input_memory = cuda.mem_alloc(batch_input_image.nbytes)
            cuda.memcpy_htod_async(input_memory, batch_input_image, stream)
            tensor_addrs[self.input_name] = int(input_memory)

            # Allocate output(s)
            host_outputs = []
            output_memories = []
            output_shapes = []
            for name in self.output_names:
                shape = context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape)
                host_buf = cuda.pagelocked_empty(size, dtype)
                cuda_buf = cuda.mem_alloc(host_buf.nbytes)
                tensor_addrs[name] = int(cuda_buf)
                host_outputs.append(host_buf)
                output_memories.append(cuda_buf)
                output_shapes.append(tuple(shape))

            # Set all tensor addresses
            for name, addr in tensor_addrs.items():
                context.set_tensor_address(name, addr)

            # Run inference
            context.execute_async_v3(stream_handle=stream.handle)

            # Copy outputs back
            for host_buf, cuda_buf in zip(host_outputs, output_memories):
                cuda.memcpy_dtoh_async(host_buf, cuda_buf, stream)

            stream.synchronize()

        if not host_outputs:
            return np.array([])

        return host_outputs[0].reshape(output_shapes[0])

    def load(self, path_to_model: str = "latest", options: Dict = None):
        """
        TODO: describe method
        """
        if options is None:
            options = dict()
        self.__dict__.update(options)
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.get_classname())
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model,
                                                        self.get_classname(),
                                                        self.get_classname())
            path_to_model = model_info["path"]
        return self.load_model(path_to_model)
    
    def preprocess(self, imgs):
        xs = np.zeros((len(imgs), self.color_channels, self.height, self.width)).astype('float32')
        for i, img in enumerate(imgs):
            x = normalize_img(img,
                              width=self.width,
                              height=self.height)
            x = np.moveaxis(x, 2, 0)
            xs[i, :, :, :] = x
        return xs

    def predict(self, xs: List, return_acc: bool = False) -> Any:
        if not len(xs):
            return ([], []) if return_acc else []
        net_out_value = self.run_engine(xs)
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        if return_acc:
            if len(net_out_value):
                net_out_value = np.array(net_out_value)
                net_out_value = net_out_value.reshape((net_out_value.shape[1],
                                                       net_out_value.shape[0],
                                                       net_out_value.shape[2]))
            return pred_texts, net_out_value
        return pred_texts

    def forward(self, xs, return_acc: bool = False):
        if not len(xs):
            return ([], []) if return_acc else []
        net_out_value = self.run_engine(xs)
        return net_out_value

    def postprocess(self, net_out_value):
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        return pred_texts


if __name__ == "__main__":
    det = OcrTrt()
    det.letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                        "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    det.max_text_len = 9
    det.max_plate_length = 9
    det.letters_max = len(det.letters)+1
    det.init_label_converter()

    det.load(os.path.join(
        os.getcwd(),
        "./data/model_repository/ocr-eu/1/model.trt"))

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)

    image_path = os.path.join(os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png")
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)
