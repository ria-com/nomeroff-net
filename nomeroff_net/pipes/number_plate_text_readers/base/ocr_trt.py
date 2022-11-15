"""
python3 -m nomeroff_net.text_detectors.base.ocr_trt -f nomeroff_net/text_detectors/base/ocr_trt.py
"""
from typing import List, Any, Dict
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import threading
import numpy as np
import torch
import cv2
import os

from nomeroff_net.tools import modelhub
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

        stream = cuda.Stream()
        trt_logger = trt.Logger(trt.Logger.INFO)

        with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        context = self.engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = self.engine.max_batch_size

        return self.engine
    
    def run_engine(self, batch_input_image):
        threading.Thread.__init__(self)

        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings


        batch_input_image = np.ascontiguousarray(batch_input_image)
        
        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())

        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()

        output = np.array(host_outputs)
        output = output.reshape((self.label_length, len(output), self.letters_max))
        return output

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
