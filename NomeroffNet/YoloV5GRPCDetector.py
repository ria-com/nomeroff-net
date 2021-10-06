import tritonclient.grpc as grpcclient
from tritonclient import utils
import tritonclient.utils.shared_memory as shm

from typing import List
import numpy as np
import sys
import torch

sys.path.append("../")
from NomeroffNet.YoloV5Detector import (letterbox, non_max_suppression, scale_coords)


class YoloV5GRPCDetector(object):
    """
    YoloV5 GRPC Example
    """
    def __init__(self, 
                 url="localhost:8001",
                 model_name="yolov5",
                 model_version="",
                 verbose=False) -> None:
        self.triton_client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

        # To make sure no shared memory regions are registered with the server.
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()
        
        # Yolo model takes 1 input tensors dims [1, 3, 640, 640]
        # each and returns 4 output tensors 
        # dims: [1,25200,6]
        # dims: [1,3,80,80,6]
        # dims: [1,3,40,40,6]
        # dims: [1,3,20,20,6]
        self.model_name = model_name
        self.model_version = model_version
        
        # Create the data for the input and 4 outputs tensors
        input_images = np.zeros((1, 3, 640, 640), dtype=np.float32)
        output = np.zeros((1, 25200, 6), dtype=np.float32)
        output_397 = np.zeros((1,3,80,80,6), dtype=np.float32)
        output_458 = np.zeros((1,3,40,40,6), dtype=np.float32)
        output_519 = np.zeros((1,3,20,20,6), dtype=np.float32)
        
        # Calc input/output tensors sizes
        input_images_byte_size = input_images.size * input_images.itemsize
        output_byte_size = output.size * output.itemsize
        output_397_byte_size = output_397.size * output_397.itemsize
        output_458_byte_size = output_458.size * output_458.itemsize
        output_519_byte_size = output_519.size * output_519.itemsize
        
        # Create outputs in Shared Memory and store shared memory handles
        self.output_handle = shm.create_shared_memory_region("output",
                                                         "/output",
                                                         output_byte_size)
        self.output_397_handle = shm.create_shared_memory_region("output_397",
                                                         "/output_397",
                                                         output_397_byte_size)
        self.output_458_handle = shm.create_shared_memory_region("output_458",
                                                         "/output_458",
                                                         output_458_byte_size)
        self.output_519_handle = shm.create_shared_memory_region("output_519",
                                                         "/output_519",
                                                         output_519_byte_size)
        
        # Register outputs shared memory with Triton Server
        self.triton_client.register_system_shared_memory("output",
                                                   "/output",
                                                    output_byte_size)
        self.triton_client.register_system_shared_memory("output_397",
                                                   "/output_397",
                                                    output_397_byte_size)
        self.triton_client.register_system_shared_memory("output_458",
                                                   "/output_458",
                                                    output_458_byte_size)
        self.triton_client.register_system_shared_memory("output_519",
                                                   "/output_519",
                                                    output_519_byte_size)
        
        # Create inputs in Shared Memory and store shared memory handles
        self.input_images_handle = shm.create_shared_memory_region("images",
                                                         "/images",
                                                         input_images_byte_size)
        # Register inputs shared memory with Triton Server
        self.triton_client.register_system_shared_memory("images",
                                                   "/images",
                                                    input_images_byte_size)
        
        
        # Set the parameters to use data from shared memory
        self.inputs = []
        self.inputs.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
        self.inputs[-1].set_shared_memory("images", input_images_byte_size)
        
        self.outputs = []
        self.outputs.append(grpcclient.InferRequestedOutput('output'))
        self.outputs[-1].set_shared_memory("output", output_byte_size)
        
        self.predict(input_images)
        
    def cleanup(self):
        self.triton_client.unregister_system_shared_memory()
        shm.destroy_shared_memory_region(self.input_images_handle)
        shm.destroy_shared_memory_region(self.output_handle)
        shm.destroy_shared_memory_region(self.output_397_handle)
        shm.destroy_shared_memory_region(self.output_458_handle)
        shm.destroy_shared_memory_region(self.output_519_handle)
        
    def __enter__(self):
        return self
    
    def __del__(self):
        self.cleanup()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def predict(self, input_images):
        # Put input data values into shared memory
        shm.set_shared_memory_region(self.input_images_handle, [input_images])
        
        results = self.triton_client.infer(model_name=self.model_name,
                                           inputs=self.inputs,
                                           outputs=self.outputs)
        # Read results from the shared memory.
        output = results.get_output("output")
        output_data = shm.get_contents_as_numpy(
            self.output_handle, utils.triton_to_np_dtype(output.datatype),
            output.shape)
        
        return output_data

    def grpc_detect(self, img: np.ndarray, img_size: int = 640, stride: int = 32, min_accuracy: float = 0.5) -> List:
        """
        TODO: input img in BGR format, not RGB; To Be Implemented in release 2.2
        """
        # normalize
        img_shape = img.shape
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img = img/255.0  # 0 - 255 to 0.0 - 1.0
        img = img.reshape([1, *img.shape])

        pred = self.predict(img)
        # Apply NMS
        pred = non_max_suppression(torch.from_numpy(pred))
        res = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                res.append(det.cpu().detach().numpy())
        if len(res):
            return [[x1, y1, x2, y2, acc, b] for x1, y1, x2, y2, acc, b in res[0] if acc > min_accuracy]
        else:
            return []
