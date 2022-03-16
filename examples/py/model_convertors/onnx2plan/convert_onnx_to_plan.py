import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt


# logger to capture errors, warnings, and other information during the build and inference phases
logger = trt.Logger()


def build_engine(onnx_file_path, engine_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    for idx in range(parser.num_errors):
        raise Exception(parser.get_error(idx))
    print('Completed parsing of ONNX file')

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20  # 1 MiB
    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)


def load(engine_file_path):
    runtime = trt.Runtime(logger)
    with open(engine_file_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    return engine, context


def inference(engine_file_path):
    engine, context = load(engine_file_path)

    # get sizes of input and output and allocate memory required for input data and for output data
    device_input = None
    device_output, host_output = None, None
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    assert device_input
    assert device_output

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # preprocess input data
    host_input = np.array(np.random.rand(1, 3, 50, 200), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    print(output_data)


def __main__():
    pass
