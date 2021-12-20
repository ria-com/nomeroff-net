import time
from abc import abstractmethod
from typing import Any, Dict
from collections import Counter
from nomeroff_net.image_loaders import BaseImageLoader, DumpyImageLoader, image_loaders_map


class Pipeline(object):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.
    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:
        Input -> Pre-Processing -> Model Inference -> Post-Processing (task dependent) -> Output
    Pipeline supports running on CPU or GPU through the device argument (see below).
    """

    default_input_names = None

    def __init__(
        self,
        task: str = "",
        image_loader: BaseImageLoader or str = None,
        **kwargs,
    ):
        self.task = task
        self.image_loader = self._init_image_loader(image_loader)

        self._preprocess_params, self._forward_params, self._postprocess_params = self.sanitize_parameters(**kwargs)

    @staticmethod
    def _init_image_loader(image_loader):
        if image_loader is None:
            image_loader_class = DumpyImageLoader
        elif type(image_loader) == str:
            image_loader_class = image_loaders_map.get(image_loader, None)
            if image_loader is None:
                raise ValueError(f"{image_loader} not in {image_loaders_map.keys()}.")
        elif type(image_loader) == BaseImageLoader:
            image_loader_class = image_loader
        else:
            raise TypeError(f"The image_loader type must by in None, BaseImageLoader, str")
        return image_loader_class()

    @abstractmethod
    def sanitize_parameters(self, **pipeline_parameters):
        """
        sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".
        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Dict[str, Any]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
        `_forward` to run properly.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Dict[str, Any]:
        """
        _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.
        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    def __call__(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        """
        TODO: speed up using num_workers and batch_size
        """

        preprocess_params, forward_params, postprocess_params = self.sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # This function should become `get_iterator` again, this is a temporary
        # easy solution.
        for input_ in inputs:
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)


class RuntimePipeline(object):
    """
    Runtime Pipeline
    """

    default_input_names = None

    def __init__(self, pipelines):
        self.pipelines = pipelines
        self.time_stat = Counter()
        self.count_stat = Counter()

        self.call = self.timeit(self.__class__.__name__)(self.call)
        for pipeline in self.pipelines:
            pipeline.call = self.timeit(pipeline.__class__.__name__)(pipeline.call)

    def timeit(self, tag):
        def wrapper(method):
            def timed(*args, **kw):
                ts = time.time()
                result = method(*args, **kw)
                te = time.time()
                self.time_stat[f'{tag}.{method.__name__}'] += te - ts
                self.count_stat[f'{tag}.{method.__name__}'] += 1
                return result
            return timed
        return wrapper

    def get_timer_stat(self, batch=1):
        timer_stat = {}
        for key in self.count_stat:
            timer_stat[key] = (self.time_stat[key] / (self.count_stat[key] or 1)) / batch
        return timer_stat
