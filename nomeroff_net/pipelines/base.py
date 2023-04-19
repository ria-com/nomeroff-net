"""base pipeline constructor class


The module contains the following functions:
- `may_by_empty_method(func)`
- `empty_method`

The module contains the following classes:

"""
import os
import time
import ujson
import cv2
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from collections import Counter
from nomeroff_net.tools import promise_all
from nomeroff_net.tools import chunked_iterable
from nomeroff_net.image_loaders import BaseImageLoader, DumpyImageLoader, image_loaders_map


def may_by_empty_method(func):
    """
    if in your pipeline you want to off some thing
    """
    func.is_empty = False
    return func


def empty_method(func):
    """
    if in your pipeline you want to off some thing
    """
    func.is_empty = True
    return func


class AccuracyTestPipeline(object):
    """
    Accuracy Test Pipeline Base Class
    """
    @staticmethod
    def text_accuracy_test(true_images_texts, predicted_images_texts,
                           img_paths, images, images_bboxs,
                           images_points, images_zones,
                           region_ids, region_names,
                           count_lines, confidences,
                           matplotlib_show=False,
                           debug=True,
                           md=False):
        """
        TODO: write description
        """
        n_good = 0
        n_bad = 0
        for predicted_image_texts, \
            true_image_texts, \
            image, image_bboxs, \
            image_points, image_zones, \
            image_region_ids, image_region_names, \
            image_count_lines, image_confidences, \
            img_path in zip(predicted_images_texts,
                            true_images_texts,
                            images, images_bboxs,
                            images_points, images_zones,
                            region_ids, region_names,
                            count_lines, confidences,
                            img_paths):
            for true_image_text in true_image_texts:
                if true_image_text in predicted_image_texts:
                    message = f"+ NAME:{os.path.basename(img_path)} " \
                              f"TRUE:{true_image_text} " \
                              f"PREDICTED:{predicted_image_texts}"
                    message = message if md else colored(message, 'green')
                    n_good += 1
                else:
                    message = f"- NAME:{os.path.basename(img_path)} " \
                              f"TRUE:{true_image_text} " \
                              f"PREDICTED:{predicted_image_texts}"
                    message = message if md else colored(message, 'red')
                    n_bad += 1
                print(message)
                if debug:
                    print("[INFO] images_bboxs", image_bboxs)
                    print("[INFO] image_points", image_points)

                if matplotlib_show:
                    image = image.astype(np.uint8)
                    for cntr in image_points:
                        cntr = np.array(cntr, dtype=np.int32)
                        cv2.drawContours(image, [cntr], -1, (0, 0, 255), 2)
                    for target_box in image_bboxs:
                        cv2.rectangle(image,
                                      (int(target_box[0]), int(target_box[1])),
                                      (int(target_box[2]), int(target_box[3])),
                                      (0, 255, 0),
                                      1)
                    plt.imshow(image)
                    plt.show()

                if debug:
                    print("[INFO] image_region_ids", image_region_ids)
                    print("[INFO] image_region_names", image_region_names)
                    print("[INFO] image_count_lines", image_count_lines)
                    print("[INFO] image_confidences", image_confidences)
                if matplotlib_show:
                    for zone in image_zones:
                        plt.imshow(zone)
                        plt.show()
        total = n_good + n_bad
        print(f"TOTAL GOOD: {n_good / total}")
        print(f"TOTAL BAD: {n_bad / total}")

    def text_accuracy_test_from_file(self, accuracy_test_data_file, predicted_images_texts,
                                     img_paths, images, images_bboxs,
                                     images_points, images_zones,
                                     region_ids, region_names,
                                     count_lines, confidences,
                                     matplotlib_show=False,
                                     debug=True,
                                     md=False):
        """
        TODO: write description
        """
        with open(accuracy_test_data_file) as f:
            accuracy_test_data = ujson.load(f)
        true_images_texts = []
        for image_path in img_paths:
            key = os.path.basename(image_path)
            if key in accuracy_test_data:
                true_images_texts.append(accuracy_test_data[key])
            else:
                true_images_texts.append([])
        self.text_accuracy_test(true_images_texts, predicted_images_texts,
                                img_paths, images, images_bboxs,
                                images_points, images_zones,
                                region_ids, region_names,
                                count_lines, confidences,
                                matplotlib_show=matplotlib_show,
                                debug=debug,
                                md=md)


class Pipeline(AccuracyTestPipeline):
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
        image_loader: Optional[Union[str, BaseImageLoader]] = None,
        **kwargs,
    ):
        """
        TODO: write description
        """
        self.task = task
        self.image_loader = self._init_image_loader(image_loader)

        self._preprocess_params, self._forward_params, self._postprocess_params = self.sanitize_parameters(**kwargs)

    @staticmethod
    def _init_image_loader(image_loader):
        """
        TODO: write description
        """
        if image_loader is None:
            image_loader_class = DumpyImageLoader
        elif type(image_loader) == str:
            image_loader_class = image_loaders_map.get(image_loader, None)
            if image_loader is None:
                raise ValueError(f"{image_loader} not in {image_loaders_map.keys()}.")
        elif issubclass(image_loader, BaseImageLoader):
            image_loader_class = image_loader
        else:
            raise TypeError(f"The image_loader type must by in None, BaseImageLoader, str")
        return image_loader_class()

    def sanitize_parameters(self, **pipeline_parameters):
        """
        sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".
        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    @abstractmethod
    @may_by_empty_method
    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Dict[str, Any]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
        `_forward` to run properly.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    @may_by_empty_method
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Dict[str, Any]:
        """
        _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.
        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("forward not implemented")

    @abstractmethod
    @may_by_empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    def __call__(self, inputs, **kwargs):
        """
        TODO: write description
        """
        return self.call(inputs, **kwargs)

    def call(self, inputs, batch_size=1, num_workers=1, **kwargs):
        """
        TODO: write description
        """
        kwargs["batch_size"] = batch_size
        kwargs["num_workers"] = num_workers
        preprocess_params, forward_params, postprocess_params = self.sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        if num_workers < 0 or num_workers > batch_size:
            raise ValueError("num_workers must by grater 0 and less or equal batch_size")
        outputs = self.run_multi(inputs, batch_size, num_workers,
                                 preprocess_params, forward_params, postprocess_params)
        return outputs

    @staticmethod
    def process_worker(func, inputs, params, num_workers=1):
        """
        TODO: write description
        """
        if num_workers == 1:
            return func(inputs, **params)
        promises_outputs = []
        promise_all_args = []
        for chunk_inputs in chunked_iterable(inputs, num_workers):
            for inp in chunked_iterable(chunk_inputs, 1):
                promise_all_args.append(
                    {
                        "function": func,
                        "args": [inp],
                        "kwargs": params
                    }
                )
            # print(f"RUN promise_all {func} functions {len(promise_all_args)}")
            promise_outputs = promise_all(promise_all_args)
            promises_outputs.append(promise_outputs)

        outputs = []
        for promise_output in promises_outputs:
            for chunk in promise_output:
                for item in chunk:
                    outputs.append(item)
        return outputs

    def run_multi(self, inputs, batch_size, num_workers, preprocess_params, forward_params, postprocess_params):
        """
        TODO: write description
        """
        outputs = []
        for chunk_inputs in chunked_iterable(inputs, batch_size):
            chunk_outputs = self.run_single(chunk_inputs, num_workers,
                                            preprocess_params, forward_params, postprocess_params)
            for output in chunk_outputs:
                outputs.append(output)
        return outputs

    def run_single(self, inputs, num_workers, preprocess_params, forward_params, postprocess_params):
        """
        TODO: write description
        """
        _inputs = inputs
        if not hasattr(self.preprocess, "is_empty") or not self.preprocess.is_empty:
            _inputs = self.process_worker(self.preprocess, _inputs, preprocess_params, num_workers)
        if not hasattr(self.forward, "is_empty") or not self.forward.is_empty:
            _inputs = self.forward(_inputs, **forward_params)
        if not hasattr(self.postprocess, "is_empty") or not self.postprocess.is_empty:
            _inputs = self.process_worker(self.postprocess, _inputs, postprocess_params, num_workers)
        return _inputs


class CompositePipeline(object):
    """
    Composite pipelines Pipeline Base Class
    """
    def __init__(self, pipelines):
        """
        TODO: write description
        """
        self.pipelines = pipelines

    def sanitize_parameters(self, **kwargs):
        """
        TODO: write description
        """
        forward_parameters = {}
        for key in kwargs:
            if key == "batch_size":
                forward_parameters["batch_size"] = kwargs["batch_size"]
            if key == "num_workers":
                forward_parameters["num_workers"] = kwargs["num_workers"]
        for pipeline in self.pipelines:
            for dict_params in pipeline.sanitize_parameters(**kwargs):
                forward_parameters.update(dict_params)
        return {}, forward_parameters, {}


class RuntimePipeline(object):
    """
    Runtime Pipeline Base Class
    """

    default_input_names = None

    def __init__(self, pipelines):
        """
        TODO: write description
        """
        self.pipelines = pipelines
        self.time_stat = Counter()
        self.count_stat = Counter()

        self.call = self.timeit(self.__class__.__name__)(self.call)
        for pipeline in self.pipelines:
            pipeline.call = self.timeit(pipeline.__class__.__name__)(pipeline.call)

    def timeit(self, tag):
        """
        TODO: write description
        """
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

    def clear_stat(self):
        """
        TODO: write description
        """
        self.time_stat = Counter()
        self.count_stat = Counter()

    def get_timer_stat(self, count_processed_images):
        """
        TODO: write description
        """
        timer_stat = {}
        for key in self.count_stat:
            timer_stat[key] = self.time_stat[key] / count_processed_images
        return timer_stat
