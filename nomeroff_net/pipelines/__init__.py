from typing import Any, Dict, Optional, Union

from nomeroff_net.pipelines.base \
    import Pipeline
from nomeroff_net.image_loaders \
    import BaseImageLoader
from nomeroff_net.pipelines.number_plate_localization \
    import NumberPlateLocalization
from nomeroff_net.pipelines.number_plate_bbox_filling \
    import NumberPlateBboxFilling
from nomeroff_net.pipelines.number_plate_key_points_detection \
    import NumberPlateKeyPointsDetection
from nomeroff_net.pipelines.number_plate_key_points_filling \
    import NumberPlateKeyPointsFilling
from nomeroff_net.pipelines.number_plate_classification \
    import NumberPlateClassification
from nomeroff_net.pipelines.number_plate_text_reading \
    import NumberPlateTextReading
from nomeroff_net.pipelines.number_plate_detection_and_reading \
    import NumberPlateDetectionAndReading
from nomeroff_net.pipelines.number_plate_detection_and_reading_v2 \
    import NumberPlateDetectionAndReadingV2
from nomeroff_net.pipelines.number_plate_detection_and_reading_runtime \
    import NumberPlateDetectionAndReadingRuntime
from nomeroff_net.pipelines.number_plate_detection_and_reading_runtime_v2 \
    import NumberPlateDetectionAndReadingRuntimeV2
from nomeroff_net.pipelines.number_plate_short_detection_and_reading \
    import NumberPlateShortDetectionAndReading
from nomeroff_net.pipelines.multiline_number_plate_detection_and_reading \
    import MultilineNumberPlateDetectionAndReading
from nomeroff_net.pipelines.multiline_number_plate_detection_and_reading_runtime \
    import MultilineNumberPlateDetectionAndReadingRuntime


SUPPORTED_TASKS = {
    "multiline_number_plate_detection_and_reading_runtime": {
        "impl": MultilineNumberPlateDetectionAndReadingRuntime,
    },
    "multiline_number_plate_detection_and_reading": {
        "impl": MultilineNumberPlateDetectionAndReading,
    },
    "number_plate_short_detection_and_reading": {
        "impl": NumberPlateShortDetectionAndReading,
    },
    "number_plate_localization": {
        "impl": NumberPlateLocalization,
    },
    "number_plate_bbox_filling": {
        "impl": NumberPlateBboxFilling
    },
    "number_plate_key_points_detection": {
        "impl": NumberPlateKeyPointsDetection
    },
    "number_plate_key_points_filling": {
        "impl": NumberPlateKeyPointsFilling
    },
    "number_plate_classification": {
        "impl": NumberPlateClassification
    },
    "number_plate_text_reading": {
        "impl": NumberPlateTextReading
    },
    "number_plate_detection_and_reading_v2": {
        "impl": NumberPlateDetectionAndReadingV2
    },
    "number_plate_detection_and_reading_runtime_v2": {
        "impl": NumberPlateDetectionAndReadingRuntimeV2
    },
    "number_plate_detection_and_reading": {
        "impl": NumberPlateDetectionAndReading
    },
    "number_plate_detection_and_reading_runtime": {
        "impl": NumberPlateDetectionAndReadingRuntime
    },
}


def check_task(task: str) -> Dict:
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task

    raise KeyError(f"Unknown task {task}, available tasks are {SUPPORTED_TASKS.keys()}")


def pipeline(
    task: str = None,
    image_loader: Optional[Union[str, BaseImageLoader]] = None,
    pipeline_kwargs: Dict[str, Any] = None,
    **kwargs,
) -> Pipeline:
    """
    Args:

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if task is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task"
            "being specified."
            "Please provide a task class"
        )

    # Retrieve the task
    targeted_task = check_task(task)
    pipeline_class = targeted_task["impl"]

    return pipeline_class(task, image_loader, **pipeline_kwargs, **kwargs)
