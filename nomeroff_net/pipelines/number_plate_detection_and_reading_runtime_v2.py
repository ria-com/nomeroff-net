from .number_plate_detection_and_reading_v2 import NumberPlateDetectionAndReadingV2
from nomeroff_net.pipelines.base import RuntimePipeline


class NumberPlateDetectionAndReadingRuntimeV2(NumberPlateDetectionAndReadingV2, RuntimePipeline):
    """
    Number Plate Detection and reading runtime
    """

    def __init__(self, *args, **kwargs):
        NumberPlateDetectionAndReadingV2.__init__(self, *args, **kwargs)
        RuntimePipeline.__init__(self, [])
