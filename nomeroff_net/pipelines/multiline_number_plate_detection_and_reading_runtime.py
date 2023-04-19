"""

"""
from .multiline_number_plate_detection_and_reading import MultilineNumberPlateDetectionAndReading
from nomeroff_net.pipelines.base import RuntimePipeline


class MultilineNumberPlateDetectionAndReadingRuntime(MultilineNumberPlateDetectionAndReading,
                                                     RuntimePipeline):
    """
    Multiline Number Plate Detection and reading runtime
    """

    def __init__(self,
                 *args,
                 **kwargs):
        MultilineNumberPlateDetectionAndReading.__init__(self, *args, **kwargs)
        RuntimePipeline.__init__(self, self.pipelines)
