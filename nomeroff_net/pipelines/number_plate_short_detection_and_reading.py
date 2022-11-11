from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from .number_plate_localization import NumberPlateLocalization
from .number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.tools.image_processing import crop_number_plate_rect_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip


class NumberPlateShortDetectionAndReading(Pipeline, CompositePipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 text_reader_name: str = "eu",
                 text_reader_path: str = "latest",
                 default_lines_count: int = 1,
                 **kwargs):
        self.number_plate_localization = NumberPlateLocalization(
            "number_plate_localization",
            image_loader,
            path_to_model=path_to_model)
        self.text_reader_name = text_reader_name
        self.default_lines_count = default_lines_count
        presets = {
            text_reader_name: {
                "for_regions": [text_reader_name],
                "model_path": text_reader_path
            }
        }
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            default_label=text_reader_name,
            default_lines_count=default_lines_count,
        )
        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_text_reading,
        ]
        super().__init__(task, image_loader, **kwargs)
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        return inputs

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        zones, image_ids = crop_number_plate_rect_zones_from_images(images, images_bboxs)
        region_names = [self.text_reader_name for _ in zones]
        lines = [self.default_lines_count for _ in zones]
        number_plate_text_reading_res = unzip(
            self.number_plate_text_reading(unzip([zones,
                                                  region_names,
                                                  lines]), **forward_parameters))
        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
        else:
            texts = []
        (texts, zones) = group_by_image_ids(image_ids, (texts, zones))
        return unzip([images, images_bboxs,
                      zones, texts])

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs
