import os
import warnings
from glob import glob
from _paths import current_dir

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

    image_paths = glob(os.path.join(current_dir, "../images/*"))
    result = number_plate_detection_and_reading(image_paths, quality_profile=[3, 1, 0])

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    number_plate_detection_and_reading.text_accuracy_test_from_file(
        os.path.join(current_dir, "../accuracy_test_data.json"),
        texts, image_paths,
        images, images_bboxs,
        images_points, images_zones,
        region_ids, region_names,
        count_lines, confidences,
        matplotlib_show=False)
