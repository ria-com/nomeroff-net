import os
from glob import glob

from _paths import nomeroff_net_dir
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

if __name__ == '__main__':
    multiline_number_plate_detection_and_reading = pipeline("multiline_number_plate_detection_and_reading",
                                                            image_loader="opencv")

    result = multiline_number_plate_detection_and_reading(glob(os.path.join(nomeroff_net_dir,
                                                                            './data/examples/multiline_images/*')))

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    print(texts)
