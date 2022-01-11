import os
from examples.py._paths import current_dir
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

if __name__ == '__main__':
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

    result = number_plate_detection_and_reading([
        os.path.join(current_dir, '../images/example1.jpeg'),
        os.path.join(current_dir, '../images/example2.jpeg'),
    ])

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    print(texts)
