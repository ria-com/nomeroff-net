import os
from _paths import nomeroff_net_dir
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

if __name__ == '__main__':
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

    result = number_plate_detection_and_reading([
        os.path.join(nomeroff_net_dir, './data/examples/oneline_images/example1.jpeg'),
    ])

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    # (['AC4921CB'], ['RP70012', 'JJF509'])
    print(texts)
