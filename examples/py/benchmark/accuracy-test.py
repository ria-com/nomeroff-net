import os
import warnings
import argparse
from glob import glob
from _paths import nomeroff_net_dir

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pipeline_name", default="number_plate_detection_and_reading",
                    required=False, type=str, help="Pipeline name")
    ap.add_argument("-l", "--image_loader_name", default="opencv",
                    required=False, type=str, help="Image loader name")
    ap.add_argument("-g", "--images_glob", default="./data/examples/benchmark_oneline_np_images/1.jpeg",
                    required=False, type=str, help="Images glob path")
    ap.add_argument("-f", "--test_file", default="./data/examples/accuracy_test_data_example.json",
                    required=False, type=str, help="Test json file path")
    kwargs = vars(ap.parse_args())
    return kwargs


def main(pipeline_name, image_loader_name, images_glob, test_file, **_):
    number_plate_detection_and_reading = pipeline(pipeline_name,
                                                  image_loader=image_loader_name)
    if os.path.isabs(images_glob):
        image_paths = glob(images_glob)
    else:
        image_paths = glob(os.path.join(nomeroff_net_dir, images_glob))
    result = number_plate_detection_and_reading(image_paths, quality_profile=[5, 1, 0])

    (images, images_bboxs,
     images_points, images_zones, region_ids,
     region_names, count_lines,
     confidences, texts) = unzip(result)

    if os.path.isabs(images_glob):
        test_file = test_file
    else:
        test_file = os.path.join(nomeroff_net_dir, test_file)

    number_plate_detection_and_reading.text_accuracy_test_from_file(
        test_file,
        texts, image_paths,
        images, images_bboxs,
        images_points, images_zones,
        region_ids, region_names,
        count_lines, confidences,
        matplotlib_show=False,
        debug=False,
        md=True
    )


if __name__ == '__main__':
    main(**parse_args())
