import os
import warnings
import argparse
from glob import glob

from _paths import nomeroff_net_dir
from nomeroff_net import pipeline
import faulthandler


faulthandler.enable()

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pipeline_name", default="number_plate_detection_and_reading_runtime",
                    required=False, type=str, help="Pipeline name")
    ap.add_argument("-l", "--image_loader_name", default="opencv", # Try 'turbo' for faster performance.
                    required=False, type=str, help="Image loader name")
    ap.add_argument("-g", "--images_glob", default="./data/examples/benchmark_oneline_np_images/1.jpeg",
                    required=False, type=str, help="Images glob path")
    ap.add_argument("-n", "--num_run", default=1,
                    required=False, type=int, help="Number loops")
    ap.add_argument("-b", "--batch_size", default=1,
                    required=False, type=int, help="Batch size")
    ap.add_argument("-w", "--num_workers", default=1,
                    required=False, type=int, help="Number worker for parallel processing "
                                                   "preprocess and postprocess functions")
    kwargs = vars(ap.parse_args())
    return kwargs


def main(pipeline_name, image_loader_name, images_glob,
         num_run, batch_size, num_workers,  **_):
    number_plate_detection_and_reading = pipeline(
        pipeline_name,
        image_loader=image_loader_name
    )

    if os.path.isabs(images_glob):
        images = glob(images_glob)
    else:
        images = glob(os.path.join(nomeroff_net_dir, images_glob))

    number_plate_detection_and_reading.clear_stat()
    for i in range(num_run):
        number_plate_detection_and_reading(images,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
    timer_stat = number_plate_detection_and_reading.get_timer_stat(len(images) * num_run)
    timer_stat["count_photos"] = len(images)

    # print timer stat result
    print(f"Processed {timer_stat['count_photos']} photos")
    print(f"One photo process {timer_stat['NumberPlateDetectionAndReadingRuntime.call']} seconds")
    print()
    print(f"detect_bbox_time_all {timer_stat['NumberPlateLocalization.call']} per one photo")
    print(f"craft_time_all {timer_stat['NumberPlateKeyPointsDetection.call']} per one photo")
    print(f"classification_time_all {timer_stat['NumberPlateClassification.call']} per one photo")
    print(f"ocr_time_all {timer_stat['NumberPlateTextReading.call']} per one photo")


if __name__ == '__main__':
    main(**parse_args())
