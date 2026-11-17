"""
Number plate recognition on a video file, a camera or a stream.

python3 tutorials/py/inference/video-demo.py --source video.mp4
python3 tutorials/py/inference/video-demo.py --source rtsp://user:pass@host/stream --every 5
python3 tutorials/py/inference/video-demo.py --source 0                     # webcam
python3 tutorials/py/inference/video-demo.py --source video.mp4 --output out.mp4
"""
import os
import sys
import argparse
from collections import Counter

import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
nomeroff_net_dir = os.path.join(current_dir, "../../../")
sys.path.append(nomeroff_net_dir)

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip


def is_cut_by_frame(bbox, w, h, edge=5):
    """Plates cut by the frame edge are read unreliably, skip them."""
    x1, y1, x2, y2 = bbox[:4]
    return min(x1, y1, w - x2, h - y2) < edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video file, stream url or camera index")
    parser.add_argument("--every", type=int, default=3, help="process every N-th frame")
    parser.add_argument("--batch", type=int, default=4, help="frames per batch")
    parser.add_argument("--output", default=None, help="save video with drawn results")
    args = parser.parse_args()

    # create the pipeline once, loading the models is the slowest part
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
                                                  image_loader=None)  # frames are numpy arrays

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {args.source}")
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.every, (w, h))

    seen = Counter()  # plate text -> number of frames it was read on
    frames, frame_ids, frame_id = [], [], 0

    def process(frames, frame_ids):
        rgb = [f[..., ::-1] for f in frames]  # OpenCV BGR -> RGB
        result = number_plate_detection_and_reading(rgb)
        (_, images_bboxs, _, _, _, _, _, _, texts) = unzip(result)
        for frame, fid, bboxs, plate_texts in zip(frames, frame_ids, images_bboxs, texts):
            h, w = frame.shape[:2]
            for bbox, text in zip(bboxs, plate_texts):
                if not text or is_cut_by_frame(bbox, w, h):
                    continue
                seen[text] += 1
                print(f"frame {fid}: {text}")
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if writer is not None:
                writer.write(frame)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % args.every == 0:
            frames.append(frame)
            frame_ids.append(frame_id)
            if len(frames) == args.batch:
                process(frames, frame_ids)
                frames, frame_ids = [], []
        frame_id += 1
    if frames:
        process(frames, frame_ids)

    cap.release()
    if writer is not None:
        writer.release()

    # a plate seen on several frames is much more reliable than a single reading
    print("\nPlates (text: frames):")
    for text, count in seen.most_common():
        print(f"{text}: {count}")


if __name__ == "__main__":
    main()
