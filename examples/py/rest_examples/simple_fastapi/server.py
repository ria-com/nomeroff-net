"""
Flask REST API

EXAMPLE RUN:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  python3 ./server.py

REQUEST '/version' location: curl 127.0.0.1:8887/version
REQUEST '/detect' location: curl --header "Content-Type: application/json" \
                                 --request POST --data '{"path": "../../../../data/examples/oneline_images/example1.jpeg"}' 127.0.0.1:8887/detect
"""
import os
import cv2
import sys
import traceback
import uvicorn
import ujson
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette_prometheus import PrometheusMiddleware
from starlette_prometheus import metrics
from typing import Dict, List

from _paths import nomeroff_net_dir
from nomeroff_net import __version__
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

print("[INFO], nomeroff net root dir", nomeroff_net_dir)
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading")

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)


@app.get('/version')
def version():
    return __version__


@app.post('/detect_from_bytes')
def detect_from_bytes(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        try:
            img = cv2.imdecode(np.frombuffer(file.file.read(), dtype=np.uint8), 1)
            images.append(img[:, :, ::-1])
        except Exception:
            return ujson.dumps({"error": "There was an error uploading the file(s)"})
        finally:
            file.file.close()
    try:
        result = number_plate_detection_and_reading(images)
        (images, images_bboxs,
         images_points, images_zones, region_ids,
         region_names, count_lines,
         confidences, texts) = unzip(result)
        return ujson.dumps(dict(res=texts))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return ujson.dumps(dict(error=str(e)))


@app.post('/detect')
def detect(data: Dict):
    img_path = data['path']
    try:
        img = cv2.imread(img_path)[:, :, ::-1]
        result = number_plate_detection_and_reading([img])
        (images, images_bboxs,
         images_points, images_zones, region_ids,
         region_names, count_lines,
         confidences, texts) = unzip(result)
        return ujson.dumps(dict(res=texts, img_path=img_path))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return ujson.dumps(dict(error=str(e), img_path=img_path))


if __name__ == '__main__':
    uvicorn.run("server:app",
                host='0.0.0.0',
                port=os.environ.get("PORT", 8887),
                reload=False)
