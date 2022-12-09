"""
Flask REST API

EXAMPLE RUN:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  python3 ./simple_flask-rest.py

REQUEST '/version' location: curl 127.0.0.1:8888/version
REQUEST '/detect' location: curl --header "Content-Type: application/json" \
                                 --request POST --data '{"path": "../images/example1.jpeg"}' 127.0.0.1:8888/detect
"""
import os
import sys
import traceback
import uvicorn
import ujson
from fastapi import FastAPI
from starlette_prometheus import PrometheusMiddleware
from starlette_prometheus import metrics
from typing import Dict

from _paths import nomeroff_net_dir
from nomeroff_net import __version__
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

print("[INFO], nomeroff net root dir", nomeroff_net_dir)
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)


@app.get('/version')
def version():
    return __version__


@app.post('/detect')
def detect(data: Dict):
    img_path = data['path']
    try:
        result = number_plate_detection_and_reading([img_path])
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
                port=os.environ.get("PORT", 8888),
                reload=False)
