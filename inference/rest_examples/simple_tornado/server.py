"""
Tornado REST API

EXAMPLE RUN:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 ./server.py

REQUEST '/version' location: curl 127.0.0.1:8888/version
REQUEST '/detect' location: curl --header "Content-Type: application/json" \
                                 --request POST --data '{"path": "../images/example1.jpeg"}' 127.0.0.1:8888/detect
"""
import sys
import traceback
import logging
import ujson
import tornado.ioloop
import tornado.web
from tornado.web import Application
import tornado.web

from _paths import nomeroff_net_dir
from nomeroff_net import __version__
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

print("[INFO], nomeroff net root dir", nomeroff_net_dir)
number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")


hn = logging.NullHandler()
hn.setLevel(logging.DEBUG)
logging.getLogger("tornado.access").addHandler(hn)
logging.getLogger("tornado.access").propagate = False


class GetVersion(tornado.web.RequestHandler):
    def get(self):
        self.write(__version__)


class GetMask(tornado.web.RequestHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        img_path = data['path']
        try:
            result = number_plate_detection_and_reading([img_path])
            (images, images_bboxs,
             images_points, images_zones, region_ids,
             region_names, count_lines,
             confidences, texts) = unzip(result)
            res = ujson.dumps(dict(res=texts, img_path=img_path))
            self.write(res)
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
            res = ujson.dumps(dict(error=str(e), img_path=img_path))
            self.write(res)


if __name__ == '__main__':
    app = Application([
            (r"/detect", GetMask),
            (r"/version", GetVersion),
        ])

    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
