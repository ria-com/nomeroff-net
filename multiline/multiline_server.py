"""CURL EXAMPLE
!flask/bin/python
curl 'http://localhost:5010/multi_to_one_line_converter' -X POST -H 'Content-Type: application/json' -d '{"img": ... }
"""
import os
import sys

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')
sys.path.append(NOMEROFF_NET_DIR)

from flask import Flask
from multiline.MultiLineNPExtractor import CCraft
from flask import request
from flask.json import jsonify
import numpy as np

craft = CCraft()

app = Flask(__name__)


@app.route('/version', methods=['GET'])
def version():
    return "v2.0"


@app.route('/multi_to_one_line_converter', methods=['POST'])
def index():
    data = request.get_json()

    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})

    img = np.array(data['img'], dtype="uint8")
    region_name = data.get('region_name', 'default')
    region_name = region_name.replace('-', '_')
    img = craft.make_one_line_from_many(img, region_name, False)
    return {"img": img.tolist()}


if __name__ == '__main__':
    app.run(debug=False, port=os.environ.get("PORT", 5010))
