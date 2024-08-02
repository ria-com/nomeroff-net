import os
import json
from pathlib import Path

# itemTemplate = {
#       "filename": "item_template.jpg",
#       "size": 99999,
#       "regions": [
#         {
#           "shape_attributes": {
#             "name": "rect",
#             "x": 20,
#             "y": 10,
#             "width": 100,
#             "height": 50
#           },
#           "region_attributes": {}
#         }
#       ]
# }


def get_key(file):
    return Path(file).stem


def get_via_rect(row):
    #print(row)
    x = round(row['xmin'])
    y = round(row['ymin'])
    width = round(row['xmax']-row['xmin'])
    height = round(row['ymax']-row['ymin'])
    return {
        "name": "rect",
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }

def get_label(row, label_type):
    if label_type == "radio":
        return {
            "label": str(row['class']),
            "confidence": str(row['confidence'])
        }
    if label_type == "text":
        return {
            "label": str(row['name']),
            "confidence": str(row['confidence'])
        }

def make_options_hash(labels):
    obj = {}
    for idx, label in enumerate(labels):
        obj[idx] = label
    return obj


class VIADataset:
    """
        VIA Dataset adapter
    """
    def __init__(self,
                 label_type="radio",
                 verbose=False):
        self.data = None
        self.labels = []
        self.idx = {}
        self.label_type = label_type
        self.verbose = verbose

        absolute_path = os.path.dirname(__file__)
        relative_path = "../../moderation/templates/via/via_region_data_template.json"
        self.template_path = os.path.join(absolute_path, relative_path)

    def load_from_template(self, template_path=None, labels=[], label_type="radio"):
        if template_path is None:
            template_path = self.templates_path
        with open(template_path, 'r') as f:
            self.data = json.load(f)
            # self.item_template = self.data['_via_img_metadata']['item_template']
            del self.data['_via_img_metadata']['item_template']
            if len(labels):
                self.labels = labels
                if label_type == "radio":
                    self.label_type = "radio"
                    self.data['_via_attributes']['region'] = {
                            "label": {
                                "type": "radio",
                                "description": "Logo label",
                                "options": make_options_hash(labels),
                                "default_options": {}
                            }
                        }
                if label_type == "text":
                    self.label_type = "text"
                    self.data['_via_attributes']['region'] = {
                        "label": {
                            "type": "text",
                            "description": "Logo label",
                            "default_value": ""
                        }
                    }


    def load_from_via_file(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            if '_via_settings' in data:
                self.data = data
            else:
                self.load_from_template()
                self.data['_via_img_metadata'] = data['_via_img_metadata']

    def add_items_from_via(self, src_via_file):
        for item in src_via_file.data['_via_img_metadata']:
            self.data['_via_img_metadata'][item] = src_via_file.data['_via_img_metadata'][item]

    def load_metadata_from_file(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.data['_via_img_metadata'] = data['_via_img_metadata']

    def update_by_file(self, file, item):
        key = get_key(file)
        self.data['_via_img_metadata'][key] = item

    def update_by_file_from_yolo_detector(self, file, size, detection_result):
        key = get_key(file)
        regions = []
        for indices, row in detection_result.iterrows():
            region_attributes = get_label(row, self.label_type)
            if region_attributes is None:
                region_attributes = {}
            regions.append({
                "shape_attributes": get_via_rect(row),
                "region_attributes": region_attributes
            })
        item_data = {
            "filename": file,
            "size": size,
            "regions": regions,
            "file_attributes": {}
        }
        if key in self.data['_via_img_metadata']:
            self.data['_via_img_metadata'][key].update(item_data)
        else:
            self.data['_via_img_metadata'][key] = item_data

    def del_by_file(self, file):
        key = get_key(file)
        del self.data['_via_img_metadata'][key]

    def get_via(self):
        return self.data

    def write_via(self, result_path):
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_via(), f, ensure_ascii=False, indent=4)

    def is_empty(self):
        return len(self.data['_via_img_metadata']) == 0

    def index_via_files(self):
        self.idx = {}
        for key in self.data['_via_img_metadata']:
            item = self.data['_via_img_metadata'][key]
            #print(f"key: {key}, filename: {item['filename']}")
            self.idx[item["filename"]] = key
        return self.idx

    def reindex_via_files(self):
        idx = {}
        for key in self.idx:
            idx[key] = self.data['_via_img_metadata'][self.idx[key]]
        self.data['_via_img_metadata'] = idx

    def add_data_by_keys(self, via_source):
        updated_keys = []
        for filename in via_source.idx:
            key_local = self.idx[filename]
            key_source = via_source.idx[filename]

            print("key_local")
            print(key_local)
            print("key_source")
            print(key_source)

            print("self.data['_via_img_metadata'][key_local]['regions']")
            print(self.data['_via_img_metadata'][key_local]['regions'])
            print(self.data['_via_img_metadata'][key_local])


            print("via_source.data['_via_img_metadata'][key_source]['regions']")
            print(via_source.data['_via_img_metadata'][key_source]['regions'])

            for region in via_source.data['_via_img_metadata'][key_source]['regions']:
                self.data['_via_img_metadata'][key_local]['regions'].append(region)

            updated_keys.append(key_local)
        return updated_keys

