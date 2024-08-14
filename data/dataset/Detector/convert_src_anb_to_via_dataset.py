"""
python3.9 ./convert_src_anb_to_via_dataset.py
"""
import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path


def convert_dataset(src_dir, anb_dir, dst_dir):
    # Створюємо вихідну директорію, якщо вона не існує
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    via_data = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all"
                },
                "image": {
                    "region_label": "",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION"
                }
            },
            "core": {
                "buffer_size": "18",
                "filepath": {
                    "/mrcnn4/": 3
                },
                "default_filepath": "./data"
            },
            "project": {
                "name": "via_data_end"
            }
        },
        "_via_img_metadata": {},
        "_via_attributes": {
            "region": {
                "class": {
                    "type": "text"
                }
            },
            "file": {}
        }
    }

    # Проходимо по всім файлам в директорії anb
    for anb_file in tqdm(list(os.listdir(anb_dir))):
        if anb_file.endswith('.json'):
            with open(os.path.join(anb_dir, anb_file), 'r') as f:
                anb_data = json.load(f)

            img_filename = anb_data['src']

            # Копіюємо зображення
            shutil.copy(os.path.join(src_dir, img_filename), os.path.join(dst_dir, img_filename))

            regions = []
            for region in anb_data['regions'].values():
                keypoints = region['keypoints']
                regions.append({
                    "region_attributes": {
                        "label": "numberplate"
                    },
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [int(float(point[0])) for point in keypoints],
                        "all_points_y": [int(float(point[1])) for point in keypoints]
                    }
                })

            via_data['_via_img_metadata'][img_filename] = {
                "file_attributes": {},
                "filename": img_filename,
                "regions": regions,
                "size": 0
            }

    # Зберігаємо via_region_data.json
    with open(os.path.join(dst_dir, 'via_region_data.json'), 'w') as f:
        json.dump(via_data, f, indent=2)


if __name__ == "__main__":
    src_dir = '/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/add.all/src'
    anb_dir = '/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/add.all/anb'
    dst_dir = '/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/added'

    convert_dataset(src_dir, anb_dir, dst_dir)
