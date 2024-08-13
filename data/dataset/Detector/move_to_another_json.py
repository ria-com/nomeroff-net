"""
RUN EXAMPLE
```
python3.9 move_to_another_json.py \
    --input_json=/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/val/via_region_data.json \
    --output_json=/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/via_region_data.json
```

```
python3.9 move_to_another_json.py \
    --input_json=/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06/val/via_region_data.json \
    --output_json=/var/www/projects_computer_vision/nomeroff-net/data/dataset/Detector/autoriaNumberplateDataset-2023-03-06-checked/val_wrong_rotation/via_region_data.json
```



"""
import shutil
import glob
import os
import json
import argparse

BLURRED_MOVE_KEYS = ["1530962", "25069188-292385382", "250028445orig", "25069188-292385382", "250029304orig", "379960333",
             "379968445", "50ee32c98d36", "300px-71-50_loh", "1492197", "1528168", "1536171", "08700034-43757278",
             "08700034-43757278", "25001232-291009984", "25121257-293405562", "25132185-293613279",
             "25137534-293729622", "184718700", "184997124", "194734656orig", "233538103orig", "239940611orig",
             "242035179orig", "244680087orig", "250018061orig", "250021449orig", "250031957orig", "250038196orig",
             "250038905orig", "299238151", "322421120", "322421120", "322421120", "322421120", "325326974",
             "326650147", "327153994", "327320850", "327516293", "369343061-28728766", "369344922-26890009",
             "369347543-28729004", "369348563-28729050", "369348563-28729050", "369351602-28729184",
             "369351870-28729201", "369353247-28729269", "369355511-28729390", "369356495-28729453",
             "369356495-28729453", "369356665-28729424", "369357988-28710137", "369361292-28729663",
             "369366990-28729954", "369366990-28729954", "369366990-28729954", "369369609-27364254",
             "369369895-28730069", "369370072-28730093", "369382911-28730819", "369385212-28730865",
             "379754279", "379754279", "379754279", "380047071", "380355251", "380688804", "380895432",
             "380987098", "381138751", "381329940", "455280390", "455561189", "455636274", "480414631",
             "481893283", "481893283", "481947542", "481957308", "482409938", "482410128", "482410128",
             "482410128", "482410128", "482413584", "482413584", "482414847", "482422700", "482422700", "482422700",
             "482425725", "FinnJAE-Nissan-Datsun-Cedric-230-02"]

WRONG_ROTATION = [
    "4a3b7ca1bf3ec881a1d74afb5870a18e", "1539987", "177171380", "250027624orig", "250028623orig", "250031896orig",
    "329169744", "369344332-28728841", "369344445-1", "369351753-28729184", "369352948-1", "482410003", "482413584",
    "482414902", "d7e4ffec33f43331bb9d40cc3bcb0bf4", "m73w3f"
]


def move_to_another_json(input_json, output_json):
    move_keys = list(set(WRONG_ROTATION))
    res_data = {
        "_via_settings": {},
        "_via_img_metadata": {},
        "_via_attributes": {}
    }
    input_dir = os.path.dirname(input_json)
    output_dir = os.path.dirname(output_json)

    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not res_data["_via_settings"] and "_via_settings" in data:
        res_data["_via_settings"] = data["_via_settings"]
    if not res_data["_via_attributes"] and "_via_attributes" in data:
        res_data["_via_attributes"] = data["_via_attributes"]

    for key in move_keys:
        img_path = glob.glob(os.path.join(input_dir, f"{key}*"))[0]
        basename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(output_dir, basename))

        res_data["_via_img_metadata"][basename] = data["_via_img_metadata"][basename]

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(res_data, file, indent=4, ensure_ascii=False)
    print(f"Merged json contains {len(res_data['_via_img_metadata'])} keys. Saved to file: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='This script move by keys images and JSON metadata ')
    parser.add_argument("--input_json",
                        default="/var/www/projects_computer_vision/nomeroff-net/data/dataset/"
                                "Detector/autoriaNumberplateDataset-2023-03-06/val/via_region_data.json",
                        help="input json")
    parser.add_argument("--output_json",
                        default="/var/www/projects_computer_vision/nomeroff-net/data/dataset/"
                                "Detector/autoriaNumberplateDataset-2023-03-06-checked/val_blurred/via_region_data.json",
                        help='output json')

    args = parser.parse_args()

    move_to_another_json(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
