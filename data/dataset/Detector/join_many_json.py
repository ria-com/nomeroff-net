"""
## Description:
This script merges JSON files of VIA markup using a given glob pattern. It is designed to work with train and val datasets. The script merges all files that match the pattern, keeping "_via_settings" and "_via_attributes" from the first file and merging "_via_img_metadata" from all files.
Command line arguments:

--mode: Required argument. Takes 'train' or 'val' values to select the mode of operation.
--input: Optional argument. Path to the directory with JSON files. Defaults to "./Via_datasets/".
--output: Optional argument. Output file name. Defaults to "merged_[mode]_data.json".

## Usage examples:

To merge training data:
```
python join_many_json.py --mode train
```

To merge training data:
```
python join_many_json.py --mode train
```

To merge validation data:
```
python join_many_json.py --mode val
```

Specifying a custom input directory and output file name:
```
python join_many_json.py --mode train --input /path/to/json/files/ --output custom_merged_train.json
```

Merging validation data with a custom output file:
```
python join_many_json.py --mode val --output merged_validation_data.json
```

"""
import glob
import json
import argparse

EXCLUDE_KEYS = {
    "train": ["455803323", "455469125", "446868931", "369353452-28729267", "250021935orig", "273013077", "455633769",
              "25124468-293465329", "381220577", "455682372", "455337891", "25134999-293678154", "25128040-293542668",
              "369353571-28729341", "325685138", "326664643", "455745278", "446258384", "369378285-28730491",
              "369351178-28729173", "281667793", "250019186orig", "369355224-28729368", "198337952orig",
              "comment_SA14b8PGDt7cFTbdFqok9aUMrIk4TdFC", "250021573orig", "326529514", "369354056-28729324",
              "369381718-28730690", "369348818-1", "239094697orig", "25134999-293678074", "262284520", "279156726",
              "1536315", "326729287", "455401331", "381313626", "443995520", "369378801-28730498", "369349548-28729084",
              "25132610-293628070", "369356023-28729411", "455696668", "455402579", "369345198-28728900", "446555086",
              "445731966", "243207701orig", "455540155", "455769323", "236176585orig", "369346479-28729022", "971709",
              "447849961", "25137800-293735775", "369344415-28728847", "455564171", "25134999-293678072",
              "369348555-28729050", "369388523-28730980", "1422484", "1355244", "25125336-293482639",
              "369347660-28697325", "455422448", "369373077-27894217", "380894532", "369349547-28729084", "455381910",
              "25123721-293450746", "455352225", "369368049-28729974", "436339344", "248160421orig",
              "369354523-28729328", "369367733-28729987", "1057778", "424045494", "455662794", "25132328-293626071",
              "369375611-28730342", "369352890-28729291", "25137800-293735763", "380308060", "369361289-28729659",
              "455784207", "325618938", "448667970", "446505321", "455574913", "369348290-28729054", "482410180",
              "463966042", "481964711", "481890411", "481980829", "481970760", "481965637", "481964377", "481928914",
              "481931033", "481890446", "481975056", "481931056", "481964359", "481888289", "481968941", "481965032",
              "481931035", "481955348", "481935612", "481957450", "481917535", "480412922", "455698511",
              "369347313-28728995", "449012754", "455507489", "369346217-28728948", "25128800-293557196",
              "447272736", "455339059", "455513586", "455476740", "25129936-293577204", "25123404-293445385",
              "25121257-293405563", "244707490orig", "446722249", "455781853", "369347032-28727249",
              "369345197-28728900", "25129936-293577240", "447836031", "25129936-293577264", "249168973orig",
              "327420454", "380618593", "381571959", "448616810", "250037131orig", "25134999-293678060",
              "369358959-28729591", "369351230-28729205", "369353850-28729341", "446404477", "25125614-293489509",
              "434288173", "369353488-28729341", "180216419", "448019332", "445779180", "445553216", "445779180",
              "446709310", "455415503", "369353625-28729341", "25134999-293678062", "455657339", "213652557",
              "1533067", "453289596", "369354768-28729418", "326831469", "250020144orig", "233538110orig",
              "239940200orig", "455645021", "oldtimer-car-old-car-classic-cars", "369355650-28729371", "1534932",
              "25132732-293651755", "481950558", "481906464", "481975061", "481931851", "484135192", "481957088",
              "481957455", "481933774", "481907868", "481973192", "481971917", "481975066", "481957093", "481935928",
              "481967698", "481965620", "481964727", "481925997"],
    "val": [
        "481876196", "481928943", "481850071", "481876242", "481967702", "481972897", "481965004", "481931859",
        "481928906", "481957089", "481964713", "369349459-1", "369377166-28730473", "446396813", "369356207-28729429",
        "250018433orig", "245510054orig", "369342646-28697223", "447201412", "369356082-28729446", "381022092",
        "25130252-293583940", "25121257-293405549", "455561112", "25119892-293375065", "369343838-28728815",
        "325810474", "315128530", "24776008-286676444", "1526631", "445731544", "447830896", "381469151", "448440836",
        "380664228", "455675503", "243835728orig", "Nomera_5", "1528499", "325783091", "369362365-28316975",
        "455706411",
        "finn_japanese_car_extravaganza2", "369352610-28729233", "250036992orig", "unnamed", "369351701-28729184",
        "369354219-28729341", "455638756", "244526387orig", "115086914orig", "25132732-293651617", "325402797"
    ]
}


def merge_json_files(glob_pattern, output_file="merged_via_data.json", mode="train"):
    exclude_keys = set(EXCLUDE_KEYS[mode])
    merged_data = {
        "_via_settings": {},
        "_via_img_metadata": {},
        "_via_attributes": {}
    }
    for file_path in glob.glob(glob_pattern):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not merged_data["_via_settings"] and "_via_settings" in data:
                merged_data["_via_settings"] = data["_via_settings"]
            if "_via_img_metadata" in data:
                merged_data["_via_img_metadata"].update(data["_via_img_metadata"])
            if not merged_data["_via_attributes"] and "_via_attributes" in data:
                merged_data["_via_attributes"] = data["_via_attributes"]
    for key in exclude_keys:
        if f"{key}.jpeg" in merged_data["_via_img_metadata"]:
            del merged_data["_via_img_metadata"][f"{key}.jpeg"]
        elif f"{key}.png" in merged_data["_via_img_metadata"]:
            del merged_data["_via_img_metadata"][f"{key}.png"]
        elif f"{key}.jpg" in merged_data["_via_img_metadata"]:
            del merged_data["_via_img_metadata"][f"{key}.jpg"]
        else:
            raise Exception(f"Unknown key {key}")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, indent=4, ensure_ascii=False)
    print(f"Merged json contains {len(merged_data['_via_img_metadata'])} keys. Saved to file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='This script merges JSON files of VIA markup using a '
                                                 'given glob pattern. '
                                                 'It is designed to work with train and val datasets. '
                                                 'The script merges all files that match the pattern, '
                                                 'keeping "_via_settings" and "_via_attributes" from the'
                                                 ' first file and merging "_via_img_metadata" from all files.')
    parser.add_argument("--mode", choices=['train', 'val'], required=True,
                        help="Required argument. Takes 'train' or 'val' values to select the mode of operation.")
    parser.add_argument("--input", default="./Via_datasets/",
                        help="Optional argument. Path to the directory with JSON files. Defaults to './Via_datasets/'.")
    parser.add_argument("--output", help='Optional argument. Output file name. '
                                         'Defaults to "merged_[mode]_data.json".')

    args = parser.parse_args()

    glob_pattern = f"{args.input}*{args.mode}*"
    output_file = args.output or f"merged_{args.mode}_data.json"

    merge_json_files(glob_pattern, output_file, args.mode)


if __name__ == "__main__":
    main()
