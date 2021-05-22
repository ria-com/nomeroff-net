import os
import json
from shutil import copyfile
custom_options_dirs = ['train', 'val', 'test']
custom_options_sub_dirs = ['ann', 'img']

def make_convertor(class_region_all):
    convertor = {}
    convertor_idx = {}
    for i, className in enumerate(class_region_all): #OptionsDetector.CLASS_REGION_ALL
        convertor[className] = int(i)
        convertor_idx[int(i)] = className
    return convertor, convertor_idx

def prepare_custom_dir(dirpath_custom):
    if not os.path.exists(dirpath_custom):
        print('Creating path "{}" for custom options'.format(dirpath_custom))
        os.mkdir(dirpath_custom)
        for option_dir in custom_options_dirs:
            option_dir_full = os.path.join(dirpath_custom, option_dir)
            if not os.path.exists(option_dir_full):
                os.mkdir(option_dir_full)
            for option_sub_dir in custom_options_sub_dirs:
                option_sub_dir_full = os.path.join(option_dir_full, option_sub_dir)
                if not os.path.exists(option_sub_dir_full):
                    os.mkdir(option_sub_dir_full)


class CustomOptionsMaker:
    """
    TODO: describe class
    """
    def __init__(self, dirpath, dirpath_custom, class_region_all, class_region) -> None:
        """
        TODO: describe __init__
        """
        # input
        self.converter_all, self.converter_all_idx = make_convertor(class_region_all)
        self.converter_custom, self.converter_custom_idx = make_convertor(class_region)
        self.dirpath = dirpath
        self.dirpath_custom = dirpath_custom

    def make(self):
        prepare_custom_dir(self.dirpath_custom)
        for option_dir in custom_options_dirs:
            self.filter_custom_dataset(option_dir)

    def filter_custom_dataset(self, option_dir):
        print(f'dir: {self.dirpath} option_dir {option_dir} custom_options_sub_dirs[0] {custom_options_sub_dirs[0]}')
        ann_dir = os.path.join(self.dirpath, option_dir, custom_options_sub_dirs[0])
        cnt = 0
        for dirName, subdirList, fileList in os.walk(ann_dir):
            for fname in fileList:
                fname = os.path.join(ann_dir, fname)
                with open(fname) as jsonF:
                    jsonData = json.load(jsonF)
                if self.converter_all_idx[int(jsonData['region_id'])] in self.converter_custom:
                    self.make_custom_record(option_dir, jsonData)
                    cnt = cnt + 1
        print("In {} prepared {} records".format(option_dir, cnt))

    def make_custom_record(self, option_dir, jsonData):
        ann_file_to = os.path.join(self.dirpath_custom, option_dir, custom_options_sub_dirs[0], "{}.json".format(jsonData['name']))

        img_file_from = os.path.join(self.dirpath, option_dir, custom_options_sub_dirs[1], "{}.png".format(jsonData['name']))
        img_file_to = os.path.join(self.dirpath_custom, option_dir, custom_options_sub_dirs[1], "{}.png".format(jsonData['name']))

        custom_region_name = self.converter_all_idx[int(jsonData['region_id'])]
        all_region_id = jsonData['region_id']
        custom_region_id = self.converter_custom[custom_region_name]
        jsonData['region_id'] = custom_region_id
        print('{} {} -> custom_region_id {} -> {}'.format(jsonData['name'], custom_region_name, all_region_id, custom_region_id))

        with open(ann_file_to, "w", encoding='utf8') as jsonWF:
            json.dump(jsonData, jsonWF, ensure_ascii=False)
        copyfile(img_file_from, img_file_to)
