import os
import json
from shutil import copyfile, rmtree
from typing import List, Tuple, Dict
from collections import Counter


class CustomOptionsMaker:
    """
    TODO: describe class
    """
    def __init__(self,
                 dirpath: str,
                 dirpath_custom: str,

                 dataset_region_classes: List = None,
                 custom_region_classes: List = None,

                 dataset_count_line_classes: List = None,
                 custom_count_line_classes: List = None,

                 custom_options_dirs: List = None,
                 custom_options_sub_dirs: List = None) -> None:
        """
        TODO: describe __init__
        """
        if custom_options_dirs is None:
            custom_options_dirs = ['train', 'val', 'test']
        if custom_options_sub_dirs is None:
            custom_options_sub_dirs = ['ann', 'img']

        self.region_converter_all_idx = None
        self.region_converter_custom = None

        self.count_line_converter_all_idx = None
        self.count_line_converter_custom = None

        self.custom_options_sub_dirs = custom_options_sub_dirs
        self.custom_options_dirs = custom_options_dirs
        self.dirpath_custom = dirpath_custom
        self.dirpath = dirpath

        if dataset_count_line_classes is not None:
            _, self.count_line_converter_all_idx = self.make_convertor(dataset_count_line_classes)
        if custom_count_line_classes is not None:
            self.count_line_converter_custom, _ = self.make_convertor(custom_count_line_classes)

        if dataset_region_classes is not None:
            _, self.region_converter_all_idx = self.make_convertor(dataset_region_classes)
        if custom_region_classes is not None:
            self.region_converter_custom, _ = self.make_convertor(custom_region_classes)

    @staticmethod
    def make_convertor(class_region: List) -> Tuple:
        convertor = {}
        convertor_idx = {}
        for i, class_name in enumerate(class_region):  # OptionsDetector.CLASS_REGION_ALL
            if type(class_name) == str:
                convertor[class_name] = int(i)
                convertor_idx[int(i)] = class_name
            elif type(class_name) == list:
                convertor_idx[int(i)] = class_name
                for name in class_name:
                    convertor[name] = int(i)
        return convertor, convertor_idx

    def prepare_custom_dir(self, dirpath_custom: str) -> None:
        if os.path.exists(dirpath_custom):
            rmtree(dirpath_custom)
        print('Creating path "{}" for custom options'.format(dirpath_custom))
        os.mkdir(dirpath_custom)
        for option_dir in self.custom_options_dirs:
            option_dir_full = os.path.join(dirpath_custom, option_dir)
            if not os.path.exists(option_dir_full):
                os.mkdir(option_dir_full)
            for option_sub_dir in self.custom_options_sub_dirs:
                option_sub_dir_full = os.path.join(option_dir_full, option_sub_dir)
                if not os.path.exists(option_sub_dir_full):
                    os.mkdir(option_sub_dir_full)

    @staticmethod
    def calc_labels_stat(option_dir, labels=None):
        options_stat = Counter()
        if labels is None:
            labels = ["count_lines", "region_id"]
        for (dirpath, dirnames, filenames) in os.walk(option_dir):
            for file in filenames:
                file = os.path.join(dirpath, file)
                with open(file) as json_file:
                    json_data = json.load(json_file)
                    key = tuple([json_data[label] for label in labels])
                    options_stat[key] += 1
        print("Labels stat for", option_dir)
        print(options_stat.most_common(1000))
        return options_stat

    def make(self, verbose: bool = False) -> None:
        self.prepare_custom_dir(self.dirpath_custom)
        for option_dir in self.custom_options_dirs:
            self.filter_custom_dataset(option_dir, verbose)

    def filter_custom_dataset(self, option_dir: str, verbose: bool = False) -> None:
        print(f'dir: {self.dirpath} '
              f'option_dir: {option_dir} '
              f'custom_options_sub_dirs[0]: {self.custom_options_sub_dirs[0]}')
        ann_dir = os.path.join(self.dirpath, option_dir, self.custom_options_sub_dirs[0])
        _ = self.calc_labels_stat(ann_dir)
        cnt = 0
        for dirName, subdirList, fileList in os.walk(ann_dir):
            for fname in fileList:
                fname = os.path.join(ann_dir, fname)
                with open(fname) as jsonF:
                    json_data = json.load(jsonF)
                cnt += self.make_custom_record(option_dir, json_data, verbose)
        print("In {} prepared {} records".format(option_dir, cnt))

    def make_custom_record(self, option_dir: str, json_data: Dict, verbose: bool = False) -> int:
        ann_file_to = os.path.join(self.dirpath_custom,
                                   option_dir,
                                   self.custom_options_sub_dirs[0],
                                   "{}.json".format(json_data['name']))

        img_file_from = os.path.join(self.dirpath,
                                     option_dir,
                                     self.custom_options_sub_dirs[1],
                                     "{}.png".format(json_data['name']))
        img_file_to = os.path.join(self.dirpath_custom,
                                   option_dir,
                                   self.custom_options_sub_dirs[1],
                                   "{}.png".format(json_data['name']))
        if self.count_line_converter_all_idx is not None and self.count_line_converter_custom is not None:
            custom_count_line_name = self.count_line_converter_all_idx[int(json_data['count_lines'])]
            if custom_count_line_name not in self.count_line_converter_custom:
                return 0
            all_count_line_id = json_data['count_lines']
            custom_count_line_id = self.count_line_converter_custom[custom_count_line_name]
            json_data['count_lines'] = custom_count_line_id
            if verbose:
                print('{} {} -> custom_count_line_id {} -> {}'.format(
                    json_data['name'],
                    custom_count_line_name,
                    all_count_line_id, custom_count_line_id))
        if self.region_converter_all_idx is not None and self.region_converter_custom is not None:
            custom_region_name = self.region_converter_all_idx[int(json_data['region_id'])]
            if custom_region_name not in self.region_converter_custom:
                return 0
            all_region_id = json_data['region_id']
            custom_region_id = self.region_converter_custom[custom_region_name]
            json_data['region_id'] = custom_region_id
            if verbose:
                print('{} {} -> custom_region_id {} -> {}'.format(
                    json_data['name'],
                    custom_region_name,
                    all_region_id, custom_region_id))
        with open(ann_file_to, "w", encoding='utf8') as jsonWF:
            json.dump(json_data, jsonWF, ensure_ascii=False)
        copyfile(img_file_from, img_file_to)
        return 1
