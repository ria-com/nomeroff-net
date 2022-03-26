import os
import json
import cv2
from shutil import copyfile, rmtree, move
from typing import List, Tuple, Dict
from collections import Counter
import random
import copy

import numpy as np

from nomeroff_net.tools.augmentations import aug


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

                 state_ids_all_labels: List = None,
                 state_ids_only_labels: List = None,

                 custom_options_dirs: List = None,
                 custom_options_sub_dirs: List = None,

                 items_per_class: int = 2000,
                 rebalance_suffix: str = 'rebalance',

                 verbose: bool = False
                 ) -> None:
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

        self.verbose = verbose

        self.items_per_class = items_per_class
        self.dirpath_custom_rebalance = '{}-{}'.format(self.dirpath_custom, rebalance_suffix)

        if dataset_count_line_classes is not None:
            _, self.count_line_converter_all_idx = self.make_convertor(dataset_count_line_classes)
        if custom_count_line_classes is not None:
            self.count_line_converter_custom, _ = self.make_convertor(custom_count_line_classes)

        if dataset_region_classes is not None:
            _, self.region_converter_all_idx = self.make_convertor(dataset_region_classes)
        if custom_region_classes is not None:
            self.region_converter_custom, _ = self.make_convertor(custom_region_classes)

        self.state_ids = []
        if state_ids_only_labels is not None:
            for state_label in state_ids_only_labels:
                state_id = state_ids_all_labels.index(state_label)
                self.state_ids.append(state_id)

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
        for (dirpath, dir_names, file_names) in os.walk(option_dir):
            for file in file_names:
                file = os.path.join(dirpath, file)
                with open(file) as json_file:
                    print(file)
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
        filtered_cnt = 0
        for dir_name, subdir_list, file_list in os.walk(ann_dir):
            for fname in file_list:
                fname = os.path.join(ann_dir, fname)
                with open(fname) as jsonF:
                    json_data = json.load(jsonF)
                if self.filter_by_state_id(json_data):
                    cnt += self.make_custom_record(option_dir, copy.deepcopy(json_data), verbose)
                else:
                    filtered_cnt += 1
        print("In {} prepared {} records".format(option_dir, cnt))
        print("In {} filtered by state_id {} records".format(option_dir, filtered_cnt))

    def filter_by_state_id(self, json_data):
        if len(self.state_ids):
            return "state_id" in json_data and int(json_data["state_id"]) in self.state_ids
        else:
            return True

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

    def get_regions_stats(self, custom_options_dir):
        ann_dir = os.path.join(self.dirpath_custom, custom_options_dir, self.custom_options_sub_dirs[0])
        options_stats = {}
        for dir_name, subdir_list, file_list in os.walk(ann_dir):
            for fname in file_list:
                fname_full = os.path.join(ann_dir, fname)
                with open(fname_full) as jsonF:
                    json_data = json.load(jsonF)
                if not (json_data["region_id"] in options_stats):
                    options_stats[json_data["region_id"]] = {}
                options_stats[json_data["region_id"]][fname] = json_data
        return options_stats

    def get_count_lines_stats(self, custom_options_dir):
        ann_dir = os.path.join(self.dirpath_custom, custom_options_dir, self.custom_options_sub_dirs[0])
        options_stats = {}
        for dir_name, subdir_list, file_list in os.walk(ann_dir):
            for fname in file_list:
                fname_full = os.path.join(ann_dir, fname)
                with open(fname_full) as jsonF:
                    json_data = json.load(jsonF)
                if not (str(json_data["count_lines"]) in options_stats):
                    options_stats[str(json_data["count_lines"])] = {}
                options_stats[str(json_data["count_lines"])][fname] = json_data
        return options_stats

    def duplicate_class_items(self,
                              options_stat: List or np.ndarray,
                              idx: int,
                              custom_options_dir: str,
                              with_aug: bool = False):
        dirpath_custom_options_dir = os.path.join(self.dirpath_custom, custom_options_dir)
        dirpath_custom_options_dir_ann = os.path.join(dirpath_custom_options_dir, self.custom_options_sub_dirs[0])
        dirpath_custom_options_dir_img = os.path.join(dirpath_custom_options_dir, self.custom_options_sub_dirs[1])
        for _dir in [dirpath_custom_options_dir_ann, dirpath_custom_options_dir_img]:
            os.makedirs(_dir, exist_ok=True)

        if self.verbose:
            print('Try make duplicate/augmentation for {} items'.format(len(options_stat.keys())))

        for fname in options_stat.keys():
            fname_copy = 'aug_{}_{}'.format(idx, fname)
            fname_full = os.path.join(dirpath_custom_options_dir_ann, fname_copy)
            json_data = copy.deepcopy(options_stat[fname])
            json_data["name"] = 'aug_{}_{}'.format(idx, json_data["name"])
            with open(fname_full, "w", encoding='utf8') as jsonWF:
                json.dump(json_data, jsonWF, ensure_ascii=False)
            fname_img_to = json_data["name"] + '.png'
            fname_img_from = options_stat[fname]["name"] + '.png'
            img_file_from = os.path.join(dirpath_custom_options_dir_img, fname_img_from)
            img_file_to = os.path.join(dirpath_custom_options_dir_img, fname_img_to)
            if with_aug:
                if self.verbose:
                    print('Make augmented file: "{}"'.format(img_file_to))
                img = cv2.imread(img_file_from)
                imgs = aug([img])
                img = imgs[0]
                cv2.imwrite(img_file_to, img)
            else:
                if self.verbose:
                    print('Copy file "{}" -> "{}"'.format(img_file_from, img_file_to))
                copyfile(img_file_from, img_file_to)
        return 1

    def move_unused_items_to_rebalance_dir(self,
                                           rebalance_options_stats: Dict,
                                           region_id: int,
                                           selected_items: List,
                                           custom_options_dir: str):
        options_stat = rebalance_options_stats[region_id]
        dirpath_custom_options_dir = os.path.join(self.dirpath_custom, custom_options_dir)
        dirpath_custom_options_dir_ann = os.path.join(dirpath_custom_options_dir,
                                                      self.custom_options_sub_dirs[0])
        dirpath_custom_options_dir_img = os.path.join(dirpath_custom_options_dir,
                                                      self.custom_options_sub_dirs[1])
        dirpath_custom_rebalance_options_dir = os.path.join(self.dirpath_custom_rebalance, custom_options_dir)
        dirpath_custom_rebalance_options_ann = os.path.join(dirpath_custom_rebalance_options_dir,
                                                            self.custom_options_sub_dirs[0])
        dirpath_custom_rebalance_options_img = os.path.join(dirpath_custom_rebalance_options_dir,
                                                            self.custom_options_sub_dirs[1])

        selected_items_set = set(selected_items)
        all_items_set = set(options_stat.keys())
        diff_items = all_items_set - selected_items_set
        for item in diff_items:
            fname_full_from = os.path.join(dirpath_custom_options_dir_ann, item)
            fname_full_to = os.path.join(dirpath_custom_rebalance_options_ann, item)
            move(fname_full_from, fname_full_to)

            img_name = options_stat[item]["name"] + '.png'
            img_file_from = os.path.join(dirpath_custom_options_dir_img, img_name)
            img_file_to = os.path.join(dirpath_custom_rebalance_options_img, img_name)

            move(img_file_from, img_file_to)
        return 1

    def add_class_entries(self, options_stat: Dict, custom_options_dir: str, with_aug: bool = False):
        items_per_class = self.items_per_class
        region_cnt = len(options_stat.keys())
        copies_cnt = items_per_class // region_cnt
        modulo = items_per_class % region_cnt
        if self.verbose:
            print('Multiply {} class data in {} times and add random {} items of class'.format(region_cnt,
                                                                                               copies_cnt,
                                                                                               modulo))
        appendix_items = random.sample(tuple(options_stat.keys()), modulo)
        for i in range(1, copies_cnt):
            if self.verbose:
                print('Make full copy for index {}'.format(i))
            self.duplicate_class_items(options_stat, i, custom_options_dir, with_aug)
        options_stat_appendix = {}
        for item in appendix_items:
            options_stat_appendix[item] = options_stat[item]
        if self.verbose:
            print('Add appendix ({} items) for index {}'.format(len(options_stat_appendix.keys()), copies_cnt))
        self.duplicate_class_items(options_stat_appendix, copies_cnt, custom_options_dir, with_aug)
        return 1

    def rebalance_region(self,
                         rebalance_options_stats: Dict,
                         region_id: int,
                         custom_options_dir: str,
                         with_aug: bool = False):
        options_stat = rebalance_options_stats[region_id]
        region_cnt = len(options_stat.keys())
        items_per_class = self.items_per_class
        if items_per_class < region_cnt:
            balanced_region_items = random.sample(list(options_stat.keys()), items_per_class)
            if self.verbose:
                print('Crop class region_id {} to {}'.format(region_id, len(balanced_region_items)))
            self.move_unused_items_to_rebalance_dir(rebalance_options_stats,
                                                    region_id,
                                                    balanced_region_items,
                                                    custom_options_dir)
        else:
            if self.verbose:
                print('Increase class region_id {} from {} to {}'.format(region_id,
                                                                         len(options_stat.keys()),
                                                                         items_per_class))
            self.add_class_entries(options_stat, custom_options_dir, with_aug)

    def rebalance_count_lines_item(self,
                                   rebalance_options_stats: Dict,
                                   count_lines: int or str,
                                   custom_options_dir: str,
                                   with_aug: bool = False):
        options_stat = rebalance_options_stats[count_lines]
        count_lines_cnt = len(options_stat.keys())
        items_per_class = self.items_per_class
        if items_per_class < count_lines_cnt:
            balanced_region_items = random.sample(list(options_stat.keys()), items_per_class)
            if self.verbose:
                print('Crop class count_lines {} to {}'.format(count_lines, len(balanced_region_items)))
            self.move_unused_items_to_rebalance_dir(rebalance_options_stats,
                                                    count_lines,
                                                    balanced_region_items,
                                                    custom_options_dir)
        else:
            if self.verbose:
                print('Increase class region_id {} from {} to {}'.format(count_lines,
                                                                         len(options_stat.keys()),
                                                                         items_per_class))
            self.add_class_entries(options_stat, custom_options_dir, with_aug)

    def rebalance_regions(self,
                          custom_options_dir: str = 'train',
                          with_aug: bool = False,
                          verbose: bool = False) -> int:
        self.verbose = verbose
        if os.path.exists(self.dirpath_custom):
            dirpath_custom_rebalance_options_dir = os.path.join(self.dirpath_custom_rebalance, custom_options_dir)
            if not os.path.exists(dirpath_custom_rebalance_options_dir):
                for sub_dir in self.custom_options_sub_dirs:
                    os.makedirs(os.path.join(dirpath_custom_rebalance_options_dir, sub_dir), exist_ok=True)
                rebalance_options_stats = self.get_regions_stats(custom_options_dir)
                for region_id in rebalance_options_stats:
                    if self.verbose:
                        print('Prepare data for region_id: {}'.format(region_id))
                    self.rebalance_region(rebalance_options_stats, region_id, custom_options_dir, with_aug)
            else:
                print('Rebalancing is possible only once!')
        else:
            print('Rebalancing is possible only after calling the make method for custom dir: "{}"!'
                  .format(self.dirpath_custom))
        return 1

    def rebalance_count_lines(self,
                              custom_options_dir: str = 'train',
                              with_aug: bool = False,
                              verbose: bool = False) -> int:
        self.verbose = verbose
        if os.path.exists(self.dirpath_custom):
            dirpath_custom_rebalance_options_dir = os.path.join(self.dirpath_custom_rebalance, custom_options_dir)
            if not os.path.exists(dirpath_custom_rebalance_options_dir):
                for sub_dir in self.custom_options_sub_dirs:
                    os.makedirs(os.path.join(dirpath_custom_rebalance_options_dir, sub_dir), exist_ok=True)
                rebalance_options_stats = self.get_count_lines_stats(custom_options_dir)
                if verbose:
                    print("Rebalance options stats", rebalance_options_stats)
                for count_lines in rebalance_options_stats:
                    if self.verbose:
                        print('Prepare data for count_lines: {}'.format(count_lines))
                    self.rebalance_count_lines_item(rebalance_options_stats, count_lines, custom_options_dir, with_aug)
            else:
                print('Rebalancing is possible only once!')
        else:
            print('Rebalancing is possible only after calling the make method for custom dir: "{}"!'
                  .format(self.dirpath_custom))
        return 1
