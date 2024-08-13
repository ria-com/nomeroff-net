"""
python3 split_dataset.py --src /mnt/datasets/nomeroff-net/mlines-test --dest /mnt/datasets/nomeroff-net/mlines-test-splited --train 0.8 --test 0.1 --val 0.1 --merge 2 --random_state 42 --debug

python3 split_dataset.py --src /mnt/datasets/nomeroff-net/mlines2-selected --dest /mnt/datasets/nomeroff-net/mlines2-selected-splited --train 0.8 --test 0.1 --val 0.1 --merge auto --random_state 42

python3 split_dataset.py --src /mnt/datasets/nomeroff-net/mlines3-selected --dest /mnt/datasets/nomeroff-net/mlines3-selected-splited3 --train 0.8 --test 0.1 --val 0.1 --merge auto --random_state 42
"""
import os
import glob
import json
import shutil
import random
import argparse
import warnings
from tqdm import tqdm


def merge_lines_two(lines, region_id, image_id):
    if region_id in [1, 2]:  # Українські
        if len(lines) == 2 and len(lines[0]) == 4 and len(lines[1]) == 4:
            return lines[0][:2] + lines[1] + lines[0][2:]
        else:
            warnings.warn(f"Bad region_id={region_id} values, {lines}, image_id={image_id}")
    elif region_id == 7:  # Казахські
        if len(lines) == 2:
            return lines[0] + lines[1][2:] + lines[1][:2]
        else:
            warnings.warn(f"Bad region_id={region_id} values {lines}, image_id={image_id}")
    return ''.join(lines)  # Інші


def merge_lines_one(lines, region_id, image_id):
    return lines[0]


def merge_lines_three(lines, region_id, image_id):
    return ''.join(lines)


def split_dataset(src_dir, dest_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, debug=True, merge_function="2", random_state=None):
    random.seed(random_state)
    
    if train_ratio + test_ratio + val_ratio > 1.001:
        raise ValueError("Сума співвідношень має дорівнювати 1")

    for split in ['train', 'test', 'val']:
        for subdir in ['anb', 'ann', 'box', 'img', 'src']:
            os.makedirs(os.path.join(dest_dir, split, subdir), exist_ok=True)

    src_files = [f for f in os.listdir(os.path.join(src_dir, 'src')) if f.endswith('.jpeg') or f.endswith('.jpg')]
    random.shuffle(src_files)

    total_files = len(src_files)
    
    merge_functions = {
        '1': merge_lines_one,
        '2': merge_lines_two,
        '3': merge_lines_three
    }

    # Групування зображень за унікальними текстами
    text_groups = {}
    for src_file in tqdm(src_files, desc="Grouping images"):
        base_name = os.path.splitext(src_file)[0]
        anb_file = f"{base_name}.json"
        
        with open(os.path.join(src_dir, 'anb', anb_file), 'r') as f:
            anb_data = json.load(f)
        
        all_texts = []
        for region_key, region_data in anb_data['regions'].items():
            lines_ann_files = sorted(glob.glob(os.path.join(src_dir, "ann", f"{region_key}*")))
            lines = []
            region_ids = []
            count_lines = []
            for ann_file in lines_ann_files:
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                region_ids.append(ann_data["region_id"])
                count_lines.append(ann_data["count_lines"])
                lines.append(ann_data["description"])
            
            if not lines or len(set(region_ids)) != 1 or len(set(count_lines)) != 1:
                continue
            
            merge_func = merge_function if merge_function != "auto" else str(len(lines))
            merged_text = merge_functions[merge_func](lines, region_ids[0], anb_file)
            all_texts.append(merged_text)
        
        for text in all_texts:
            if text not in text_groups:
                text_groups[text] = []
            text_groups[text].append(src_file)

    # Розподіл груп між наборами
    group_counts = {'train': 0, 'test': 0, 'val': 0}
    file_counts = {'train': 0, 'test': 0, 'val': 0}
    split_assignments = {}

    for text, files in text_groups.items():
        if group_counts['train'] / len(text_groups) < train_ratio:
            split = 'train'
        elif group_counts['test'] / len(text_groups) < test_ratio:
            split = 'test'
        else:
            split = 'val'
        
        group_counts[split] += 1
        file_counts[split] += len(files)
        for file in files:
            split_assignments[file] = split

    # Додатковий розподіл для досягнення бажаних пропорцій
    remaining_files = [f for f in src_files if f not in split_assignments]
    random.shuffle(remaining_files)

    for file in remaining_files:
        if file_counts['train'] / total_files < train_ratio:
            split = 'train'
        elif file_counts['test'] / total_files < test_ratio:
            split = 'test'
        else:
            split = 'val'
        
        split_assignments[file] = split
        file_counts[split] += 1

    # Копіювання файлів відповідно до призначених наборів
    for src_file, split in tqdm(split_assignments.items(), desc="Copying files"):
        copy_files_for_image(src_dir, dest_dir, src_file, split, merge_functions, merge_function)

    print(f"Розподіл завершено. Train: {file_counts['train']}, "
          f"Test: {file_counts['test']}, "
          f"Val: {file_counts['val']}")


def copy_files_for_image(src_dir, dest_dir, src_file, split, merge_functions, merge_function):
    base_name = os.path.splitext(src_file)[0]
    anb_file = f"{base_name}.json"
    
    shutil.copy2(os.path.join(src_dir, 'src', src_file), 
                 os.path.join(dest_dir, split, 'src', src_file))
    shutil.copy2(os.path.join(src_dir, 'anb', anb_file), 
                 os.path.join(dest_dir, split, 'anb', anb_file))

    with open(os.path.join(src_dir, 'anb', anb_file), 'r') as f:
        anb_data = json.load(f)

    for region_key, region_data in anb_data['regions'].items():
        box_file = f"{region_key}.png"
        if os.path.exists(os.path.join(src_dir, 'box', box_file)):
            shutil.copy2(os.path.join(src_dir, 'box', box_file), 
                         os.path.join(dest_dir, split, 'box', box_file))

        lines = region_data.get('lines', {})
        for line_num in sorted(lines.keys()):
            img_file = f"{region_key}-line-{line_num}.png"
            ann_file = f"{region_key}-line-{line_num}.json"

            if os.path.exists(os.path.join(src_dir, 'img', img_file)):
                shutil.copy2(os.path.join(src_dir, 'img', img_file), 
                             os.path.join(dest_dir, split, 'img', img_file))

            if os.path.exists(os.path.join(src_dir, 'ann', ann_file)):
                shutil.copy2(os.path.join(src_dir, 'ann', ann_file), 
                             os.path.join(dest_dir, split, 'ann', ann_file))

        merged_ann_file = f"{region_key}-merged.json"
        if os.path.exists(os.path.join(src_dir, 'ann', merged_ann_file)):
            shutil.copy2(os.path.join(src_dir, 'ann', merged_ann_file), 
                         os.path.join(dest_dir, split, 'ann', merged_ann_file))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split dataset for number plate recognition.")
    
    parser.add_argument("--src", type=str, default="/mnt/datasets/nomeroff-net/mlines-test",
                        help="Source directory containing the dataset")
    
    parser.add_argument("--dest", type=str, default="/mnt/datasets/nomeroff-net/mlines-test-splited",
                        help="Destination directory for the split dataset")
    
    parser.add_argument("--train", type=float, default=0.8,
                        help="Ratio of training set (default: 0.8)")
    
    parser.add_argument("--test", type=float, default=0.1,
                        help="Ratio of test set (default: 0.1)")
    
    parser.add_argument("--val", type=float, default=0.1,
                        help="Ratio of validation set (default: 0.1)")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    parser.add_argument("--merge", type=str, choices=["auto", "2", "3"], default="2",
                        help="Merge function to use (auto, 2, or 3)")
    
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")

    args = parser.parse_args()
        
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    split_dataset(
        src_dir=args.src,
        dest_dir=args.dest,
        train_ratio=args.train,
        test_ratio=args.test,
        val_ratio=args.val,
        debug=args.debug,
        merge_function=args.merge,
        random_state=args.random_state
    )