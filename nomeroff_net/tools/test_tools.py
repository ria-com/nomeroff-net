"""
    python3 nomeroff_net/tools/test_tools.py
"""
import os
from termcolor import colored
from collections import Counter


def get_all_files(folder):
    file_list = []
    if not os.path.exists(folder):
        return file_list

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def path_filter(p):
    if p.split(".")[-1] in ("py", "ipynb")\
            and not os.path.basename(p).startswith("_"):
        return True


def test_filter(test_str):
    test_str = test_str.strip()
    if test_str and not test_str.startswith("#"):
        return True


def check_test_coverage(test_script_file, dirs_for_test, test_dir="./"):
    all_files = []
    for dir_for_test in dirs_for_test:
        all_files.extend(get_all_files(dir_for_test))

    with open(test_script_file, 'r') as stream:
        tests = stream.readlines()
    tests = filter(test_filter, tests)
    tests = [os.path.join(test_dir, test.split(" ")[-1]) for test in tests]
    tests = [os.path.normpath(test.strip()) for test in tests]

    filtered_files = filter(path_filter, all_files)
    c = Counter()
    for i, path in enumerate(filtered_files):
        path = os.path.normpath(path)
        c["all"] += 1
        if path in tests:
            print(colored(f'{i}. {path}', "green"))
            c["good"] += 1
        else:
            print(colored(f'{i}. {path}', "red"))
            c["bad"] += 1
    c["percentage"] = c["good"]/c["all"]
    return c


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    test_file = os.path.join(current_dir, "../../.github/workflows/nn-ci-cpu-testing.yml")

    dirs = [
        os.path.join(current_dir, "../../nomeroff_net"),
        os.path.join(current_dir, "../../examples")
    ]

    stat = check_test_coverage(test_file, dirs, os.path.join(current_dir, "../../"))
    print(stat)
