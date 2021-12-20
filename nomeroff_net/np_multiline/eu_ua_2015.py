import math


def split_zones(zones):
    cnt = math.ceil(len(zones)/2)
    left_zones = zones[:cnt]
    right_zones = zones[cnt:]
    return left_zones, right_zones


def sort_line(line, rects):
    if len(line) > 1:
        line = sorted([line for line in line], key=lambda x: rects[x['idx']][0][0])
    return line


def add_to_zones_right_zones(lines_count, new_zones, right_zone, right_zones):
    if lines_count == 2 and right_zone is not None:
        new_zones.append(right_zone)
    if lines_count == 2 and right_zones is not None:
        for tmp_zone in right_zones:
            new_zones.append(tmp_zone)
    return new_zones


def add_to_zones_left_zones(new_zones, left_zone, left_zones):
    if left_zone is not None:
        new_zones.append(left_zone)
    if left_zones is not None:
        for tmp_zone in left_zones:
            new_zones.append(tmp_zone)
    return new_zones


def define_left_and_right_zones(lines_arr, zones):
    left_zone = None
    right_zone = None
    left_zones = None
    right_zones = None
    if len(lines_arr) == 1:
        zone = zones[lines_arr[0]['idx']]
        w = int(zone.shape[1] / 2)
        left_zone = zone[:, :w]
        right_zone = zone[:, w:]
    elif len(lines_arr) > 2:
        left_zones, right_zones = split_zones([zones[line['idx']] for line in lines_arr])
    if len(lines_arr) == 2:
        left_zone = zones[lines_arr[0]['idx']]
        right_zone = zones[lines_arr[1]['idx']]
    return left_zone, right_zone, left_zones, right_zones


def prepare_multiline_rects(rects, zones, lines):
    """
    :param rects: rectangles with CRAFT-matched letters zones
    :param zones: normalized image parts
    :param lines: spetial dict with stucture
    :return: updated rectangles for joining, oneline numberplate builder configuration
    """
    new_zones = []
    np_config = {}
    lines_count = len(lines.keys())
    right_zone = None
    right_zones = None
    for idx in lines.keys():
        lines[idx] = sort_line(lines[idx], rects)
        lines_arr = lines[idx]
        if idx == 1 and lines_count == 2:
            left_zone, right_zone, left_zones, right_zones = define_left_and_right_zones(lines_arr, zones)
            new_zones = add_to_zones_left_zones(new_zones, left_zone, left_zones)
        else:
            for line in lines[idx]:
                new_zones.append(zones[line['idx']])
    new_zones = add_to_zones_right_zones(lines_count, new_zones, right_zone, right_zones)
    return new_zones, np_config
