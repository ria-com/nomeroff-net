import numpy as np


def prepare_multiline_rects(rects, zones, lines):
    """
    :param rects: rectangles with CRAFT-matched letters zones
    :param zones: normalized image parts
    :param lines: spetial dict with stucture
    :return: updated rectangles for joining
    """
    new_zones = []
    lines_count = len(lines.keys())
    for idx in lines.keys():
        if len(lines[idx]) > 1:
            lines[idx] = sorted([line for line in lines[idx]], key=lambda x: rects[x['idx']][0][0])
        lines_arr = lines[idx]
        if idx == 1 and lines_count == 2:
            if len(lines_arr) == 1 or len(lines_arr) > 2:
                if len(lines_arr) == 1:
                    zone = zones[lines_arr[0]['idx']]
                if len(lines_arr) > 2:
                    zone = np.hstack([zones[line['idx']] for line in lines_arr])
                w = int(zone.shape[1]/2)
                left_zone = zone[:, :w]
                right_zone = zone[:, w:]
            if len(lines_arr) == 2:
                left_zone = zones[lines_arr[0]['idx']]
                right_zone = zones[lines_arr[1]['idx']]
            new_zones.append(left_zone)
        else:
            for line in lines[idx]:
                new_zones.append(zones[line['idx']])
    if lines_count == 2:
        new_zones.append(right_zone)
    return new_zones
