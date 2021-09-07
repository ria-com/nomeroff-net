import numpy as np


def prepare_multiline_rects(rects, zones, lines):
    """
    :param rects: rectangles with CRAFT-matched letters zones
    :param zones: normalized image parts
    :param lines: spetial dict with stucture
    :return: updated rectangles for joining, oneline numberplate builder configuration
    """
    new_zones = []
    np_config = {}
    for idx in lines.keys():
        if len(lines[idx]) > 1:
            lines[idx] = sorted([line for line in lines[idx]], key=lambda x: rects[x['idx']][0][0])
        lines_arr = lines[idx]
        right_zone = None
        left_zone = None
        if idx == 2:
            if len(lines_arr) == 1:
                zone = zones[lines_arr[0]['idx']]
                w = int(zone.shape[1]/2.5)
                left_zone = zone[:, :w]
                right_zone = zone[:, w:]
            if len(lines_arr) == 2:
                left_zone = zones[lines_arr[0]['idx']]
                right_zone = zones[lines_arr[1]['idx']]
            if len(lines_arr) > 2:
                left_zone = zones[lines_arr[0]['idx']]
                right_zone = np.hstack([zones[line['idx']] for line in lines_arr[1:]])
            if right_zone is not None:
                new_zones.append(right_zone)
            if left_zone is not None:
                new_zones.append(left_zone)
        else:
            for line in lines[idx]:
                new_zones.append(zones[line['idx']])
    return new_zones, np_config
