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
        for line in lines[idx]:
            new_zones.append(zones[line['idx']])
    return new_zones, np_config
