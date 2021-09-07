from .default import prepare_multiline_rects as prepare_multiline_rects_parent


def prepare_multiline_rects(rects, zones, lines):
    """
    :param rects: rectangles with CRAFT-matched letters zones
    :param zones: normalized image parts
    :param lines: spetial dict with stucture
    :return: updated rectangles for joining, oneline numberplate builder configuration
    """
    new_zones, np_config = prepare_multiline_rects_parent(rects, zones, lines)
    # np_config = {
    #     "padding-left-coef": 2,
    #     "padding-right-coef": 4,
    # }
    return new_zones, np_config
