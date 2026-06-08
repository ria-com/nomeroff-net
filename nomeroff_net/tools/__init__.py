"""Lazy top-level exports for nomeroff_net.tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "np_split": ("nomeroff_net.tools.splitter", "np_split"),
    "modelhub": ("nomeroff_net.tools.mcm", "modelhub"),
    "get_mode_torch": ("nomeroff_net.tools.mcm", "get_mode_torch"),
    "get_device_name": ("nomeroff_net.tools.mcm", "get_device_name"),
    "chunked_iterable": ("nomeroff_net.tools.pipeline_tools", "chunked_iterable"),
    "unzip": ("nomeroff_net.tools.pipeline_tools", "unzip"),
    "promise_all": ("nomeroff_net.tools.pipeline_tools", "promise_all"),
    "fline": ("nomeroff_net.tools.image_processing", "fline"),
    "distance": ("nomeroff_net.tools.image_processing", "distance"),
    "normalize_color": ("nomeroff_net.tools.image_processing", "normalize_color"),
    "normalize": ("nomeroff_net.tools.image_processing", "normalize"),
    "linear_line_matrix": ("nomeroff_net.tools.image_processing", "linear_line_matrix"),
    "get_y_by_matrix": ("nomeroff_net.tools.image_processing", "get_y_by_matrix"),
    "find_distances": ("nomeroff_net.tools.image_processing", "find_distances"),
    "rotate": ("nomeroff_net.tools.image_processing", "rotate"),
    "build_perspective": ("nomeroff_net.tools.image_processing", "build_perspective"),
    "get_cv_zone_rgb": ("nomeroff_net.tools.image_processing", "get_cv_zone_rgb"),
    "fix_clockwise2": ("nomeroff_net.tools.image_processing", "fix_clockwise2"),
    "minimum_bounding_rectangle": ("nomeroff_net.tools.image_processing", "minimum_bounding_rectangle"),
    "detect_intersection": ("nomeroff_net.tools.image_processing", "detect_intersection"),
    "find_min_x_idx": ("nomeroff_net.tools.image_processing", "find_min_x_idx"),
    "get_mean_distance": ("nomeroff_net.tools.image_processing", "get_mean_distance"),
    "reshape_points": ("nomeroff_net.tools.image_processing", "reshape_points"),
    "generate_image_rotation_variants": ("nomeroff_net.tools.image_processing", "generate_image_rotation_variants"),
    "get_cv_zones_rgb": ("nomeroff_net.tools.image_processing", "get_cv_zones_rgb"),
    "convert_cv_zones_rgb_to_bgr": ("nomeroff_net.tools.image_processing", "convert_cv_zones_rgb_to_bgr"),
    "get_cv_zones_bgr": ("nomeroff_net.tools.image_processing", "get_cv_zones_bgr"),
}


def __getattr__(name: str) -> Any:
    """Load tools lazily to avoid bootstrapping ModelHub in lightweight contexts."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [*_EXPORTS.keys()]
