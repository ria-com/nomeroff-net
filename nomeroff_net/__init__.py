"""Top-level package exports for nomeroff_net."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "4.0.1"

_EXPORTS = {
    "TextDetector": ("nomeroff_net.pipes.number_plate_text_readers.text_detector", "TextDetector"),
    "OptionsDetector": ("nomeroff_net.pipes.number_plate_classificators.options_detector", "OptionsDetector"),
    "Detector": ("nomeroff_net.pipes.number_plate_localizators.yolo_kp_detector", "Detector"),
    "pipeline": ("nomeroff_net.pipelines", "pipeline"),
}


def __getattr__(name: str) -> Any:
    """Load heavy exports lazily so lightweight helpers can import the package."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [*_EXPORTS.keys(), "__version__"]
