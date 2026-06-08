"""Helpers for working with the CUDA primary context from PyCUDA."""

from __future__ import annotations

import threading

import pycuda.driver as cuda


_context_lock = threading.Lock()
_primary_context = None


def get_primary_context(device_id: int = 0):
    """Return a retained primary CUDA context for the selected device."""
    global _primary_context

    if _primary_context is None:
        with _context_lock:
            if _primary_context is None:
                cuda.init()
                _primary_context = cuda.Device(device_id).retain_primary_context()
    return _primary_context


class primary_cuda_context:
    """Make the retained primary CUDA context current for the lifetime of the block."""

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self.context = None

    def __enter__(self):
        self.context = get_primary_context(self.device_id)
        self.context.push()
        return self.context

    def __exit__(self, exc_type, exc, tb) -> None:
        cuda.Context.pop()
