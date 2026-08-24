"""Shared-library loader and ctypes declarations."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-graph-tool.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mgt_pagerank": ([I, I, I, I, I, I, I, I, I, I, F, F, I], I),
    "mgt_bfs": ([I, I, I, I, I, I, I], I),
    "mgt_components": ([I, I, I, I, I], I),
    "mgt_kcore": ([I, I, I, I, I, I, I, I], None),
    "mgt_local_clustering": ([I, I, I, I, I], F),
    "mgt_local_clustering_range": ([I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the library when it is absent or older than its Mojo source."""
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo is not on PATH; run this package through pixi")
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")],
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def address(array) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("native buffers must be contiguous NumPy arrays")
    pointer = int(array.ctypes.data)
    if array.size and pointer == 0:
        raise ValueError("native buffer has a null data pointer")
    return pointer
