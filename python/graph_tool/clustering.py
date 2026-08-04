"""Triangle-based clustering coefficients."""

from __future__ import annotations

import numpy as np

from ._lib import address, lib
from .core import require_vertex_property


def local_clustering(g, undirected: bool | None = None, c=None):
    if undirected is False or (g.is_directed() and undirected is not True):
        raise NotImplementedError("directed clustering is outside the covered subset")
    offsets, neighbors = g._csr("undirected", simple=True)
    result = g.new_vertex_property("double") if c is None else c
    result_array = require_vertex_property(result, g, np.dtype(np.float64), "c")
    triangles = np.empty(g.num_vertices(), dtype=np.int64)
    lib().mgt_local_clustering(address(offsets), address(neighbors), address(triangles),
                               address(result_array), g.num_vertices())
    return result


def global_clustering(g, weight=None, ret_counts: bool = False, sampled: bool = False, m: int = 1000):
    """Return ``(coefficient, jackknife_error)`` as graph-tool does."""
    del m
    if weight is not None:
        raise NotImplementedError("weighted clustering is outside the covered subset")
    if sampled:
        raise NotImplementedError("sampled clustering is outside the covered subset")
    if g.is_directed():
        raise NotImplementedError("directed clustering is outside the covered subset")
    offsets, neighbors = g._csr("undirected", simple=True)
    triangles = np.empty(g.num_vertices(), dtype=np.int64)
    local = np.empty(g.num_vertices(), dtype=np.float64)
    coefficient = lib().mgt_local_clustering(address(offsets), address(neighbors), address(triangles),
                                              address(local), g.num_vertices())
    degrees = np.diff(offsets)
    wedge_per_vertex = degrees * (degrees - 1) // 2
    triangle_per_vertex = local * wedge_per_vertex
    wedges = int(wedge_per_vertex.sum())
    triangles_total = float(triangle_per_vertex.sum())
    error2 = 0.0
    if wedges:
        for triangles, vertex_wedges in zip(triangle_per_vertex, wedge_per_vertex):
            remaining = wedges - vertex_wedges
            if remaining:
                leave_one_out = (triangles_total - triangles) / remaining
                error2 += (coefficient - leave_one_out) ** 2
    result = (coefficient, float(np.sqrt(error2)))
    if not ret_counts:
        return result
    triangles = int(round(coefficient * wedges / 3))
    return result, triangles, wedges
