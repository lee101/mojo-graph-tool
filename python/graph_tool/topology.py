"""Connectivity, unweighted distance, and core decomposition."""

from __future__ import annotations

import numpy as np

from ._lib import address, lib
from .core import require_vertex_property


def shortest_distance(g, source=None, target=None, weights=None, max_dist=None,
                      directed=None, pred_map=False, dist_map=None):
    if weights is not None:
        raise NotImplementedError("weighted shortest paths are outside the covered subset")
    if pred_map:
        raise NotImplementedError("predecessor maps are outside the covered subset")
    if source is None:
        raise NotImplementedError("all-pairs distances are outside the covered subset")
    source = int(source)
    if not 0 <= source < g.num_vertices():
        raise ValueError("source is not a vertex in g")
    if target is not None and not 0 <= int(target) < g.num_vertices():
        raise ValueError("target is not a vertex in g")
    if max_dist is not None and int(max_dist) < 0:
        raise ValueError("max_dist must be non-negative")
    use_directed = g.is_directed() if directed is None else bool(directed)
    mode = "out" if use_directed else "undirected"
    offsets, neighbors = g._csr(mode, simple=True)
    result = g.new_vertex_property("int64_t") if dist_map is None else dist_map
    result_array = require_vertex_property(result, g, np.dtype(np.int64), "dist_map")
    queue = np.empty(g.num_vertices(), dtype=np.int64)
    lib().mgt_bfs(address(offsets), address(neighbors), address(result_array), address(queue),
                  g.num_vertices(), source, -1 if max_dist is None else int(max_dist))
    return result[int(target)] if target is not None else result


def _strong_components(g):
    n = g.num_vertices()
    out, neighbors = g._csr("out", simple=True)
    rev, incoming = g._csr("in", simple=True)
    seen = np.zeros(n, dtype=bool)
    order = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, int(out[start]))]
        while stack:
            vertex, cursor = stack[-1]
            if cursor < int(out[vertex + 1]):
                stack[-1] = (vertex, cursor + 1)
                neighbor = int(neighbors[cursor])
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append((neighbor, int(out[neighbor])))
            else:
                order.append(vertex)
                stack.pop()
    labels = np.full(n, -1, dtype=np.int64)
    count = 0
    for start in reversed(order):
        if labels[start] >= 0:
            continue
        labels[start] = count
        stack = [start]
        while stack:
            vertex = stack.pop()
            for slot in range(int(rev[vertex]), int(rev[vertex + 1])):
                neighbor = int(incoming[slot])
                if labels[neighbor] < 0:
                    labels[neighbor] = count
                    stack.append(neighbor)
        count += 1
    return labels, count


def label_components(g, vprop=None, directed=None, attractors=False):
    if attractors:
        raise NotImplementedError("attractor reporting is outside the covered subset")
    use_directed = g.is_directed() if directed is None else bool(directed)
    result = g.new_vertex_property("int64_t") if vprop is None else vprop
    result_array = require_vertex_property(result, g, np.dtype(np.int64), "vprop")
    if use_directed:
        result_array[:], count = _strong_components(g)
    else:
        offsets, neighbors = g._csr("undirected", simple=True)
        queue = np.empty(g.num_vertices(), dtype=np.int64)
        count = lib().mgt_components(address(offsets), address(neighbors), address(result_array), address(queue),
                                     g.num_vertices())
    hist = np.bincount(result.a, minlength=count).astype(np.int64)
    return result, hist


def kcore_decomposition(g, deg=None):
    if g.is_directed():
        raise ValueError("k-core decomposition requires an undirected graph")
    offsets, neighbors = g._csr("undirected", simple=True)
    result = g.new_vertex_property("int64_t") if deg is None else deg
    result_array = require_vertex_property(result, g, np.dtype(np.int64), "deg")
    work = np.empty(g.num_vertices(), dtype=np.int64)
    bins = np.empty(g.num_vertices() + 1, dtype=np.int64)
    order = np.empty(g.num_vertices(), dtype=np.int64)
    position = np.empty(g.num_vertices(), dtype=np.int64)
    lib().mgt_kcore(address(offsets), address(neighbors), address(work), address(result_array),
                    address(bins), address(order), address(position), g.num_vertices())
    return result
