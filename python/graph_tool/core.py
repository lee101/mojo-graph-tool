"""Small compatible graph and property-map layer for the covered algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, order=True)
class Vertex:
    graph: "Graph"
    index: int

    def __int__(self) -> int:
        return self.index


@dataclass(frozen=True)
class Edge:
    graph: "Graph"
    source_index: int
    target_index: int

    def source(self) -> Vertex:
        return Vertex(self.graph, self.source_index)

    def target(self) -> Vertex:
        return Vertex(self.graph, self.target_index)


class VertexPropertyMap:
    """Array-backed vertex property map with graph-tool's useful ``.a`` view."""

    def __init__(self, graph: "Graph", value_type: str = "double", vals=None):
        self.graph = graph
        dtype = np.float64 if value_type in {"double", "float", "long double"} else np.int64
        self.value_type = value_type
        self.a = np.zeros(graph.num_vertices(), dtype=dtype) if vals is None else np.asarray(vals, dtype=dtype)
        if self.a.ndim != 1 or self.a.size != graph.num_vertices():
            raise ValueError("vertex property values must be a one-dimensional array of graph size")

    @property
    def fa(self):
        return self.a

    def __getitem__(self, vertex):
        return self.a[int(vertex)]

    def __setitem__(self, vertex, value):
        self.a[int(vertex)] = value

    def copy(self):
        return VertexPropertyMap(self.graph, self.value_type, self.a.copy())


class EdgePropertyMap:
    def __init__(self, graph: "Graph", value_type: str = "double", vals=None):
        self.graph = graph
        dtype = np.float64 if value_type in {"double", "float", "long double"} else np.int64
        self.value_type = value_type
        self.a = np.zeros(graph.num_edges(), dtype=dtype) if vals is None else np.asarray(vals, dtype=dtype)
        if self.a.ndim != 1 or self.a.size != graph.num_edges():
            raise ValueError("edge property values must be a one-dimensional array of graph size")

    def __getitem__(self, edge):
        return self.a[self.graph._edges.index((edge.source_index, edge.target_index))]

    def __setitem__(self, edge, value):
        self.a[self.graph._edges.index((edge.source_index, edge.target_index))] = value


class Graph:
    """Mutable directed/undirected graph using integer vertex descriptors."""

    def __init__(self, g=None, directed: bool = True, prune: bool = False):
        del prune
        self._directed = bool(directed)
        self._n = 0
        self._edges: list[tuple[int, int]] = []
        self._csr_cache: dict[tuple[str, bool], tuple[np.ndarray, ...]] = {}
        if g is not None:
            self._directed = g.is_directed()
            self.add_vertex(g.num_vertices())
            self.add_edge_list((int(e.source()), int(e.target())) for e in g.edges())

    def add_vertex(self, n: int = 1):
        n = int(n)
        if n < 0:
            raise ValueError("vertex count must be non-negative")
        first = self._n
        self._n += n
        self._csr_cache.clear()
        return Vertex(self, first) if n == 1 else None

    def add_edge(self, source, target):
        source, target = int(source), int(target)
        if min(source, target) < 0 or max(source, target) >= self._n:
            raise ValueError("edge endpoint is not a vertex in this graph")
        self._edges.append((source, target))
        self._csr_cache.clear()
        return Edge(self, source, target)

    def add_edge_list(self, edges: Iterable):
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def num_vertices(self) -> int:
        return self._n

    def num_edges(self) -> int:
        return len(self._edges)

    def is_directed(self) -> bool:
        return self._directed

    def set_directed(self, directed: bool):
        self._directed = bool(directed)
        self._csr_cache.clear()

    def vertex(self, index: int) -> Vertex:
        index = int(index)
        if not 0 <= index < self._n:
            raise ValueError("invalid vertex")
        return Vertex(self, index)

    def vertices(self):
        return (Vertex(self, i) for i in range(self._n))

    def edges(self):
        return (Edge(self, s, t) for s, t in self._edges)

    def new_vertex_property(self, value_type: str = "double", vals=None) -> VertexPropertyMap:
        return VertexPropertyMap(self, value_type, vals)

    new_vp = new_vertex_property

    def new_edge_property(self, value_type: str = "double", vals=None) -> EdgePropertyMap:
        return EdgePropertyMap(self, value_type, vals)

    new_ep = new_edge_property

    def _csr(self, mode: str = "out", weights=None, simple: bool = False):
        """Return sorted CSR, optionally with a one-way weight per CSR slot."""
        cache_key = (mode, simple)
        if weights is None and cache_key in self._csr_cache:
            return self._csr_cache[cache_key]
        rows: list[list[tuple[int, float]]] = [[] for _ in range(self._n)]
        w = None if weights is None else edge_values(weights, self)
        for i, (source, target) in enumerate(self._edges):
            value = 1.0 if w is None else float(w[i])
            if value < 0:
                raise ValueError("edge weights must be non-negative")
            if mode == "out":
                rows[source].append((target, value))
                if not self._directed:
                    rows[target].append((source, value))
            elif mode == "in":
                rows[target].append((source, value))
                if not self._directed:
                    rows[source].append((target, value))
            elif mode == "undirected":
                rows[source].append((target, value))
                if source != target:
                    rows[target].append((source, value))
            else:
                raise ValueError("unknown CSR mode")
        if simple:
            rows = [sorted({target for target, _ in row}) for row in rows]
            offsets = np.empty(self._n + 1, dtype=np.int64)
            offsets[0] = 0
            for i, row in enumerate(rows):
                offsets[i + 1] = offsets[i] + len(row)
            neighbors = np.fromiter((x for row in rows for x in row), dtype=np.int64,
                                    count=int(offsets[-1]))
            result = (offsets, neighbors)
            if weights is None:
                self._csr_cache[cache_key] = result
            return result
        rows = [sorted(row) for row in rows]
        offsets = np.empty(self._n + 1, dtype=np.int64)
        offsets[0] = 0
        for i, row in enumerate(rows):
            offsets[i + 1] = offsets[i] + len(row)
        count = int(offsets[-1])
        neighbors = np.empty(count, dtype=np.int64)
        values = np.empty(count, dtype=np.float64)
        pos = 0
        for row in rows:
            for target, value in row:
                neighbors[pos], values[pos] = target, value
                pos += 1
        result = (offsets, neighbors, values)
        if weights is None:
            self._csr_cache[cache_key] = result
        return result


def require_vertex_property(prop, graph, dtype, name):
    """Validate an output property before exposing its buffer to native code."""
    if not isinstance(prop, VertexPropertyMap) or prop.graph is not graph:
        raise ValueError(f"{name} must be a vertex property belonging to g")
    array = prop.a
    if not isinstance(array, np.ndarray) or array.dtype != dtype or array.ndim != 1:
        raise TypeError(f"{name} must have a one-dimensional {dtype} NumPy array")
    if array.size != graph.num_vertices() or not array.flags.c_contiguous or not array.flags.writeable:
        raise ValueError(f"{name} must be a writable, contiguous vertex property of g")
    return array


def edge_values(values, graph, name="weight"):
    """Make a checked Float64 copy suitable for a native CSR call."""
    if isinstance(values, EdgePropertyMap):
        if values.graph is not graph:
            raise ValueError(f"{name} must belong to g")
        values = values.a
    array = np.asarray(values)
    if array.ndim != 1 or array.size != graph.num_edges():
        raise ValueError(f"{name} must have one value per edge")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(array.dtype, np.integer) and (
        np.any(array > 2**53) or (np.issubdtype(array.dtype, np.signedinteger) and np.any(array < -(2**53)))
    ):
        raise ValueError(f"{name} integers must be exactly representable as Float64")
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"{name} dtype cannot be narrowed to Float64")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result
