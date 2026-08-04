"""Convenience namespace compatible with ``import graph_tool.all as gt``."""

from .centrality import pagerank
from .clustering import global_clustering, local_clustering
from .core import Edge, EdgePropertyMap, Graph, Vertex, VertexPropertyMap
from .stats import edge_average, edge_hist, vertex_average, vertex_hist
from .topology import kcore_decomposition, label_components, shortest_distance

__all__ = [name for name in globals() if not name.startswith("_")]
