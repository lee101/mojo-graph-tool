"""Benchmark this port against the conda-forge graph-tool build."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from graph_tool import centrality, clustering, topology
from graph_tool.core import Graph


def edges_for(n, m, seed=42):
    rng = np.random.default_rng(seed)
    edges = np.column_stack((rng.integers(0, n, m), rng.integers(0, n, m)))
    return [tuple(map(int, edge)) for edge in edges if edge[0] != edge[1]]


def make_graph(n, m, directed):
    graph = Graph(directed=directed)
    graph.add_vertex(n)
    graph.add_edge_list(edges_for(n, m))
    return graph


def best(fn, reps=3):
    elapsed = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        elapsed = min(elapsed, time.perf_counter() - start)
    return elapsed


UPSTREAM = r'''
import os, sys, time, numpy as np
root, kernel, n, m = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != os.path.join(root, "python")]
import graph_tool.all as gt
from graph_tool.centrality import pagerank
from graph_tool.clustering import local_clustering
from graph_tool.topology import kcore_decomposition
rng = np.random.default_rng(42)
raw = np.column_stack((rng.integers(0, n, m), rng.integers(0, n, m)))
edges = [tuple(map(int, edge)) for edge in raw if edge[0] != edge[1]]
g = gt.Graph(directed=(kernel == "pagerank")); g.add_vertex(n); g.add_edge_list(edges)
fn = {"pagerank": lambda: pagerank(g, epsilon=1e-8),
      "clustering": lambda: local_clustering(g),
      "kcore": lambda: kcore_decomposition(g)}[kernel]
best = float("inf")
for _ in range(3):
    start = time.perf_counter(); fn(); best = min(best, time.perf_counter() - start)
print(best)
'''


def upstream(kernel, n, m):
    env = os.environ.copy()
    completed = subprocess.run([sys.executable, "-c", UPSTREAM, ROOT, kernel, str(n), str(m)],
                               text=True, capture_output=True, check=True, timeout=300, env=env)
    return float(completed.stdout.strip())


def report(name, mojo_seconds, reference_seconds):
    speedup = reference_seconds / mojo_seconds
    print(f"| {name} | {mojo_seconds * 1e3:.2f} ms | {reference_seconds * 1e3:.2f} ms | {speedup:.2f}x |")


def main():
    print(f"Machine: {platform.platform()} | Python {platform.python_version()}")
    print("| Kernel | Mojo port | graph-tool 3.6 | Speedup |")
    print("|---|---:|---:|---:|")

    n, m = 30_000, 300_000
    g = make_graph(n, m, directed=True)
    mojo = best(lambda: centrality.pagerank(g, epsilon=1e-8))
    report(f"PageRank ({n:,} vertices, {m:,} arcs)", mojo, upstream("pagerank", n, m))

    n, m = 12_000, 100_000
    g = make_graph(n, m, directed=False)
    mojo = best(lambda: clustering.local_clustering(g))
    report(f"local_clustering ({n:,} vertices, {m:,} edges)", mojo, upstream("clustering", n, m))

    n, m = 3_000, 18_000
    g = make_graph(n, m, directed=False)
    mojo = best(lambda: topology.kcore_decomposition(g))
    report(f"kcore_decomposition ({n:,} vertices, {m:,} edges)", mojo, upstream("kcore", n, m))


if __name__ == "__main__":
    main()
