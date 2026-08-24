# mojo-graph-tool

`mojo-graph-tool` is a standalone Mojo port of a useful, tested subset of
[graph-tool](https://graph-tool.skewed.de/): sparse graph algorithms and graph
statistics with a Python API shaped like `graph_tool`. It is deliberately a
small compatibility layer, not a second implementation of graph-tool's large
property-map, filtering, drawing, inference, and I/O systems.

## Covered subset

The package supports `import graph_tool.all as gt`, mutable `Graph` objects,
integer vertex descriptors, edge lists, and NumPy-backed vertex/edge property
maps (`.a` / `.fa`). Its covered algorithm calls are:

- `graph_tool.centrality.pagerank`, including weights, personalization,
  dangling vertices, convergence tolerance, and `ret_iter`.
- `graph_tool.topology.shortest_distance` for single-source unweighted BFS,
  `label_components` (weak and strong), and `kcore_decomposition`.
- `graph_tool.clustering.local_clustering` and `global_clustering`, including
  graph-tool's jackknife error and triangle/triple counts.
- `graph_tool.stats.vertex_hist`, `edge_hist`, `vertex_average`, and
  `edge_average`.

Weighted shortest paths, directed/sampled/weighted clustering, all-pairs
distances, predecessor maps, graph filters, parallel edges as distinct
topological relationships, I/O, layout, inference, and the rest of graph-tool
are intentionally outside this release. Unsupported covered-call variants
raise `NotImplementedError` instead of producing an approximation silently.

## Install and run

Mojo is pinned in `pixi.toml`, and conda-forge `graph-tool` is included for
parity tests and benchmarks.

```bash
pixi install
pixi run build
pixi run test
```

The shared library is written to `dist/libmojo-graph-tool.so`. Run Python only
through Pixi so that its `PYTHONPATH=python` activation is present.

```bash
pixi run python - <<'PY'
import graph_tool.all as gt

g = gt.Graph(directed=True)
g.add_vertex(4)
g.add_edge_list([(0, 1), (0, 2), (1, 2), (2, 0)])

rank = gt.pagerank(g, epsilon=1e-12)
distance = gt.shortest_distance(g, source=0)
print(rank.a.round(6).tolist())
print(distance.a.tolist())
PY
```

## How it works

Python owns a compact edge list and materializes sorted, contiguous CSR arrays
for each call. The Mojo shared library has one compilation unit and accepts all
buffers as integer addresses over a C ABI; it rebuilds typed pointers internally
using `AnyOrigin[mut=True]`. The caller owns every output and scratch array, so
there is no native allocation or cross-language lifetime to manage. PageRank
uses incoming CSR plus outgoing strengths; BFS/components use queue scratch;
clustering counts each vertex's triangles by intersecting sorted neighbour rows;
k-core uses exact degree peeling. PageRank precomputes source contributions and
uses SIMD for its dense update/reduction passes. Large clustering calls split
independent vertex ranges across a persistent host thread pool, while smaller
calls stay serial to avoid launch overhead.

## Validation

`tests/` has independent pure-Python reference implementations and published
complete-graph/cycle vectors. It also launches the installed conda-forge
`graph-tool` 3.6 in a clean subprocess and compares PageRank, local/global
clustering, and k-core results directly. It includes FFI boundary tests for
foreign, non-contiguous, incorrectly typed, and lossy buffers. The current
suite has 18 tests.

## Benchmark

Measured with `pixi run bench` on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`,
Python 3.13.14, comparing complete public API calls against conda-forge
graph-tool 3.6. Times are the best of three runs.

| Kernel | Mojo port | graph-tool 3.6 | Speedup |
|---|---:|---:|---:|
| PageRank (30,000 vertices, 300,000 arcs) | 8.56 ms | 7.20 ms | 0.84x |
| local_clustering (12,000 vertices, 100,000 edges) | 2.89 ms | 2.12 ms | 0.73x |
| kcore_decomposition (3,000 vertices, 18,000 edges) | 0.26 ms | 0.39 ms | 1.49x |

These kernels are sparse, memory-bound, and branch-heavy: their arithmetic
intensity is below the threshold where host/device transfers can pay for a GPU
launch. Consequently, no GPU path is included; CPU remains the default.
