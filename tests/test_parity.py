import numpy as np
import pytest
import json
import os
import subprocess
import sys

import graph_tool.all as gt
from graph_tool import centrality, clustering, stats, topology

from reference import bfs_reference, pagerank_reference


def graph(edges, n, directed=True):
    g = gt.Graph(directed=directed)
    g.add_vertex(n)
    g.add_edge_list(edges)
    return g


def test_pagerank_matches_independent_reference_with_dangling_vertex():
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (3, 2)]
    g = graph(edges, 5)
    result, iterations = centrality.pagerank(g, epsilon=1e-12, ret_iter=True)
    expected = pagerank_reference(5, edges)
    assert iterations > 0
    assert result.a == pytest.approx(expected, abs=2e-12)
    assert result.a.sum() == pytest.approx(1.0)


def test_weighted_personalized_pagerank_matches_reference_and_reuses_prop():
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
    weights = np.array([2.0, 1.0, 4.0, 3.0, 2.0])
    pers = np.array([0.0, 1.0, 2.0, 0.0])
    g = graph(edges, 4)
    out = g.new_vp("double")
    got = centrality.pagerank(g, weight=weights, pers=pers, prop=out, epsilon=1e-12)
    assert got is out
    assert got.a == pytest.approx(pagerank_reference(4, edges, weights=weights, personalization=pers), abs=2e-12)


def test_pagerank_known_published_cycle_vector():
    g = graph([(0, 1), (1, 2), (2, 0)], 3)
    assert centrality.pagerank(g, epsilon=1e-14).a == pytest.approx([1 / 3] * 3, abs=1e-14)


def test_pagerank_simd_tail_and_parallel_threshold():
    tail = graph([(i, (i + 1) % 17) for i in range(17)], 17)
    assert centrality.pagerank(tail, epsilon=1e-14).a == pytest.approx([1 / 17] * 17, abs=1e-14)
    n = 65_536
    parallel = graph([(i, (i + 1) % n) for i in range(n)], n)
    assert np.allclose(centrality.pagerank(parallel, epsilon=1e-14).a, 1 / n, atol=1e-14)


def test_bfs_matches_reference_and_target_scalar():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    g = graph(edges, 7)
    result = topology.shortest_distance(g, source=g.vertex(0))
    assert result.a.tolist() == bfs_reference(7, edges, 0).tolist()
    assert topology.shortest_distance(g, source=0, target=5) == 4


def test_bfs_undirected_and_distance_cutoff():
    edges = [(0, 1), (1, 2), (2, 3)]
    g = graph(edges, 4, directed=True)
    got = topology.shortest_distance(g, source=3, directed=False, max_dist=2)
    assert got.a.tolist() == bfs_reference(4, edges, 3, directed=False, max_dist=2).tolist()


def test_label_components_weak_and_strong():
    g = graph([(0, 1), (1, 0), (1, 2), (3, 4), (4, 3)], 6)
    weak, weak_hist = topology.label_components(g, directed=False)
    assert sorted(weak_hist.tolist()) == [1, 2, 3]
    assert weak.a[0] == weak.a[2] and weak.a[3] != weak.a[0]
    strong, strong_hist = topology.label_components(g, directed=True)
    assert sorted(strong_hist.tolist()) == [1, 1, 2, 2]
    assert strong.a[0] == strong.a[1] != strong.a[2]
    assert strong.a[3] == strong.a[4]


def test_kcore_matches_known_nested_core_numbers():
    # K4 on 0..3, plus a degree-one leaf and a triangle joined through vertex 3.
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
             (0, 4), (3, 5), (5, 6), (5, 7), (6, 7)]
    got = topology.kcore_decomposition(graph(edges, 8, directed=False))
    assert got.a.tolist() == [3, 3, 3, 3, 1, 2, 2, 2]


def test_local_and_global_clustering_match_complete_and_path_vectors():
    complete = graph([(i, j) for i in range(4) for j in range(i + 1, 4)], 4, directed=False)
    assert clustering.local_clustering(complete).a == pytest.approx(np.ones(4))
    (coefficient, error), triangles, wedges = clustering.global_clustering(complete, ret_counts=True)
    assert coefficient == pytest.approx(1.0)
    assert error == 0.0
    assert (triangles, wedges) == (4, 12)
    path = graph([(0, 1), (1, 2), (2, 3)], 4, directed=False)
    assert clustering.local_clustering(path).a == pytest.approx(np.zeros(4))
    assert clustering.global_clustering(path) == (0.0, 0.0)


def test_clustering_triangle_with_tail():
    g = graph([(0, 1), (1, 2), (2, 0), (0, 3)], 4, directed=False)
    assert clustering.local_clustering(g).a == pytest.approx([1 / 3, 1.0, 1.0, 0.0])
    assert clustering.global_clustering(g) == pytest.approx((0.6, np.sqrt(0.18)))


def test_statistics_histograms_and_moments_match_numpy():
    g = graph([(0, 1), (0, 2), (2, 1)], 3)
    counts, bins = stats.vertex_hist(g, "out", bins=[0, 1, 2, 3], float_count=False)
    assert np.array_equal(counts, np.histogram([2, 0, 1], bins=[0, 1, 2, 3])[0])
    assert np.array_equal(bins, [0, 1, 2, 3])
    values = np.array([2.0, 3.0, 8.0])
    assert stats.edge_average(g, values) == pytest.approx((values.mean(), values.std()))
    assert stats.vertex_average(g, np.array([1.0, 3.0, 8.0])) == pytest.approx((4.0, np.std([1.0, 3.0, 8.0])))


def test_graph_and_property_map_surface():
    g = gt.Graph(directed=False)
    v = g.add_vertex()
    g.add_vertex(2)
    g.add_edge(v, 1)
    prop = g.new_vertex_property("double")
    prop[g.vertex(1)] = 4.5
    assert [int(x) for x in g.vertices()] == [0, 1, 2]
    assert prop.fa.tolist() == [0.0, 4.5, 0.0]


def test_unweighted_csr_cache_is_reused_and_invalidated():
    g = graph([(0, 1)], 3)
    first = g._csr("out")
    assert g._csr("out") is first
    g.add_edge(1, 2)
    offsets, neighbors, _ = g._csr("out")
    assert offsets.tolist() == [0, 1, 2, 2]
    assert neighbors.tolist() == [1, 2]


def test_unsupported_weighted_shortest_path_fails_explicitly():
    g = graph([(0, 1)], 2)
    with pytest.raises(NotImplementedError):
        topology.shortest_distance(g, source=0, weights=np.array([1.0]))


def test_native_calls_reject_wrong_output_maps_and_lossy_weights():
    g = graph([(0, 1)], 2)
    foreign = graph([], 2).new_vp("double")
    with pytest.raises(ValueError):
        centrality.pagerank(g, prop=foreign)
    with pytest.raises(TypeError):
        centrality.pagerank(g, prop=g.new_vp("int64_t"))
    output = g.new_vp("double")
    output.a = output.a[::-1]
    with pytest.raises(ValueError):
        centrality.pagerank(g, prop=output)
    with pytest.raises(ValueError):
        centrality.pagerank(g, weight=np.array([2**53 + 1], dtype=np.int64))


def test_edge_hist_and_explicitly_unsupported_clustering_modes():
    g = graph([(0, 1), (1, 2)], 3)
    counts, bins = stats.edge_hist(g, np.array([1.0, 2.0]), bins=[0, 1.5, 3], float_count=False)
    assert np.array_equal(counts, [1, 1])
    assert np.array_equal(bins, [0, 1.5, 3])
    with pytest.raises(NotImplementedError):
        clustering.local_clustering(g)
    with pytest.raises(NotImplementedError):
        clustering.global_clustering(g)


def _upstream_vectors():
    """Ask the conda-forge graph-tool installation without importing this port."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = r'''
import json, os, sys
root = sys.argv[1]
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != os.path.join(root, "python")]
import graph_tool.all as gt
from graph_tool.centrality import pagerank
from graph_tool.clustering import local_clustering, global_clustering
from graph_tool.topology import kcore_decomposition
edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (3, 2)]
g = gt.Graph(directed=True); g.add_vertex(5); g.add_edge_list(edges)
u_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4), (3, 5), (5, 6), (5, 7), (6, 7)]
u = gt.Graph(directed=False); u.add_vertex(8); u.add_edge_list(u_edges)
triangle = gt.Graph(directed=False); triangle.add_vertex(4); triangle.add_edge_list([(0, 1), (1, 2), (2, 0), (0, 3)])
print(json.dumps({"pagerank": pagerank(g, epsilon=1e-12).a.tolist(),
                  "kcore": kcore_decomposition(u).a.tolist(),
                  "local": local_clustering(triangle).a.tolist(),
                  "global": global_clustering(triangle)}))
'''
    try:
        completed = subprocess.run([sys.executable, "-c", code, root], check=True,
                                   text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("conda-forge graph-tool is not importable in this environment")
    return json.loads(completed.stdout)


def test_parity_against_conda_forge_graph_tool():
    ref = _upstream_vectors()
    directed = graph([(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (3, 2)], 5)
    assert centrality.pagerank(directed, epsilon=1e-12).a == pytest.approx(ref["pagerank"], abs=2e-12)
    core_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                  (0, 4), (3, 5), (5, 6), (5, 7), (6, 7)]
    assert topology.kcore_decomposition(graph(core_edges, 8, directed=False)).a.tolist() == ref["kcore"]
    triangle = graph([(0, 1), (1, 2), (2, 0), (0, 3)], 4, directed=False)
    assert clustering.local_clustering(triangle).a == pytest.approx(ref["local"])
    assert clustering.global_clustering(triangle) == pytest.approx(ref["global"])
