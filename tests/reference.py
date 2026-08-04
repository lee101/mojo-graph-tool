"""Small independent reference implementations used when graph-tool is unavailable."""

from __future__ import annotations

import numpy as np


def pagerank_reference(n, edges, damping=0.85, personalization=None, weights=None,
                       epsilon=1e-12, max_iter=10_000):
    if personalization is None:
        personalization = np.full(n, 1 / n)
    else:
        personalization = np.asarray(personalization, dtype=float)
        personalization = personalization / personalization.sum()
    weights = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=float)
    strength = np.zeros(n)
    for (source, _), value in zip(edges, weights):
        strength[source] += value
    rank = np.full(n, 1 / n)
    for _ in range(max_iter):
        nxt = (1 - damping) * personalization + damping * rank[strength == 0].sum() * personalization
        for (source, target), value in zip(edges, weights):
            if strength[source]:
                nxt[target] += damping * rank[source] * value / strength[source]
        if np.abs(nxt - rank).sum() < epsilon:
            return nxt
        rank = nxt
    raise AssertionError("reference PageRank did not converge")


def bfs_reference(n, edges, source, directed=True, max_dist=None):
    rows = [[] for _ in range(n)]
    for left, right in edges:
        rows[left].append(right)
        if not directed:
            rows[right].append(left)
    distance = np.full(n, -1, dtype=np.int64)
    distance[source] = 0
    queue = [source]
    for vertex in queue:
        if max_dist is not None and distance[vertex] >= max_dist:
            continue
        for neighbor in rows[vertex]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[vertex] + 1
                queue.append(neighbor)
    return distance
