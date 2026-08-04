"""Property and degree summary functions matching graph-tool's common calls."""

from __future__ import annotations

import numpy as np


def _degree(g, deg):
    if deg not in {"in", "out", "total"}:
        raise ValueError("deg must be 'in', 'out', or 'total'")
    out = np.zeros(g.num_vertices(), dtype=np.int64)
    incoming = np.zeros(g.num_vertices(), dtype=np.int64)
    for source, target in g._edges:
        out[source] += 1
        incoming[target] += 1
        if not g.is_directed() and source != target:
            out[target] += 1
            incoming[source] += 1
    if deg == "out":
        return out
    if deg == "in":
        return incoming
    return out + incoming


def vertex_hist(g, deg="out", bins=None, float_count=True):
    values = _degree(g, deg).astype(float)
    counts, edges = np.histogram(values, bins=bins if bins is not None else "auto")
    return (counts.astype(float) if float_count else counts, edges)


def edge_hist(g, eprop, bins=None, float_count=True):
    values = np.asarray(getattr(eprop, "a", eprop), dtype=float)
    if values.size != g.num_edges():
        raise ValueError("eprop must have one value per edge")
    counts, edges = np.histogram(values, bins=bins if bins is not None else "auto")
    return (counts.astype(float) if float_count else counts, edges)


def vertex_average(g, vprop):
    values = np.asarray(getattr(vprop, "a", vprop), dtype=float)
    if values.size != g.num_vertices():
        raise ValueError("vprop must have one value per vertex")
    return float(values.mean()), float(values.std())


def edge_average(g, eprop):
    values = np.asarray(getattr(eprop, "a", eprop), dtype=float)
    if values.size != g.num_edges():
        raise ValueError("eprop must have one value per edge")
    return float(values.mean()), float(values.std())
