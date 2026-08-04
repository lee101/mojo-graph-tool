"""Centrality algorithms backed by Mojo CSR kernels."""

from __future__ import annotations

import numpy as np

from ._lib import address, lib
from .core import edge_values, require_vertex_property


def pagerank(g, damping: float = 0.85, pers=None, weight=None, prop=None,
             epsilon: float = 1e-6, max_iter: int | None = None, ret_iter: bool = False):
    """Return PageRank, using graph-tool's personalization and dangling-node rule."""
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be between zero and one")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter is not None and max_iter <= 0:
        raise ValueError("max_iter must be positive")
    n = g.num_vertices()
    result = g.new_vertex_property("double") if prop is None else prop
    result_array = require_vertex_property(result, g, np.dtype(np.float64), "prop")
    if n == 0:
        return (result, 0) if ret_iter else result
    in_offsets, incoming, weights = g._csr("in", weight)
    out_offsets, _, out_weights = g._csr("out", weight)
    out_strength = np.zeros(n, dtype=np.float64)
    if out_weights.size:
        owners = np.repeat(np.arange(n), np.diff(out_offsets))
        np.add.at(out_strength, owners, out_weights)
    if pers is None:
        personalization = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        raw_pers = np.asarray(getattr(pers, "a", pers))
        if raw_pers.ndim != 1 or raw_pers.size != n or not (
            np.issubdtype(raw_pers.dtype, np.integer) or np.issubdtype(raw_pers.dtype, np.floating)
        ):
            raise ValueError("pers must be a one-dimensional numeric vertex property of g")
        if (np.issubdtype(raw_pers.dtype, np.integer) and (
            np.any(raw_pers > 2**53) or (
                np.issubdtype(raw_pers.dtype, np.signedinteger) and np.any(raw_pers < -(2**53))
            )
        )) or (np.issubdtype(raw_pers.dtype, np.floating) and raw_pers.dtype.itemsize > 8):
            raise TypeError("pers values cannot be narrowed to Float64")
        personalization = np.ascontiguousarray(raw_pers, dtype=np.float64).copy()
        if not np.all(np.isfinite(personalization)) or np.any(personalization < 0) or personalization.sum() == 0:
            raise ValueError("pers must be a non-negative vertex property with positive sum")
        personalization /= personalization.sum()
    result_array[:] = 1.0 / n
    scratch = np.empty(n, dtype=np.float64)
    iterations = lib().mgt_pagerank(address(in_offsets), address(incoming), address(weights),
                                   address(out_strength), address(personalization), address(result_array),
                                   address(scratch), n, damping, float(epsilon), int(max_iter or 0))
    return (result, iterations) if ret_iter else result
