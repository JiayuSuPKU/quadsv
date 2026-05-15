"""
Shared :func:`spatialdata.rasterize_bins` wrappers used by
:class:`quadsv.DetectorGrid` and :class:`quadsv.ComparatorGrid`.

Both consumers need the same boilerplate:

1. Coerce the table's X matrix to CSC sparse (required by ``rasterize_bins``).
2. Forward to ``spatialdata.rasterize_bins`` with the user-supplied keys.

Callers differ only in what they do with the rasterized image afterwards —
``DetectorGrid`` stashes it back into ``sdata.images`` under a derived key,
``ComparatorGrid`` extracts the array and reindexes the gene axis — so the
shared helper stops at the ``rasterize_bins`` return value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import spatialdata as sd

__all__ = ["ensure_csc_table", "rasterize_table"]


def ensure_csc_table(sdata: Any, table_name: str) -> None:
    """Coerce ``sdata.tables[table_name].X`` to CSC sparse in-place, if sparse.

    ``spatialdata.rasterize_bins`` performs column-wise slicing and requires CSC.
    Dense arrays are left untouched.
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} not found in sdata.")
    table = sdata.tables[table_name]
    X = getattr(table, "X", None)
    if X is None or isinstance(X, np.ndarray):
        return
    if sp.issparse(X) and X.format != "csc":
        table.X = X.tocsc()


def rasterize_table(
    sdata: Any,
    *,
    bins: str,
    table_name: str,
    col_key: str,
    row_key: str,
    value_key: str | None = None,
    return_region_as_labels: bool = False,
):
    """Call :func:`spatialdata.rasterize_bins` after coercing the table to CSC.

    Returns the raw ``rasterize_bins`` output — a DataArray-like object whose
    ``.data`` attribute holds an ``(n_genes, ny, nx)`` array when
    ``value_key`` is ``None``.
    """
    ensure_csc_table(sdata, table_name)
    return sd.rasterize_bins(
        sdata,
        bins=bins,
        table_name=table_name,
        col_key=col_key,
        row_key=row_key,
        value_key=value_key,
        return_region_as_labels=return_region_as_labels,
    )
