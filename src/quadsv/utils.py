"""General-purpose helpers: grid/coordinate generators, distance matrices, and Visium I/O."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    # Grid & coordinate helpers
    "get_rect_coords",
    "get_visium_coords",
    "convert_visium_to_physical",
    "compute_torus_distance_matrix",
    # Visium I/O
    "VISIUM_V1_SPOT_SPACING_UM",
    "visium_hex_spacing_um",
    "load_visium_sample",
    "visium_to_grid",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid & coordinate generators
# ---------------------------------------------------------------------------


def get_rect_coords(n_rows: int = 32, n_cols: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Generate rectangular grid coordinates with unit spacing.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.

    Returns
    -------
    coords : np.ndarray
        ``(n, 2)`` row-major ``[[y, x], ...]`` with ``n = n_rows · n_cols``.
    grid_dims : tuple of int
        ``(n_rows, n_cols)``.

    Examples
    --------
    >>> coords, dims = get_rect_coords(n_rows=10, n_cols=10)
    >>> coords.shape
    (100, 2)
    >>> dims
    (10, 10)
    """
    y = np.arange(n_rows)
    x = np.arange(n_cols)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()]), (n_rows, n_cols)


def get_visium_coords(n_rows: int = 78, n_cols: int = 64) -> tuple[np.ndarray, tuple[int, int]]:
    """Generate Visium-like hexagonal array indices.

    Offset-column layout: even rows use even ``array_col`` indices, odd rows
    use odd indices. Each row holds ``n_cols`` spots.

    Parameters
    ----------
    n_rows : int, default 78
        Number of array rows (78 for a full Visium v1 slide).
    n_cols : int, default 64
        Spots per row.

    Returns
    -------
    coords : np.ndarray
        ``(n_rows · n_cols, 2)`` integer ``[[row, col], ...]``.
    grid_dims : tuple of int
        ``(n_rows, n_cols)``.

    See Also
    --------
    convert_visium_to_physical : Map these indices to physical ``(y, x)``.

    Examples
    --------
    >>> coords, dims = get_visium_coords(n_rows=78, n_cols=64)
    >>> coords.shape[0]
    4992
    """
    coords = []
    for r in range(n_rows):
        start_col = 0 if r % 2 == 0 else 1
        for i in range(n_cols):
            coords.append([r, start_col + 2 * i])
    return np.array(coords), (n_rows, n_cols)


def convert_visium_to_physical(coords: np.ndarray) -> np.ndarray:
    """Convert integer Visium ``(row, col)`` indices to physical ``(y, x)``.

    Hexagonal geometry with equilateral triangles: ``Δcol = 2 → Δx = 1`` and
    ``Δrow = 1 → Δy = √3/2``.

    Parameters
    ----------
    coords : np.ndarray
        ``(n, 2)`` ``[[row, col], ...]`` (integers, from
        :func:`get_visium_coords` or ``adata.obs[['array_row','array_col']]``).

    Returns
    -------
    np.ndarray
        ``(n, 2)`` ``[[y, x], ...]`` with ``y = row · √3/2``, ``x = col / 2``.

    Examples
    --------
    >>> coords = np.array([[0, 0], [0, 2], [1, 1]])
    >>> convert_visium_to_physical(coords)
    array([[0.       , 0.       ],
           [0.       , 1.       ],
           [0.8660254, 0.5      ]])
    """
    rows = coords[:, 0]
    cols = coords[:, 1]
    phys_y = rows * np.sqrt(3) / 2.0
    phys_x = cols * 0.5
    return np.column_stack([phys_y, phys_x])


def compute_torus_distance_matrix(
    phys_coords: np.ndarray, domain_dims: tuple[float, float]
) -> np.ndarray:
    """Pairwise Euclidean distances on a rectangular torus.

    Each pair's distance is the minimum over direct and wrap-around offsets:
    ``d_wrapped = min(|Δp|, D - |Δp|)`` per axis.

    Parameters
    ----------
    phys_coords : np.ndarray
        ``(n, 2)`` ``[[y, x], ...]``.
    domain_dims : tuple of float
        Periodic domain extent ``(height, width)``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` pairwise distances.

    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [9.9, 0.0]])
    >>> dists = compute_torus_distance_matrix(coords, (10.0, 10.0))
    >>> round(dists[0, 2], 2)
    0.1
    """
    domain_h, domain_w = domain_dims
    diff_y = np.abs(phys_coords[:, None, 0] - phys_coords[None, :, 0])
    diff_x = np.abs(phys_coords[:, None, 1] - phys_coords[None, :, 1])
    wrapped_dy = np.minimum(diff_y, domain_h - diff_y)
    wrapped_dx = np.minimum(diff_x, domain_w - diff_x)
    return np.sqrt(wrapped_dy**2 + wrapped_dx**2)


# ---------------------------------------------------------------------------
# Visium I/O and hex → regular-grid rasterization
# ---------------------------------------------------------------------------

#: Physical center-to-center distance between Visium v1 (6.5 mm capture) spots, in μm.
VISIUM_V1_SPOT_SPACING_UM: float = 100.0


def visium_hex_spacing_um(
    spot_spacing_um: float = VISIUM_V1_SPOT_SPACING_UM,
    grid: str = "dense",
) -> tuple[float, float]:
    """Physical ``(dy, dx)`` per grid cell for a Visium hex raster.

    Parameters
    ----------
    spot_spacing_um : float, default 100.0
        Center-to-center distance between adjacent spots (100 μm for Visium v1).
    grid : {'dense', 'collapsed'}, default 'dense'
        Rasterization mode — see :func:`visium_to_grid`.

    Returns
    -------
    tuple of float
        ``(dy, dx)`` in micrometres per grid cell.

    Raises
    ------
    ValueError
        If ``grid`` is unknown.
    """
    dy = spot_spacing_um * np.sqrt(3.0) / 2.0
    if grid == "dense":
        dx = spot_spacing_um / 2.0
    elif grid == "collapsed":
        dx = spot_spacing_um
    else:
        raise ValueError(f"grid must be 'dense' or 'collapsed', got '{grid}'.")
    return float(dy), float(dx)


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    """Read ``tissue_positions_list.csv`` (SR v1) or ``tissue_positions.csv`` (v2)."""
    candidates = [
        spatial_dir / "tissue_positions_list.csv",  # Space Ranger < 2.0, no header
        spatial_dir / "tissue_positions.csv",  # Space Ranger >= 2.0, with header
    ]
    for path in candidates:
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No tissue_positions[_list].csv found in {spatial_dir}.")

    if path.name == "tissue_positions_list.csv":
        return pd.read_csv(
            path,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    return pd.read_csv(path)


def load_visium_sample(  # noqa: C901
    path: str | Path,
    matrix_path: str | Path | None = None,
    in_tissue_only: bool = True,
) -> "anndata.AnnData":  # noqa: F821, UP037
    """Load a Visium Space Ranger output directory as :class:`anndata.AnnData`.

    Accepts either the flat layout (``<path>/<sample>_filtered_feature_bc_matrix.h5``
    + ``<path>/spatial/``) or the canonical Space Ranger ``outs/`` layout
    (``<path>/filtered_feature_bc_matrix.h5`` + ``<path>/spatial/``); auto-detects.

    Parameters
    ----------
    path : str or Path
        Directory holding the filtered matrix and ``spatial/`` subfolder.
    matrix_path : str or Path, optional
        Explicit path to the filtered ``.h5`` matrix. Defaults to
        ``<path>/filtered_feature_bc_matrix.h5`` or ``*_filtered_feature_bc_matrix.h5``.
    in_tissue_only : bool, default True
        Restrict to spots with ``in_tissue == 1``.

    Returns
    -------
    anndata.AnnData
        ``adata.obs`` has ``in_tissue``, ``array_row``, ``array_col``,
        ``pxl_row_in_fullres``, ``pxl_col_in_fullres``.
        ``adata.obsm['spatial']`` holds ``(pxl_col, pxl_row)`` full-resolution
        pixel coords. ``adata.uns['spatial']`` stores ``scalefactors_json.json``
        and the source path.

    Raises
    ------
    FileNotFoundError
        If the matrix or spatial folder cannot be located.
    ImportError
        If :mod:`anndata` / :mod:`scanpy` are not installed.
    """
    try:
        import anndata  # noqa: F401
        import scanpy as sc
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_visium_sample requires scanpy + anndata. "
            "Install with `pip install 'quadsv[spatial]'` or `pip install scanpy`."
        ) from e

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory.")

    spatial_dir = path / "spatial"
    if not spatial_dir.is_dir():
        alt = path / "outs" / "spatial"
        if alt.is_dir():
            spatial_dir = alt
            path = path / "outs"
        else:
            raise FileNotFoundError(f"No spatial/ subfolder found under {path}.")

    if matrix_path is None:
        mp = path / "filtered_feature_bc_matrix.h5"
        if not mp.exists():
            matches = sorted(path.glob("*_filtered_feature_bc_matrix.h5"))
            if not matches:
                raise FileNotFoundError(f"No filtered_feature_bc_matrix.h5 found under {path}.")
            mp = matches[0]
    else:
        mp = Path(matrix_path)

    logger.info("Reading Visium matrix: %s", mp)
    adata = sc.read_10x_h5(mp)
    adata.var_names_make_unique()

    tp = _read_tissue_positions(spatial_dir).set_index("barcode")

    missing = set(adata.obs_names) - set(tp.index)
    if missing:
        warnings.warn(
            f"{len(missing)} barcodes in matrix lack entries in tissue_positions; "
            "they will be dropped.",
            UserWarning,
            stacklevel=2,
        )
        adata = adata[adata.obs_names.isin(tp.index)].copy()
    tp = tp.reindex(adata.obs_names)
    for col in ("in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"):
        adata.obs[col] = tp[col].astype(int if col == "in_tissue" else float).values

    adata.obsm["spatial"] = np.column_stack(
        [tp["pxl_col_in_fullres"].to_numpy(), tp["pxl_row_in_fullres"].to_numpy()]
    )

    scalefactors_path = spatial_dir / "scalefactors_json.json"
    if scalefactors_path.exists():
        with scalefactors_path.open() as f:
            scalefactors = json.load(f)
    else:
        scalefactors = {}
    adata.uns["spatial"] = {"scalefactors": scalefactors, "path": str(path)}

    if in_tissue_only:
        adata = adata[adata.obs["in_tissue"].to_numpy().astype(bool)].copy()

    logger.info(
        "Loaded Visium sample: %d spots (%d in tissue), %d genes.",
        adata.n_obs,
        int(adata.obs["in_tissue"].sum()) if "in_tissue" in adata.obs else adata.n_obs,
        adata.n_vars,
    )
    return adata


def _fill_nearest_hex_neighbor(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill empty cells of a Visium hex raster from same-row neighbours.

    For the alternating-parity layout, empty cells at ``(r, c)`` with
    ``c % 2 != r % 2`` have two real spot neighbours on the same row at
    ``(r, c±1)``. Average them (fall back to one side at edges).
    """
    out = grid.copy()
    ny, nx = mask.shape
    empty = np.argwhere(~mask)
    if empty.size == 0:
        return out
    for r, c in empty:
        left = out[..., r, c - 1] if c - 1 >= 0 and mask[r, c - 1] else None
        right = out[..., r, c + 1] if c + 1 < nx and mask[r, c + 1] else None
        if left is not None and right is not None:
            out[..., r, c] = 0.5 * (left + right)
        elif left is not None:
            out[..., r, c] = left
        elif right is not None:
            out[..., r, c] = right
    return out


def visium_to_grid(  # noqa: C901
    adata: "anndata.AnnData",  # noqa: F821, UP037
    genes: list[str] | None = None,
    layer: str | None = None,
    grid: str = "dense",
    fill: str = "nearest",
    spot_spacing_um: float = VISIUM_V1_SPOT_SPACING_UM,
    max_row: int | None = None,
    max_col: int | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize a Visium ``adata`` onto a regular rectangular grid.

    Rasterization modes:

    - ``grid='dense'`` (default): ``(78, 128)`` array filled from real spots on
      half the cells; remaining cells imputed from their two nearest hex
      neighbours on the same row. Physical spacing ``(100·√3/2, 50)`` μm.
      Exact hex geometry preserved.
    - ``grid='collapsed'``: ``(78, 64)`` array using ``array_col // 2`` as
      column index. Physical spacing ``(100·√3/2, 100)`` μm. Faster, but
      drops the 50 μm horizontal offset between alternating rows
      (≤5 % geometric distortion).

    Parameters
    ----------
    adata : anndata.AnnData
        Must carry ``adata.obs['array_row']`` and ``adata.obs['array_col']``
        (e.g. from :func:`load_visium_sample`).
    genes : list of str, optional
        Gene subset. ``None`` → all ``adata.var_names``.
    layer : str, optional
        ``adata.layers`` key. ``None`` → ``adata.X``.
    grid : {'dense', 'collapsed'}, default 'dense'
    fill : {'nearest', 'zero'}, default 'nearest'
        Empty-cell handling. ``'nearest'`` averages the two row-neighbour spots
        (avoids FFT aliasing); ``'zero'`` leaves cells at 0. Ignored when
        ``grid='collapsed'``.
    spot_spacing_um : float, default 100.0
    max_row, max_col : int, optional
        Pad output to a common size; defaults to the maxima observed in ``adata``.

    Returns
    -------
    grid_arr : np.ndarray
        ``(n_genes, ny, nx)`` float64, ready for
        :func:`quadsv.power_spectrum_2d`.
    spacing_um : tuple of float
        ``(dy, dx)`` in μm.

    Raises
    ------
    KeyError
        If ``array_row`` / ``array_col`` are absent from ``adata.obs``.
    ValueError
        If ``grid`` / ``fill`` are unknown, or ``layer`` is missing.
    """
    if "array_row" not in adata.obs or "array_col" not in adata.obs:
        raise KeyError(
            "adata.obs must contain 'array_row' and 'array_col'. "
            "Load the sample with quadsv.utils.load_visium_sample first."
        )
    if grid not in ("dense", "collapsed"):
        raise ValueError(f"grid must be 'dense' or 'collapsed', got '{grid}'.")
    if fill not in ("nearest", "zero"):
        raise ValueError(f"fill must be 'nearest' or 'zero', got '{fill}'.")

    rows = adata.obs["array_row"].to_numpy().astype(int)
    cols = adata.obs["array_col"].to_numpy().astype(int)

    ny = int(rows.max() + 1) if max_row is None else max_row
    if grid == "dense":
        nx = int(cols.max() + 1) if max_col is None else max_col
    else:  # collapsed: use array_col // 2 as column index
        nx = int(cols.max() // 2 + 1) if max_col is None else max_col

    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    if genes is not None:
        gene_idx = [adata.var_names.get_loc(g) for g in genes]
        X = X[:, gene_idx]

    n_genes = X.shape[1]
    out = np.zeros((n_genes, ny, nx), dtype=np.float64)
    filled = np.zeros((ny, nx), dtype=bool)

    if grid == "dense":
        for i in range(X.shape[0]):
            r, c = rows[i], cols[i]
            if 0 <= r < ny and 0 <= c < nx:
                out[:, r, c] = X[i]
                filled[r, c] = True
        if fill == "nearest":
            out = _fill_nearest_hex_neighbor(out, filled)
    else:  # collapsed
        for i in range(X.shape[0]):
            r, c = rows[i], cols[i] // 2
            if 0 <= r < ny and 0 <= c < nx:
                out[:, r, c] = X[i]

    spacing = visium_hex_spacing_um(spot_spacing_um=spot_spacing_um, grid=grid)
    logger.info(
        "Rasterized %d genes x %d spots onto %s grid of shape (%d, %d), spacing=%s μm.",
        n_genes,
        adata.n_obs,
        grid,
        ny,
        nx,
        spacing,
    )
    return out, spacing
