"""
:class:`ComparatorGrid` — cross-sample pattern comparison on a list of
:class:`spatialdata.SpatialData` (regular rasterized bins, FFT
backend).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import spatialdata as sd

from quadsv.comparators.base import (
    _ComparatorBase,
    _validate_common,
)
from quadsv.comparators.multisample import compute_sample_spectrum

__all__ = ["ComparatorGrid"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComparatorGrid — SpatialData / regular bins via rasterize_bins
# ---------------------------------------------------------------------------


class ComparatorGrid(_ComparatorBase):
    """
    Cross-sample pattern comparison on regular bins via FFT + SpatialData.

    Accepts a list of :class:`spatialdata.SpatialData` (one per sample). For
    each sample, :func:`spatialdata.rasterize_bins` turns the designated bin
    shape + table into a dense ``(n_genes, ny, nx)`` image, which is then fed
    to the batched 2D FFT. All samples are expected to share the same
    rasterization schema (``bins`` / ``table_name`` / ``col_key`` / ``row_key``
    / ``value_key``) — this mirrors :class:`~quadsv.DetectorGrid`.

    Parameters
    ----------
    samples : sequence of :class:`spatialdata.SpatialData`
    bins : str
        SpatialElement key for the bin shapes in each ``sdata``.
    table_name : str
        Table key in each ``sdata.tables``.
    col_key, row_key : str
        Column / row-index columns in the table's ``.obs``.
    value_key : str, optional
        Expression column in ``.obs``; defaults to ``None`` (rasterizes counts
        / presence directly off ``.X``).
    gene_names : sequence of str, optional
        If None, inferred from the first sample's table. All samples must
        share ``var_names``.
    fft_chunk_size : int, default 256
        Genes per batched ``scipy.fft`` call on the rasterized block. Keeps
        transient memory bounded at ``O(ny · nx · chunk · 8 B)``. The raster
        itself is still built once per sample (full ``(n_genes, ny, nx)``
        footprint is unavoidable on SpatialData).

    Other Parameters
    ----------------
    feature_mode, n_radial_bins, fft_solver, workers, freq_edges, presence_threshold
        See :class:`_ComparatorBase`.

    Notes
    -----
    The comparator carries no design / contrast state — supply the
    cross-sample contrast directly to :meth:`test_diff_freq` /
    :meth:`test_diff_expr`. A single fitted comparator can therefore
    serve any number of unrelated comparisons on the same spectra.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        *,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        gene_names: Sequence[str] | None = None,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        fft_solver: str = "rfft2",
        workers: int | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        presence_threshold: float = 0.0,
        fft_chunk_size: int = 256,
    ) -> None:
        fft_solver = _validate_common(feature_mode, fft_solver, presence_threshold)
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, sd.SpatialData):
                raise TypeError(
                    f"sample {i} is {type(s).__name__}, expected spatialdata.SpatialData."
                )

        resolved = _resolve_spatialdata_gene_names(samples_list, gene_names, table_name)

        self.samples = samples_list
        self.gene_names = list(resolved)
        self.feature_mode = feature_mode
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        # Private (internal-config) state.
        self._n_radial_bins = int(n_radial_bins)
        self._fft_solver = fft_solver
        self._workers = workers
        self._presence_threshold = float(presence_threshold)
        self._fft_chunk_size = max(1, int(fft_chunk_size))
        self._spectrum_fft_solver = fft_solver

        self._bins = bins
        self._table_name = table_name
        self._col_key = col_key
        self._row_key = row_key
        self._value_key = value_key

        # Grid shape is determined per-sample inside compute_spectra by rasterize_bins
        # (the raster's .shape[-2:] carries it). Record the placeholder now
        # and fill in during _compute_spectra; spacing is always (1.0, 1.0)
        # because rasterize_bins outputs one pixel per bin — users can
        # override via the spacing kwarg when the bins encode a physical
        # pitch.
        self._spacing_override = None if spacing is None else (float(spacing[0]), float(spacing[1]))
        self._grid_shapes = []  # populated by _compute_spectra
        self._spacings = None  # populated alongside

    # ------------------------------------------------------------------
    def _rasterize_one(self, sdata: Any) -> np.ndarray:
        """Wrap :func:`spatialdata.rasterize_bins`. Returns a
        ``(n_genes, ny, nx)`` float array in :attr:`gene_names` order.
        """
        from quadsv._rasterize import rasterize_table

        table = sdata.tables[self._table_name]
        img = rasterize_table(
            sdata,
            bins=self._bins,
            table_name=self._table_name,
            col_key=self._col_key,
            row_key=self._row_key,
            value_key=self._value_key,
            return_region_as_labels=False,
        )
        arr = np.asarray(img.data if hasattr(img, "data") else img, dtype=np.float64)
        # Expected shape: (n_genes_in_table, ny, nx). Reindex gene axis to
        # self.gene_names (which was validated to match table var_names).
        if arr.ndim != 3:
            raise ValueError(
                f"rasterize_bins returned shape {arr.shape}, expected (n_genes, ny, nx)."
            )
        table_names = list(table.var_names)
        if table_names == list(self.gene_names):
            return arr
        idx = np.asarray([table_names.index(g) for g in self.gene_names], dtype=int)
        return arr[idx]

    def _compute_spectra(
        self, n_jobs: int, progress: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        chunk = self._fft_chunk_size

        def _one(i: int, pbar: tqdm | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raster = self._rasterize_one(self.samples[i])
            n_genes, ny, nx = raster.shape
            # Lock grid shape + spacing from the first-seen rasterization.
            self._grid_shapes_local_i = (ny, nx)

            frac_nonzero = (raster != 0).reshape(n_genes, -1).mean(axis=1)
            presence_i = frac_nonzero >= self._presence_threshold
            dc = raster.mean(axis=(1, 2))

            expected_kx = nx if self._spectrum_fft_solver == "fft2" else nx // 2 + 1
            spec_stack = np.empty((n_genes, ny, expected_kx), dtype=np.float64)
            for start in range(0, n_genes, chunk):
                stop = min(start + chunk, n_genes)
                # Reuse the shared helper; it always mean-centres each gene
                # before the FFT.
                spec_chunk = compute_sample_spectrum(
                    raster[start:stop],
                    fft_solver=self._spectrum_fft_solver,
                    workers=self._workers,
                    return_dc=False,
                )
                spec_stack[start:stop] = spec_chunk
                if pbar is not None:
                    pbar.update(1)

            return spec_stack, dc, presence_i

        # Run the per-sample loop and collect the grid shapes (only known
        # after rasterize_bins returns).
        n_samples_total = len(self.samples)
        raw_2d: list[np.ndarray | None] = [None] * n_samples_total
        dc_list: list[np.ndarray | None] = [None] * n_samples_total
        pres_list: list[np.ndarray | None] = [None] * n_samples_total
        grids: list[tuple[int, int]] = []

        run_sequential = progress or n_jobs == 1
        if run_sequential:
            n_chunks_total = sum(
                int(np.ceil(len(self.gene_names) / chunk)) for _ in range(n_samples_total)
            )
            pbar: tqdm | None = (
                tqdm(total=n_chunks_total, desc="FFT spectra (per-gene chunks)")
                if progress
                else None
            )
            for i in range(n_samples_total):
                if pbar is not None:
                    pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
                r0, r1, r2 = _one(i, pbar=pbar)
                raw_2d[i] = r0
                dc_list[i] = r1
                pres_list[i] = r2
                grids.append(self._grid_shapes_local_i)
            if pbar is not None:
                pbar.close()
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(i) for i in range(n_samples_total)
            )
            for i, r in enumerate(results):
                raw_2d[i], dc_list[i], pres_list[i] = r
                # When running via joblib the `self._grid_shapes_local_i`
                # side-channel isn't reliable — infer from the returned spec.
                grids.append(raw_2d[i].shape[-2:])

        # Record per-sample grids + spacings for downstream radial binning.
        self._grid_shapes = grids
        if self._spacing_override is not None:
            self._spacings = [self._spacing_override] * len(grids)
        else:
            self._spacings = [(1.0, 1.0)] * len(grids)
        del self._grid_shapes_local_i  # tidy attr soup

        dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
        presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
        return [np.asarray(x) for x in raw_2d], dc, presence

    # ------------------------------------------------------------------
    def _covariate_features_from_keys(self, keys: Sequence[str]) -> list[np.ndarray]:
        """Forward ``keys`` to :func:`spatialdata.rasterize_bins` as
        ``value_key`` and turn each per-sample raster into covariate features.

        ``keys`` may name any combination of ``.obs`` columns and
        ``var_names`` in the comparator's ``table_name`` table —
        ``rasterize_bins`` accepts both. Each sample's resulting
        ``(n_keys, ny, nx)`` raster is then funneled through the same
        spectrum + radial-binning pipeline as the gene panel via
        :meth:`_covariate_features_from_array`.
        """
        from quadsv._rasterize import rasterize_table

        keys = list(keys)
        out: list[np.ndarray] = []
        for i, sdata in enumerate(self.samples):
            img = rasterize_table(
                sdata,
                bins=self._bins,
                table_name=self._table_name,
                col_key=self._col_key,
                row_key=self._row_key,
                value_key=keys,
                return_region_as_labels=False,
            )
            arr = np.asarray(img.data if hasattr(img, "data") else img, dtype=np.float64)
            if arr.ndim == 2:
                # Single key → rasterize_bins drops the leading axis. Re-insert.
                arr = arr[None, :, :]
            if arr.ndim != 3:
                raise ValueError(
                    f"sample {i} covariate raster has shape {arr.shape}; "
                    "expected (n_keys, ny, nx) from rasterize_bins."
                )
            out.append(self._covariate_features_from_array(arr, sample_index=i))
        return out


def _resolve_spatialdata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    table_name: str,
) -> list[str]:
    first = samples[0]
    if table_name not in first.tables:
        raise KeyError(
            f"sample 0 has no table '{table_name}'; available: {list(first.tables.keys())}."
        )
    if gene_names is None:
        gene_names = list(first.tables[table_name].var_names)
    for i, s in enumerate(samples):
        if table_name not in s.tables:
            raise KeyError(f"sample {i} has no table '{table_name}'.")
        tbl_names = list(s.tables[table_name].var_names)
        if tbl_names != list(gene_names):
            raise ValueError(f"sample {i}'s table has var_names that do not match the reference.")
    return list(gene_names)
