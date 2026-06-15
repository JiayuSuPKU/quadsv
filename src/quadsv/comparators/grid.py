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

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import spatialdata as sd

from quadsv.comparators.base import (
    _ComparatorBase,
    _run_per_sample,
    _unpack_sample_quads,
    _validate_common,
)
from quadsv.comparators.features import (
    compute_sample_spectrum,
    estimate_rotations_from_landmarks,
    stream_geomean_landmark,
    stream_polar_features,
    stream_radial_features,
)

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
    feature_mode : {'radial', '2d'}, default 'radial'
    n_radial_bins : int, default 30
        Number of radial frequency intervals used when ``freq_edges`` is not
        supplied. Radial mode excludes only the DC cell from the first interval,
        so the automatic radial feature count is ``n_radial_bins``.
    n_theta_bins : int, default 36
        Number of angular bins on the half-plane ``[0, π)`` for
        ``feature_mode='2d'``. Ignored by radial mode. The 2D feature
        count is ``n_radius_bins * n_theta_bins``, where
        ``n_radius_bins = len(freq_edges) - 1`` (or ``n_radial_bins`` when
        edges are auto-generated).
    fft_chunk_size : int or 'auto', default 'auto'
        Genes per chunk. The rasterised image is always kept **lazy** (dask)
        and materialised one ``chunk``-gene block at a time, FFT'd, and
        immediately reduced (radial-binned in ``feature_mode='radial'``;
        rotated + physical-frequency polar-resampled in ``feature_mode='2d'``).
        Peak memory is ``O(chunk · ny · nx · 8 B)`` and the full ``(n_genes, ny,
        nx)`` raster / 2D spectra are *never* held in either mode.
        ``'auto'`` sizes the chunk from the (lazily-known) lattice shapes via
        :func:`quadsv.statistics.resolve_chunk_size` — the FFT cache sweet-spot
        cap (32) capped further by the live-memory budget.
        In 2d mode the rotation is learned from a streamed cross-gene geometric-mean
        landmark by default, or from an explicit ``landmark_genes`` set passed
        to :meth:`compute_spectra`.

    spacing : (dy, dx) or sequence of (dy, dx), optional
        Physical pitch of one rasterised bin. ``rasterize_bins`` emits one
        pixel per bin, so this maps the pixel lattice to physical frequency
        units (for example cycles/μm when spacing is in μm).
        Pass a single ``(dy, dx)`` to apply to all samples, or one ``(dy, dx)``
        per sample (length ``n_samples``) so that sections with different bin pitches
        still bin onto a common physical frequency grid.
        Defaults to ``(1.0, 1.0)`` (cycles per raster bin).

    Other Parameters
    ----------------
    fft_solver, workers, freq_edges, presence_threshold
        See :class:`_ComparatorBase`.

    Notes
    -----
    The comparator carries no design / contrast state — supply the
    cross-sample contrast directly to :meth:`test_diff_freq` /
    :meth:`test_diff_expr`. A single fitted comparator can therefore
    serve any number of unrelated comparisons on the same spectra.
    """

    # FFT cache sweet-spot cap for fft_chunk_size='auto' (statistics.auto_chunk_size).
    _auto_chunk_cap: int = 32

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
        n_theta_bins: int = 36,
        fft_solver: str = "rfft2",
        workers: int | None = None,
        spacing: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        freq_edges: np.ndarray | None = None,
        presence_threshold: float = 0.0,
        fft_chunk_size: int | str = "auto",
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
        self._n_theta_bins = self._normalize_n_theta_bins(n_theta_bins)
        self._fft_solver = fft_solver
        self._workers = workers
        self._presence_threshold = float(presence_threshold)
        # 'auto' is resolved lazily in compute_spectra once grid shapes are
        # known (rasterize_bins determines them per sample); an int is fixed now.
        self._fft_chunk_size_spec = self._normalize_chunk_spec(fft_chunk_size)
        self._fft_chunk_size = (
            32 if self._fft_chunk_size_spec == "auto" else (self._fft_chunk_size_spec)
        )
        self._spectrum_fft_solver = fft_solver

        self._bins = bins
        self._table_name = table_name
        self._col_key = col_key
        self._row_key = row_key
        self._value_key = value_key

        # Grid shape is determined per-sample inside compute_spectra by rasterize_bins
        # (the raster's .shape[-2:] carries it). Record the placeholder now
        # and fill in during _compute_spectra; spacing defaults to (1.0, 1.0)
        # because rasterize_bins outputs one pixel per bin — users can override
        # via the spacing kwarg when the bins encode a physical pitch. Pass a
        # single (dy, dx) to apply to all samples, or one (dy, dx) per sample
        # so heterogeneous bin pitches map to the same physical frequency units.
        self._spacing_override = self._broadcast_pairs(
            spacing, len(samples_list), name="spacing", cast=float
        )
        self._grid_shapes = []  # populated by _compute_spectra
        self._spacings = None  # populated alongside

    # ------------------------------------------------------------------
    def _rasterize_one(self, sdata: Any):
        """Lazy-rasterize one sample using ``spatialdata.rasterize_bins``.

        ``spatialdata.rasterize_bins`` returns a dask-backed ``xarray.DataArray``
        with a named gene coordinate ``c`` and one-gene-per-block chunking. We
        deliberately do **not** materialise the full ``(n_genes, ny, nx)`` spectra.
        Instead the lazy ``DataArray`` is returned so :meth:`_compute_spectra`
        can pull one **gene-name batch** at a time via
        ``img.sel(c=batch).values`` (label-based, so it also handles any
        ``gene_names`` reordering — same idiom as ``DetectorGrid``'s
        ``_qstat_worker_fft``). Returns the lazy ``(n_genes, ny, nx)``
        ``DataArray``.
        """
        from quadsv._rasterize import rasterize_table

        img = rasterize_table(
            sdata,
            bins=self._bins,
            table_name=self._table_name,
            col_key=self._col_key,
            row_key=self._row_key,
            value_key=self._value_key,
            return_region_as_labels=False,
        )
        if img.ndim != 3:
            raise ValueError(
                f"rasterize_bins returned shape {img.shape}, expected (n_genes, ny, nx)."
            )
        return img

    def _batch_raster(self, img: Any, names: Sequence[str]) -> np.ndarray:
        """Materialise the ``(len(names), ny, nx)`` raster block for ``names``.

        Label-based selection off the lazy ``DataArray`` (``img.sel(c=names)``),
        so only this gene batch is pulled into memory and the result follows
        :attr:`gene_names` order regardless of the raster's own gene order —
        the same idiom as ``DetectorGrid._qstat_worker_fft``.
        """
        return np.asarray(img.sel(c=list(names)).values, dtype=np.float64)

    def _sample_spectrum_chunker(self, i: int):
        """Build a ``(spectrum_chunk_fn, n_genes, (ny, nx))`` for one sample.

        ``spectrum_chunk_fn(start, stop)`` materialises just that gene-batch of
        the lazy raster (by name, in :attr:`gene_names` order), mean-centres +
        FFTs it, and returns the ``(stop-start, ny, n_kx)`` power spectrum.
        Callers pass it to the streaming reducers, so the dense raster is never
        held.
        """
        img = self._rasterize_one(self.samples[i])
        _, ny, nx = img.shape
        n_genes = len(self.gene_names)

        def _spec_chunk(start: int, stop: int) -> np.ndarray:
            block = self._batch_raster(img, self.gene_names[start:stop])
            return compute_sample_spectrum(
                block,
                fft_solver=self._spectrum_fft_solver,
                workers=self._workers,
                return_dc=False,
            )

        return _spec_chunk, n_genes, (ny, nx)

    def _compute_spectra(
        self,
        n_jobs: int,
        progress: bool,
        landmark_genes: Sequence[str] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Per-sample features via lazy, gene-chunk-streamed rasterisation.

        Radial mode streams rasterise → per-gene-chunk FFT → radial-bin. 2d mode
        does a two-pass streamed rotation alignment (geomean or explicit-landmark
        angle estimation, then rotate + physical-frequency polar resample per
        chunk). The full ``(n_genes, ny, nx)`` raster / 2D spectra are never held
        in either mode (see :meth:`_ComparatorBase._compute_spectra`).
        """
        # Resolve the spacing for each sample and define the shared frequency edges.
        n_samples = len(self.samples)
        spacings = (
            self._spacing_override
            if self._spacing_override is not None
            else [(1.0, 1.0)] * n_samples
        )
        self._spacings = list(spacings)
        self._resolve_freq_edges()

        # Resolve 'auto' chunk size. The lazy rasters carry their (ny, nx) as
        # cheap dask metadata, so we can scan shapes without materialising any
        # pixels, then size the chunk to bound the per-chunk dense footprint.
        if self._fft_chunk_size_spec == "auto":
            shapes = [tuple(self._rasterize_one(s).shape[-2:]) for s in self.samples]
            self._fft_chunk_size = self._resolve_chunk_size("auto", shapes, n_jobs=n_jobs)
            logger.info(
                "auto fft_chunk_size=%d (max lattice %d px, cap %d, budget %.1f GiB).",
                self._fft_chunk_size,
                max((ny * nx for (ny, nx) in shapes), default=1),
                self._auto_chunk_cap,
                self._auto_chunk_budget_bytes / 1024**3,
            )
        if self.feature_mode == "radial":
            return self._compute_spectra_radial(n_jobs, progress, spacings)
        return self._compute_spectra_2d(n_jobs, progress, spacings, landmark_genes)

    def _compute_spectra_radial(
        self, n_jobs: int, progress: bool, spacings: list[tuple[float, float]]
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        chunk = self._fft_chunk_size

        # Per-sample parallelism worker function
        def _one(
            i: int, pbar: Any = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
            img = self._rasterize_one(self.samples[i])
            _, ny, nx = img.shape
            n_genes = len(self.gene_names)
            dc = np.empty(n_genes, dtype=np.float64)
            presence = np.empty(n_genes, dtype=bool)

            def _spec_chunk(start: int, stop: int) -> np.ndarray:
                block = self._batch_raster(img, self.gene_names[start:stop])
                frac_nonzero = (block != 0).reshape(block.shape[0], -1).mean(axis=1)
                presence[start:stop] = frac_nonzero >= self._presence_threshold
                dc[start:stop] = block.mean(axis=(1, 2))
                return compute_sample_spectrum(
                    block,
                    fft_solver=self._spectrum_fft_solver,
                    workers=self._workers,
                    return_dc=False,
                )

            feat = stream_radial_features(
                _spec_chunk,
                n_genes,
                (ny, nx),
                chunk_size=chunk,
                n_bins=self._n_radial_bins,
                fft_solver=self._spectrum_fft_solver,
                spacing=spacings[i],
                edges=self.freq_edges,
                pbar=pbar,
            )
            return feat, dc, presence, (ny, nx)

        feats, dc, presence, grids = _unpack_sample_quads(
            _run_per_sample(
                _one,
                len(self.samples),
                n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk)),
                desc="Radial spectra (streaming)",
                n_jobs=n_jobs,
                progress=progress,
            )
        )
        self._grid_shapes = grids
        return feats, dc, presence

    def _compute_spectra_2d(  # noqa: C901 — two-pass streamed rotation alignment
        self,
        n_jobs: int,
        progress: bool,
        spacings: list[tuple[float, float]],
        landmark_genes: Sequence[str] | None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        chunk = self._fft_chunk_size
        name_to_idx = {g: j for j, g in enumerate(self.gene_names)}
        lm_idx: np.ndarray | None = None
        if landmark_genes is not None:
            missing = [g for g in landmark_genes if g not in name_to_idx]
            if missing:
                raise KeyError(f"landmark_genes not in gene_names: {missing}")
            lm_idx = np.asarray([name_to_idx[g] for g in landmark_genes], dtype=int)

        # ---- Pass A: per-sample rotation landmark + DC/presence ----
        # The per-sample grid shape is only known after the first rasterise, so
        # the explicit-landmark cache-budget decision (cache per-gene spectra vs.
        # warn + collapse to a streamed geomean) is made on the first iteration.
        if lm_idx is not None:
            probe_grids = [tuple(self._rasterize_one(s).shape[-2:]) for s in self.samples]
            self._grid_shapes = probe_grids
            cache_landmarks = self._landmark_cache_fits(len(lm_idx))
        else:
            cache_landmarks = False

        # Per-sample parallelism worker for landmark spectra
        def _landmark_one(
            i: int, pbar: Any = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
            img = self._rasterize_one(self.samples[i])
            _, ny, nx = img.shape
            n_genes = len(self.gene_names)
            dc = np.empty(n_genes, dtype=np.float64)
            presence = np.empty(n_genes, dtype=bool)

            def _spec_chunk(start: int, stop: int, _img=img, _dc=dc, _pr=presence):
                block = self._batch_raster(_img, self.gene_names[start:stop])
                frac_nonzero = (block != 0).reshape(block.shape[0], -1).mean(axis=1)
                _pr[start:stop] = frac_nonzero >= self._presence_threshold
                _dc[start:stop] = block.mean(axis=(1, 2))
                return compute_sample_spectrum(
                    block, fft_solver=self._spectrum_fft_solver, workers=self._workers
                )

            if lm_idx is None:
                # Default: stream the cross-gene geomean landmark; this single
                # pass over all genes also fills dc/presence (side effect).
                lm = stream_geomean_landmark(
                    _spec_chunk, n_genes, (ny, nx), chunk_size=chunk, pbar=pbar
                )
            else:
                # Explicit landmarks: dc/presence need a full pass over ALL genes.
                for start in range(0, n_genes, chunk):
                    _spec_chunk(start, min(start + chunk, n_genes))
                    if pbar is not None:
                        pbar.update(1)
                if cache_landmarks:
                    # Within budget: cache the landmark genes' per-gene spectra.
                    lm_names = [self.gene_names[j] for j in lm_idx]
                    lm = compute_sample_spectrum(
                        self._batch_raster(img, lm_names),
                        fft_solver=self._spectrum_fft_solver,
                        workers=self._workers,
                    )
                    if pbar is not None:
                        pbar.update(1)
                else:
                    # Over budget: stream the geometric mean of the landmark genes.
                    lm = stream_geomean_landmark(
                        _spec_chunk,
                        n_genes,
                        (ny, nx),
                        chunk_size=chunk,
                        gene_subset=lm_idx,
                        pbar=pbar,
                    )
            return lm, dc, presence, (ny, nx)

        if lm_idx is None:
            n_pass_a_chunks = int(np.ceil(len(self.gene_names) / chunk))
        elif cache_landmarks:
            n_pass_a_chunks = int(np.ceil(len(self.gene_names) / chunk)) + 1
        else:
            n_pass_a_chunks = int(np.ceil(len(self.gene_names) / chunk)) + int(
                np.ceil(len(lm_idx) / chunk)
            )
        landmarks, dc, presence, grids = _unpack_sample_quads(
            _run_per_sample(
                _landmark_one,
                len(self.samples),
                n_chunks_per_sample=max(1, n_pass_a_chunks),
                desc="2D landmarks (streaming)",
                n_jobs=n_jobs,
                progress=progress,
            )
        )

        angles = estimate_rotations_from_landmarks(
            landmarks, grids, fft_solver=self._spectrum_fft_solver, progress=progress
        )
        self.rotation_angles_ = angles

        # Record the rasterize_bins lattice shape learned during the landmark pass.
        self._grid_shapes = grids

        # ---- Pass B: re-stream, rotate, physical-frequency polar resample ----
        # Per-sample parallelism worker for feature spectra (binned)
        def _feature_one(i: int, pbar: Any = None) -> np.ndarray:
            spec_chunk_fn, n_genes, (ny, nx) = self._sample_spectrum_chunker(i)
            return stream_polar_features(
                spec_chunk_fn,
                n_genes,
                (ny, nx),
                float(angles[i]),
                chunk_size=chunk,
                freq_edges=self.freq_edges,
                spacing=spacings[i],
                n_theta=self._n_theta_bins,
                fft_solver=self._spectrum_fft_solver,
                pbar=pbar,
            )

        feats = _run_per_sample(
            _feature_one,
            len(self.samples),
            n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk)),
            desc="2D features (streaming)",
            n_jobs=n_jobs,
            progress=progress,
        )
        return [np.asarray(f) for f in feats], dc, presence

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
