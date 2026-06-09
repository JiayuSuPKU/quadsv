"""
:class:`ComparatorIrregular` — cross-sample pattern comparison on a
list of :class:`anndata.AnnData` (irregular spots, NUFFT backend).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anndata as _ad
import numpy as np
import scipy.sparse as sp

from quadsv.comparators.base import (
    _ComparatorBase,
    _run_per_sample,
    _unpack_sample_triples,
    _validate_common,
)
from quadsv.comparators.multisample import (
    estimate_rotations_from_landmarks,
    radial_bin_spectrum,
    stream_geomean_landmark,
    stream_polar_features,
    stream_radial_features,
)

__all__ = ["ComparatorIrregular"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComparatorIrregular — AnnData / irregular spots
# ---------------------------------------------------------------------------


class ComparatorIrregular(_ComparatorBase):
    """
    Cross-sample pattern comparison on irregular spots via NUFFT.

    Accepts a list of :class:`anndata.AnnData` (one per sample). For each
    sample, the per-sample ``obsm[obsm_key]`` supplies the irregular
    ``(y, x)`` coordinates and ``.X`` (or ``.layers[layer]`` when set) is the
    expression matrix. Spectra are evaluated with a batched type-1 NUFFT
    (``finufft.nufft2d1``), densifying at most :attr:`nufft_chunk_size`
    columns of ``.X`` at a time so the full slab is never materialized.

    Parameters
    ----------
    samples : sequence of :class:`anndata.AnnData`
    gene_names : sequence of str, optional
        If None, inferred from the first sample; every other sample must share
        the same ``var_names``.
    feature_mode : {'radial', '2d'}, default 'radial'
    n_radial_bins : int, default 30
        Number of radial frequency edges minus one. With the default DC
        exclusion, radial mode returns ``n_radial_bins - 1`` columns.
    n_theta_bins : int, default 36
        Number of angular bins on the half-plane ``[0, π)`` for
        ``feature_mode='2d'``. Ignored by radial mode. 2D features have
        ``n_radial_bins * n_theta_bins`` columns unless explicit
        ``freq_edges`` are supplied, in which case the radius count is
        ``len(freq_edges) - 1``.
    obsm_key : str, default 'spatial'. Key for the spatial coordinates in ``obsm``.
    layer : str, optional
    unit_scales : sequence of float, optional
        Per-sample multiplier applied to coords before NUFFT (e.g. pixels→μm).
    grid_shape, spacing : optional
        Internal NUFFT k-grid definition. Each argument may be a single pair
        applied to every sample — ``grid_shape=(ny, nx)`` and
        ``spacing=(dy, dx)`` — or a per-sample sequence of pairs of length
        ``n_samples``. ``spacing`` is in the same physical units as
        ``coords * unit_scale`` and sets the frequency unit used downstream
        (for example cycles/μm when coordinates are scaled to μm).

        Manual grid control is used only when **both** ``grid_shape`` and
        ``spacing`` are supplied. If either is omitted, both values are
        auto-inferred independently for each sample from its unit-scaled
        coordinates with :func:`quadsv.kernels.nufft._infer_grid_from_coords`.

        These values define each sample's raw NUFFT lattice; they do not by
        themselves choose the comparison bins. Unless ``freq_edges`` is given,
        :meth:`compute_spectra` builds one shared physical-frequency bin grid
        from all resolved spacings: edges span ``[0, min Nyquist]``, where each
        sample's Nyquist is ``1 / (2 * max(dy, dx))``. Radial features bin each
        sample's spectrum onto those shared edges; ``feature_mode='2d'`` uses
        the same edges as the polar radius grid after rotation alignment.
    freq_edges : np.ndarray, optional
        Explicit shared radial-frequency bin edges. When supplied, these edges
        are used as-is for all samples and override the automatic
        ``[0, min Nyquist]`` construction. Edges must be in the same units as
        ``spacing``. With the default DC exclusion, radial output has
        ``len(freq_edges) - 2`` columns; otherwise the automatic
        ``n_radial_bins + 1`` edges produce ``n_radial_bins - 1`` radial
        columns. In ``feature_mode='2d'``, the number of radius bins is
        ``len(freq_edges) - 1`` and the feature axis has
        ``(len(freq_edges) - 1) * n_theta_bins`` columns.
    eps : float, default 1e-6
        NUFFT tolerance.
    presence_threshold : float, default 0.0
        Minimum fraction of non-zero spots for a gene to count as "observed"
        in a sample (feeds :attr:`presence_` and, transitively, the masked
        pattern test).
    nufft_chunk_size : int or 'auto', default 'auto'
        Number of genes per batched NUFFT call. 32–128 balances finufft's
        per-call overhead against the `(n_spots, chunk)` transient RAM.
        ``'auto'`` sizes the chunk from the per-sample k-grid shapes via
        :func:`quadsv.statistics.resolve_chunk_size` — the NUFFT cache
        sweet-spot cap (64) capped further by the live-memory budget.
    workers : int, optional
        Forwarded to per-sample FFTs used by :meth:`normalize_covariates`.

    Notes
    -----
    The comparator carries no design / contrast state — supply the
    cross-sample contrast directly to :meth:`test_diff_freq` /
    :meth:`test_diff_expr`. A single fitted comparator can therefore
    serve any number of unrelated comparisons on the same spectra.
    """

    # NUFFT cache sweet-spot cap for nufft_chunk_size='auto' (statistics.auto_chunk_size).
    _auto_chunk_cap: int = 64

    def __init__(  # noqa: C901 — flat per-arg config assembly + per-sample grid setup
        self,
        samples: Sequence[Any],
        gene_names: Sequence[str] | None = None,
        *,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        n_theta_bins: int = 36,
        obsm_key: str = "spatial",
        layer: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
        spacing: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        freq_edges: np.ndarray | None = None,
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        nufft_chunk_size: int | str = "auto",
        workers: int | None = None,
    ) -> None:
        fft_solver = _validate_common(feature_mode, "fft2", presence_threshold)
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, _ad.AnnData):
                raise TypeError(f"sample {i} is {type(s).__name__}, expected anndata.AnnData.")

        resolved = _resolve_anndata_gene_names(samples_list, gene_names, layer=layer)

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
        # 'auto' resolved below once per-sample k-grids are known; int fixed now.
        self._nufft_chunk_size_spec = self._normalize_chunk_spec(nufft_chunk_size)
        self._nufft_chunk_size = (
            64 if self._nufft_chunk_size_spec == "auto" else (self._nufft_chunk_size_spec)
        )
        # NUFFT always produces full-2D layout (fft2), regardless of user's
        # ``fft_solver`` (which is moot here).
        self._spectrum_fft_solver = "fft2"

        self._layer = layer
        self._obsm_key = obsm_key
        self._nufft_eps = float(eps)

        # Per-sample coords / grids.
        from quadsv.kernels.nufft import _infer_grid_from_coords

        if unit_scales is None:
            unit_scales = [1.0] * len(samples_list)
        if len(unit_scales) != len(samples_list):
            raise ValueError(
                f"unit_scales length {len(unit_scales)} does not match "
                f"n_samples={len(samples_list)}."
            )
        self._unit_scales: list[float] = [float(s) for s in unit_scales]

        # ``grid_shape`` / ``spacing`` may be a single pair (applied to every
        # sample) or one pair per sample, so heterogeneous samples can be pinned
        # to grids whose frequencies share the same physical units. The manual
        # override is taken only when BOTH are supplied; otherwise each sample's
        # k-grid is inferred from its (unit-scaled) coords.
        grid_override = self._broadcast_pairs(
            grid_shape, len(samples_list), name="grid_shape", cast=int
        )
        spacing_override = self._broadcast_pairs(
            spacing, len(samples_list), name="spacing", cast=float
        )

        coords_list: list[np.ndarray] = []
        grids: list[tuple[int, int]] = []
        spacings: list[tuple[float, float]] = []
        for i, ad_s in enumerate(samples_list):
            if obsm_key not in ad_s.obsm:
                raise KeyError(
                    f"sample {i} has no obsm['{obsm_key}']; "
                    f"available: {list(ad_s.obsm.keys())}."
                )
            c = np.asarray(ad_s.obsm[obsm_key], dtype=np.float64)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(f"sample {i} obsm['{obsm_key}'] must be (n, 2), got {c.shape}.")
            coords_list.append(c)
            if grid_override is not None and spacing_override is not None:
                gs_i = grid_override[i]
                sp_i = spacing_override[i]
            else:
                gs_i, sp_i = _infer_grid_from_coords(c * self._unit_scales[i], oversample=2.0)
            grids.append(gs_i)
            spacings.append(sp_i)
        self._coords = coords_list
        self._grid_shapes = grids
        self._spacings = spacings
        # Now that per-sample k-grids are known, resolve an 'auto' chunk size.
        if self._nufft_chunk_size_spec == "auto":
            self._nufft_chunk_size = self._resolve_chunk_size("auto", grids)
            logger.info(
                "auto nufft_chunk_size=%d (max k-grid %d px, cap %d, budget %.1f GiB).",
                self._nufft_chunk_size,
                max((ny * nx for (ny, nx) in grids), default=1),
                self._auto_chunk_cap,
                self._auto_chunk_budget_bytes / 1024**3,
            )

    # ------------------------------------------------------------------
    def _nufft_spectrum_chunker(self, i: int):
        """Build ``(spectrum_chunk_fn, dc, presence, n_genes, grid)`` for sample ``i``.

        ``spectrum_chunk_fn(start, stop)`` mean-centres + NUFFTs just that
        gene-chunk to its ``(stop-start, ny, nx)`` power spectrum, filling the
        shared ``dc`` / ``presence`` arrays on the way. The dense
        ``(n_genes, ny, nx)`` spectrum is never assembled by the caller.
        """
        from quadsv.kernels.nufft import power_spectrum_2d_nufft

        adata = self.samples[i]
        pts = self._coords[i]
        scale = self._unit_scales[i]
        grid_i = self._grid_shapes[i]
        spacing_i = self._spacings[i]
        X_src = adata.X if self._layer is None else adata.layers[self._layer]
        n_genes = len(self.gene_names)
        n_spots = X_src.shape[0]

        if sp.issparse(X_src):
            dc = np.asarray(X_src.mean(axis=0)).ravel()
            nnz_per = np.asarray((X_src != 0).sum(axis=0)).ravel()
            X_csc = X_src.tocsc()
            X_dense = None
        else:
            X_dense = np.asarray(X_src, dtype=np.float64)
            dc = X_dense.mean(axis=0)
            nnz_per = (X_dense != 0).sum(axis=0)
            X_csc = None
        presence = (nnz_per / max(n_spots, 1)) >= self._presence_threshold

        def _spec_chunk(start: int, stop: int) -> np.ndarray:
            cols = slice(start, stop)
            if X_csc is not None:
                block = np.asarray(X_csc[:, cols].toarray(), dtype=np.float64)
            else:
                block = X_dense[:, cols].astype(np.float64, copy=True)
            # Per-gene mean centering removes the DC bin and prevents per-sample
            # mean-shift leakage into low-frequency bins. Raw DC scalars are kept
            # on ``self.dc_`` for the complementary :meth:`test_diff_expr` path.
            block -= dc[None, cols]
            p_chunk = power_spectrum_2d_nufft(
                pts,
                block,
                grid_shape=grid_i,
                spacing=spacing_i,
                unit_scale=scale,
                eps=self._nufft_eps,
                center_coords=True,
            )
            return np.moveaxis(p_chunk, -1, 0)  # (chunk, ny, nx)

        return _spec_chunk, dc, presence, n_genes, grid_i

    def _compute_spectra(  # noqa: C901 — radial-stream / 2d two-pass dispatch
        self,
        n_jobs: int,
        progress: bool,
        landmark_genes: Sequence[str] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        chunk_size = self._nufft_chunk_size
        n_samples_total = len(self.samples)
        self._resolve_freq_edges()

        if self.feature_mode != "radial":
            return self._compute_spectra_2d(n_jobs, progress, landmark_genes)

        # Per-sample parallelism helper
        def _one(i: int, pbar: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            spec_chunk_fn, dc, presence, n_genes, grid_i = self._nufft_spectrum_chunker(i)
            spacing_i = self._spacings[i]
            # Within each sample, compute the radial spectrum sequentially, chunk by chunk
            feat = stream_radial_features(
                spec_chunk_fn,
                n_genes,
                grid_i,
                chunk_size=chunk_size,
                n_bins=self._n_radial_bins,
                fft_solver=self._spectrum_fft_solver,
                spacing=spacing_i,
                # Shared edges were resolved at the top of this backend dispatcher.
                edges=self.freq_edges,
                pbar=pbar,
            )
            return feat, dc, presence

        return _unpack_sample_triples(
            _run_per_sample(
                _one,
                n_samples_total,
                n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk_size)),
                desc="NUFFT spectra (per-gene chunks)",
                n_jobs=n_jobs,
                progress=progress,
            )
        )

    def _compute_spectra_2d(  # noqa: C901 — two-pass landmark/feature worker setup
        self,
        n_jobs: int,
        progress: bool,
        landmark_genes: Sequence[str] | None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Two-pass streamed rotation alignment (mirrors ``ComparatorGrid``)."""
        chunk_size = self._nufft_chunk_size
        n_samples = len(self.samples)
        name_to_idx = {g: j for j, g in enumerate(self.gene_names)}
        lm_idx: np.ndarray | None = None
        if landmark_genes is not None:
            missing = [g for g in landmark_genes if g not in name_to_idx]
            if missing:
                raise KeyError(f"landmark_genes not in gene_names: {missing}")
            lm_idx = np.asarray([name_to_idx[g] for g in landmark_genes], dtype=int)
        # Cache explicit landmark spectra only within budget; else warn + use a
        # single streamed geometric-mean of the landmark genes.
        cache_landmarks = lm_idx is not None and self._landmark_cache_fits(len(lm_idx))

        # Pass A: landmark + dc/presence per sample.
        grids = list(self._grid_shapes)

        # Per-sample parallelism helper for landmark spectra
        def _landmark_one(i: int, pbar: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            spec_chunk_fn, dc, presence, n_genes, grid_i = self._nufft_spectrum_chunker(i)
            if lm_idx is None:
                lm = stream_geomean_landmark(
                    spec_chunk_fn, n_genes, grid_i, chunk_size=chunk_size, pbar=pbar
                )
            else:
                if cache_landmarks:
                    parts = []
                    for j in lm_idx:
                        parts.append(spec_chunk_fn(int(j), int(j) + 1))
                        if pbar is not None:
                            pbar.update(1)
                    lm = np.concatenate(parts, axis=0)
                else:
                    lm = stream_geomean_landmark(
                        spec_chunk_fn,
                        n_genes,
                        grid_i,
                        chunk_size=chunk_size,
                        gene_subset=lm_idx,
                        pbar=pbar,
                    )
            return lm, dc, presence

        if lm_idx is None:
            n_pass_a_chunks = int(np.ceil(len(self.gene_names) / chunk_size))
        elif cache_landmarks:
            n_pass_a_chunks = int(len(lm_idx))
        else:
            n_pass_a_chunks = int(np.ceil(len(lm_idx) / chunk_size))
        landmarks, dc, presence = _unpack_sample_triples(
            _run_per_sample(
                _landmark_one,
                n_samples,
                n_chunks_per_sample=max(1, n_pass_a_chunks),
                desc="NUFFT 2D landmarks (per-sample chunks)",
                n_jobs=n_jobs,
                progress=progress,
            )
        )

        angles = estimate_rotations_from_landmarks(
            landmarks, grids, fft_solver=self._spectrum_fft_solver, progress=progress
        )
        self.rotation_angles_ = angles

        # Pass B: re-stream, rotate, physical-frequency polar resample.
        # Per-sample parallelism helper for feature spectra (binned)
        def _feature_one(i: int, pbar: Any = None) -> np.ndarray:
            spec_chunk_fn, _dc, _pres, n_genes, grid_i = self._nufft_spectrum_chunker(i)
            return stream_polar_features(
                spec_chunk_fn,
                n_genes,
                grid_i,
                float(angles[i]),
                chunk_size=chunk_size,
                # Shared edges were resolved at the top of this backend dispatcher.
                freq_edges=self.freq_edges,
                spacing=self._spacings[i],
                n_theta=self._n_theta_bins,
                fft_solver=self._spectrum_fft_solver,
                pbar=pbar,
            )

        feats = _run_per_sample(
            _feature_one,
            n_samples,
            n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk_size)),
            desc="NUFFT 2D features (per-sample chunks)",
            n_jobs=n_jobs,
            progress=progress,
        )
        return [np.asarray(f) for f in feats], dc, presence

    # ------------------------------------------------------------------
    def _covariate_features_from_keys(  # noqa: C901 — per-key obs/var dispatch + per-sample loop
        self, keys: Sequence[str]
    ) -> list[np.ndarray]:
        """Per-spot column lookup → per-sample covariate features.

        Each ``key`` is resolved against the first sample as either:

        - an ``adata.obs`` column (per-spot scalar — typical for
          deconvolution outputs, region labels, depth proxies); or
        - an entry in ``adata.var_names`` (treats that gene's
          per-spot expression as the covariate — useful for
          regressing out a housekeeping gene's spatial pattern).

        Resolution prefers ``obs`` when a name appears in both. Every
        subsequent sample must resolve each key to the same source as
        the first sample (i.e., a key is "obs everywhere" or "var
        everywhere") — anything else is treated as a schema mismatch.

        For each sample the resolved per-spot vectors are stacked into
        an ``(n_obs, n_covariates)`` block, mean-centred per column,
        and NUFFTed directly onto the sample's k-grid. The 2-D spectra
        are then radial-binned with the same edges as the gene panel.

        Raises
        ------
        KeyError
            If a key is missing from both ``adata.obs.columns`` and
            ``adata.var_names`` in any sample, or if a key resolves to
            different sources across samples.
        ValueError
            If an obs column cannot be cast to float (e.g., string
            categoricals — encode them first).
        """
        from quadsv.kernels.nufft import power_spectrum_2d_nufft

        keys = list(keys)
        # Classify each key once against the first sample; require all
        # later samples to agree on the source.
        first = self.samples[0]
        sources: dict[str, str] = {}
        for k in keys:
            in_obs = k in first.obs.columns
            in_var = k in first.var_names
            if not (in_obs or in_var):
                raise KeyError(
                    f"covariate key {k!r} is in neither obs.columns nor "
                    f"var_names of sample 0. Available obs (first 10): "
                    f"{list(first.obs.columns)[:10]}; available var_names "
                    f"(first 10): {list(first.var_names)[:10]}."
                )
            sources[k] = "obs" if in_obs else "var"

        out: list[np.ndarray] = []
        for i, adata in enumerate(self.samples):
            cols: list[np.ndarray] = []
            for k in keys:
                src = sources[k]
                if src == "obs":
                    if k not in adata.obs.columns:
                        raise KeyError(
                            f"sample {i} resolves covariate {k!r} differently from "
                            f"sample 0 (sample 0 → obs, sample {i} → not in obs)."
                        )
                    try:
                        cols.append(np.asarray(adata.obs[k], dtype=np.float64))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"sample {i} obs[{k!r}] cannot be cast to float "
                            f"({type(exc).__name__}); encode categoricals before passing."
                        ) from exc
                else:  # var_names path
                    if k not in adata.var_names:
                        raise KeyError(
                            f"sample {i} resolves covariate {k!r} differently from "
                            f"sample 0 (sample 0 → var_names, sample {i} → not in "
                            "var_names)."
                        )
                    idx = adata.var_names.get_loc(k)
                    X_src = adata.X if self._layer is None else adata.layers[self._layer]
                    col = X_src[:, idx]
                    if sp.issparse(col):
                        col = col.toarray()
                    cols.append(np.asarray(col, dtype=np.float64).ravel())
            block = np.column_stack(cols)  # (n_obs, n_cov)
            # Mean-centre each column, matching the gene panel's per-column DC removal.
            block = block - block.mean(axis=0, keepdims=True)

            p = power_spectrum_2d_nufft(
                self._coords[i],
                block,
                grid_shape=self._grid_shapes[i],
                spacing=self._spacings[i],
                unit_scale=self._unit_scales[i],
                eps=self._nufft_eps,
                center_coords=True,
            )
            # power_spectrum_2d_nufft returns (ny, nx, M) for multi-column values.
            cov_2d = np.moveaxis(p, -1, 0)  # (n_cov, ny, nx)
            ny, nx = self._grid_shapes[i]
            if self.feature_mode == "radial":
                cov_feat = radial_bin_spectrum(
                    cov_2d,
                    grid_shape=(ny, nx),
                    n_bins=self._n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=self._spacings[i],
                    edges=self.freq_edges,
                )
            else:
                if self.freq_edges is None:
                    raise RuntimeError("2D covariate features require .compute_spectra() first.")
                angle = 0.0 if self.rotation_angles_ is None else float(self.rotation_angles_[i])

                def _cov_chunk(start: int, stop: int, _cov_2d: np.ndarray = cov_2d) -> np.ndarray:
                    return _cov_2d[start:stop]

                cov_feat = stream_polar_features(
                    _cov_chunk,
                    cov_2d.shape[0],
                    (ny, nx),
                    angle,
                    chunk_size=max(1, cov_2d.shape[0]),
                    freq_edges=self.freq_edges,
                    spacing=self._spacings[i],
                    n_theta=self._n_theta_bins,
                    fft_solver=self._spectrum_fft_solver,
                )
            out.append(cov_feat)
        return out


# ---------------------------------------------------------------------------
# Gene-name resolution helpers
# ---------------------------------------------------------------------------


def _resolve_anndata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    *,
    layer: str | None,
) -> list[str]:
    first = samples[0]
    if gene_names is None:
        gene_names = list(first.var_names)
    for i, s in enumerate(samples):
        if list(s.var_names) != list(gene_names):
            raise ValueError(
                f"sample {i} has var_names that do not match the reference "
                "(all AnnData samples must share the same gene axis)."
            )
        if layer is not None and layer not in s.layers:
            raise KeyError(f"sample {i} is missing layer '{layer}'.")
    return list(gene_names)
