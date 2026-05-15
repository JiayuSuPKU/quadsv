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
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from quadsv.comparators.base import (
    _ComparatorBase,
    _validate_common,
)
from quadsv.comparators.multisample import radial_bin_spectrum

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
    obsm_key : str, default 'spatial'
    layer : str, optional
    unit_scales : sequence of float, optional
        Per-sample multiplier applied to coords before NUFFT (e.g. pixels→μm).
    grid_shape, spacing : optional
        When both given, used for every sample. Otherwise each sample's
        k-grid is auto-inferred from coords via
        :func:`quadsv.kernels.nufft._infer_grid_from_coords`.
    freq_edges : np.ndarray, optional
    eps : float, default 1e-6
        NUFFT tolerance.
    presence_threshold : float, default 0.0
        Minimum fraction of non-zero spots for a gene to count as "observed"
        in a sample (feeds :attr:`presence_` and, transitively, the masked
        pattern test).
    nufft_chunk_size : int, default 64
        Number of genes per batched NUFFT call. 32–128 balances finufft's
        per-call overhead against the `(n_spots, chunk)` transient RAM.
    workers : int, optional
        Forwarded to per-sample FFTs used by :meth:`normalize_covariates`.

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
        gene_names: Sequence[str] | None = None,
        *,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        obsm_key: str = "spatial",
        layer: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        nufft_chunk_size: int = 64,
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
        self._fft_solver = fft_solver
        self._workers = workers
        self._presence_threshold = float(presence_threshold)
        self._nufft_chunk_size = max(1, int(nufft_chunk_size))
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
            if grid_shape is not None and spacing is not None:
                gs_i = (int(grid_shape[0]), int(grid_shape[1]))
                sp_i = (float(spacing[0]), float(spacing[1]))
            else:
                gs_i, sp_i = _infer_grid_from_coords(c * self._unit_scales[i], oversample=2.0)
            grids.append(gs_i)
            spacings.append(sp_i)
        self._coords = coords_list
        self._grid_shapes = grids
        self._spacings = spacings

    # ------------------------------------------------------------------
    def _compute_spectra(  # noqa: C901
        self, n_jobs: int, progress: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        from quadsv.kernels.nufft import power_spectrum_2d_nufft

        chunk_size = self._nufft_chunk_size
        n_samples_total = len(self.samples)

        def _one(i: int, pbar: tqdm | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

            presence_i = (nnz_per / max(n_spots, 1)) >= self._presence_threshold

            ny, nx = grid_i
            spec_stack = np.empty((n_genes, ny, nx), dtype=np.float64)

            for start in range(0, n_genes, chunk_size):
                stop = min(start + chunk_size, n_genes)
                cols = slice(start, stop)
                if X_csc is not None:
                    block = np.asarray(X_csc[:, cols].toarray(), dtype=np.float64)
                else:
                    block = X_dense[:, cols].astype(np.float64, copy=True)

                # Per-gene mean centering: removes the DC bin and prevents
                # per-sample mean-shift leakage into low-frequency bins. The
                # raw DC scalars are preserved on ``self.dc_`` for the
                # complementary :meth:`test_diff_expr` path.
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
                spec_stack[start:stop] = np.moveaxis(p_chunk, -1, 0)
                if pbar is not None:
                    pbar.update(1)

            return spec_stack, dc, presence_i

        return _run_per_sample(
            _one,
            n_samples_total,
            n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk_size)),
            desc="NUFFT spectra (per-gene chunks)",
            n_jobs=n_jobs,
            progress=progress,
        )

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
                k_max = min(self._n_radial_bins, ny // 2, nx // 2)
                low = (
                    cov_2d[:, :k_max, :k_max] if cov_2d.shape[-1] > k_max else cov_2d[:, :k_max, :]
                )
                cov_feat = low.reshape(low.shape[0], -1)
            out.append(cov_feat)
        return out


# ---------------------------------------------------------------------------
# shared per-sample runner
# ---------------------------------------------------------------------------


def _run_per_sample(
    worker: Any,
    n_samples_total: int,
    *,
    n_chunks_per_sample: int,
    desc: str,
    n_jobs: int,
    progress: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Invoke ``worker(i, pbar)`` for each sample with a shared tqdm bar.

    Used by :class:`ComparatorIrregular` where each sample is split into
    multiple per-gene-chunk tqdm ticks.
    """
    raw_2d: list[np.ndarray | None] = [None] * n_samples_total
    dc_list: list[np.ndarray | None] = [None] * n_samples_total
    pres_list: list[np.ndarray | None] = [None] * n_samples_total

    run_sequential = progress or n_jobs == 1
    if run_sequential:
        n_total = n_samples_total * n_chunks_per_sample
        pbar: tqdm | None = tqdm(total=n_total, desc=desc) if progress else None
        for i in range(n_samples_total):
            if pbar is not None:
                pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
            r0, r1, r2 = worker(i, pbar)
            raw_2d[i] = r0
            dc_list[i] = r1
            pres_list[i] = r2
        if pbar is not None:
            pbar.close()
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(i, None) for i in range(n_samples_total)
        )
        for i, r in enumerate(results):
            raw_2d[i], dc_list[i], pres_list[i] = r

    dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
    presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
    return [np.asarray(x) for x in raw_2d], dc, presence


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
