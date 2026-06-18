"""
Shared mixin and input-validation helpers for the comparator layer.

This module hosts the private :class:`_ComparatorBase` mixin that
:class:`~quadsv.ComparatorIrregular` and :class:`~quadsv.ComparatorGrid`
inherit from. The mixin owns:

- the ``compute_spectra`` driver that turns per-sample 2-D images
  into the ``(n_samples, n_genes, n_feature_bins)`` ``spectra_`` tensor;
- the chainable preprocessing methods ``normalize_background()`` and
  ``normalize_covariates(covariates)`` — thin wrappers around the
  same-named standalone functions in
  :mod:`quadsv.comparators.normalization` that mutate ``spectra_`` in
  place;
- the test methods ``test_diff_freq(design, ...)`` and
  ``test_diff_expr(design, ...)`` — design-at-call-time so a single
  fitted comparator can serve any number of unrelated contrasts on
  the same spectra;
- the diagnostic ``effective_rank()``.

The shape-only / sum-1 feature representation is reached via the
``normalize_shape: bool = False`` keyword on :meth:`test_diff_freq`
(forwarded to its dispatch target), not via a chainable method — this
keeps the per-test choice non-destructive.

Concrete classes live in sibling modules:
:mod:`quadsv.comparators.irregular` and
:mod:`quadsv.comparators.grid`.
"""

from __future__ import annotations

import copy
import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

from quadsv.comparators.features import (
    compute_sample_spectrum,
    radial_bin_counts,
    radial_bin_spectrum,
    stream_polar_features,
)
from quadsv.comparators.features import (
    gene_pattern_diversity as _gene_pattern_diversity,
)
from quadsv.comparators.multisample import (
    _estimate_glm_masked_null_covariance,
    _estimate_glm_null_covariance,
    _estimate_two_group_masked_null_covariance,
    _estimate_two_group_null_covariance,
    compare_glm,
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
)

# Aliased to leading-underscore names to avoid shadowing the like-named
# instance methods on the comparator class below.
from quadsv.comparators.normalization import (
    normalize_background as _normalize_background,
)
from quadsv.comparators.normalization import (
    normalize_covariates as _normalize_covariates,
)

if TYPE_CHECKING:
    from quadsv.comparators.multisample import _AnalyticNullState

__all__: list[str] = []

logger = logging.getLogger(__name__)

# Helper functions for running per-sample computations.


def _run_per_sample(
    worker: Any,
    n_samples_total: int,
    *,
    n_chunks_per_sample: int,
    desc: str,
    n_jobs: int,
    progress: bool,
) -> list[Any]:
    """Invoke ``worker(i, pbar)`` for each sample, preserving sample order."""
    out: list[Any] = [None] * n_samples_total

    run_sequential = progress or n_jobs == 1
    if run_sequential:
        n_total = n_samples_total * n_chunks_per_sample
        pbar: tqdm | None = tqdm(total=n_total, desc=desc) if progress else None
        for i in range(n_samples_total):
            if pbar is not None:
                pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
            out[i] = worker(i, pbar)
        if pbar is not None:
            pbar.close()
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(i, None) for i in range(n_samples_total)
        )
        for i, r in enumerate(results):
            out[i] = r

    return out


def _unpack_sample_triples(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Convert ``[(features, dc, presence), ...]`` to comparator return values."""
    feats, dc_list, pres_list = zip(*results, strict=True)
    dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
    presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
    return [np.asarray(x) for x in feats], dc, presence


def _unpack_sample_quads(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, Any]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[Any]]:
    """Convert ``[(features, dc, presence, extra), ...]`` to stacked outputs."""
    feats, dc_list, pres_list, extra = zip(*results, strict=True)
    dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
    presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
    return [np.asarray(x) for x in feats], dc, presence, list(extra)


class _ComparatorBase:
    """Shared state + shared methods for NUFFT / FFT pattern comparators.

    Subclasses populate the small public surface and the underscored
    internal state listed below in ``__init__``, then implement
    :meth:`_compute_spectra` to fill the post-``compute_spectra``
    attributes. The comparator carries **no design / contrast state**:
    cross-sample comparisons are specified per-call on
    :meth:`test_diff_freq` and :meth:`test_diff_expr`, so a single
    fitted comparator can serve any number of unrelated contrasts on
    the same ``spectra_``.

    Public attributes (user-facing, inspection-worthy)
    --------------------------------------------------
    samples : list
        The exact ``samples`` sequence passed at construction (AnnData
        list for the irregular path, SpatialData list for the grid
        path). Reference-shared with the user's collection.
    gene_names : list[str]
        Per-gene labels (length ``n_genes``).
    feature_mode : {'radial', '2d'}
        Spectral feature representation chosen at construction.
    freq_edges : np.ndarray | None
        Shared radial-frequency bin edges used by both feature modes.
        Auto-derived during :meth:`compute_spectra` if not supplied at
        construction.

    Public attributes set by :meth:`compute_spectra`
    -------------------------------------------------
    spectra_ : np.ndarray | None
        Per-(sample, gene) spectral feature matrix, shape
        ``(n_samples, n_genes, n_feature_bins)``. The trailing axis is
        radial bins in ``feature_mode='radial'`` and flattened
        ``radius × theta`` bins in ``feature_mode='2d'``. This is the
        input to :meth:`test_diff_freq` and the in-place
        :meth:`normalize_background` / :meth:`normalize_covariates`
        preprocessing transforms.
    dc_ : np.ndarray | None
        DC component of the spectrum per (sample, gene), shape
        ``(n_samples, n_genes)``. Equals the sample-grid mean of each
        gene's uncentered expression. Input to :meth:`test_diff_expr`.
    presence_ : np.ndarray | None
        Boolean mask of shape ``(n_samples, n_genes)`` — ``True`` where
        a gene's per-sample spot-presence fraction cleared
        ``_presence_threshold``. Drives the masked variant of
        :meth:`test_diff_freq` when any entry is ``False``.
    rotation_angles_ : np.ndarray | None
        Per-sample rotation angle (degrees) applied during
        rotation-alignment. Populated only when ``feature_mode='2d'``.
    freq_edges_requested_ : np.ndarray | None
        The original auto-resolved or requested 1-D edge grid before adaptive
        merging and support filtering. This is for reference purposes only.
    frequency_bin_intervals_ : np.ndarray | None
        The final radial frequency intervals of shape ``(n_retained_radius_bins, 2)``.
        It is stored as intervals because after dropping empty bins the
        edges may not be contiguous.
    frequency_bin_counts_ : np.ndarray | None
        Number of FFT-cell counts for each sample and originally requested radial
        frequency bin, shape ``(n_samples, len(freq_edges_requested_) - 1)``.
        These counts diagnose requested-bin support and distinguish empty
        Fourier-grid intervals from true zero-power measurements in
        :attr:`spectra_`.

    Private state (set by subclasses; not part of the user API)
    -----------------------------------------------------------
    ``_n_radial_bins``, ``_n_theta_bins``, ``_fft_solver``, ``_workers``,
    ``_presence_threshold``, ``_spacings``,
    ``_spectrum_fft_solver``, ``_grid_shapes``, ``_raw_2d_spectra``.
    """

    # --- public attribute stubs populated by subclass __init__ --------
    samples: list[Any]
    gene_names: list[str]
    feature_mode: str
    freq_edges: np.ndarray | None

    # --- private attribute stubs populated by subclass __init__ -------
    _n_radial_bins: int
    _n_theta_bins: int
    _fft_solver: str
    _workers: int | None
    _presence_threshold: float
    _spacings: list[tuple[float, float]] | None
    _spectrum_fft_solver: str
    _grid_shapes: list[tuple[int, int]]

    # --- populated by :meth:`compute_spectra` ---------------------------
    spectra_: np.ndarray | None = None
    dc_: np.ndarray | None = None
    presence_: np.ndarray | None = None
    rotation_angles_: np.ndarray | None = None
    frequency_bin_counts_: np.ndarray | None = None
    freq_edges_requested_: np.ndarray | None = None
    frequency_bin_intervals_: np.ndarray | None = None
    _supported_frequency_bin_mask: np.ndarray | None = None

    _raw_2d_spectra: list[np.ndarray] | None = None

    # Angular resolution of the ``feature_mode='2d'`` polar feature grid:
    # the 2-D feature axis is ``n_radius_bins × _n_theta_bins`` over the
    # half-plane ``[0, π)``. Subclasses may override.
    _n_theta_bins: int = 36

    # ------------------------------------------------------------------
    @abstractmethod
    def _compute_spectra(
        self,
        n_jobs: int,
        progress: bool,
        landmark_genes: Sequence[str] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Stream per-sample **feature** spectra + DC + presence mask.

        Implemented by each backend. The dense full ``(n_genes, ny, n_kx)`` 2D
        spectra should **never** materialised in either mode and peak memory stays
        at ``O(chunk · ny · nx)``. Returns ``(feats, dc, presence)``:

        - ``feature_mode='radial'`` — rasterise/NUFFT → per-gene-chunk FFT →
          radial-bin, discarding each dense chunk; ``feats`` are
          **radial-binned** ``(n_genes, n_feature_bins)`` arrays. No alignment
          needed here, and the ``landmark_genes`` argument is ignored.
        - ``feature_mode='2d'`` — a **two-pass** streamed rotation alignment:
          (A) learn per-sample rotation angles against a reference using
          landmark genes (default: use the per-sample *geometric-mean* spectrum as
          a single landmark, like
          :func:`quadsv.comparators.normalization.normalize_background`; optionally, align
          the spectra of an explicit ``landmark_genes`` set).
          (B) for every gene-chunk, first rotate by the estimated angle, then
          resample onto a polar ``(radius, theta)`` grid whose **radius axis is
          the shared physical-frequency grid** (:attr:`freq_edges`, the same
          edges the radial path uses) and whose ``theta`` axis (length
          :attr:`_n_theta_bins`) carries direction, and flatten — discarding each
          dense chunk. ``feats`` are
          ``(n_genes, n_radius_bins · _n_theta_bins)`` and are cross-sample
          aligned (heterogeneous lattices map the same physical frequency to
          the same radius bin, like radial mode). The recovered angles are
          stored on ``self.rotation_angles_``.

        ``dc`` is ``(n_samples, n_genes)`` and ``presence`` is
        ``(n_samples, n_genes)`` bool. Each backend calls
        :meth:`_resolve_freq_edges` once its per-sample spacing metadata is known
        and before any streamed reducer consumes :attr:`freq_edges`. When an
        explicit ``landmark_genes`` set would require caching the full landmark
        spectra beyond :attr:`_landmark_cache_warn_bytes`, the backend warns and
        falls back to a single streamed geometric mean **of the landmark genes**
        (see :meth:`_landmark_cache_fits`) instead of raising.
        """
        raise NotImplementedError

    # Backend cache sweet-spot cap for ``chunk_size='auto'`` — the empirically
    # tuned caps from :func:`quadsv.statistics.auto_chunk_size` (FFT → 32,
    # NUFFT → 64). Subclasses override. The live-memory budget (across all
    # workers) is :attr:`_auto_chunk_budget_bytes`.
    _auto_chunk_cap: int = 32
    _auto_chunk_budget_bytes: int = 2 * 1024**3  # 2 GiB (statistics default)

    @staticmethod
    def _normalize_chunk_spec(spec: int | str) -> int | str:
        """Validate a chunk-size spec at construction; return ``'auto'`` or a ``>=1`` int."""
        if isinstance(spec, str):
            if spec != "auto":
                raise ValueError(f"chunk_size must be a positive int or 'auto', got {spec!r}.")
            return "auto"
        return max(1, int(spec))

    @staticmethod
    def _normalize_n_theta_bins(n_theta_bins: int) -> int:
        """Validate the angular bin count used by ``feature_mode='2d'``."""
        n_theta = int(n_theta_bins)
        if n_theta < 1:
            raise ValueError(f"n_theta_bins must be a positive int, got {n_theta_bins!r}.")
        return n_theta

    @staticmethod
    def _broadcast_pairs(
        value: Any, n_samples: int, *, name: str, cast: Any
    ) -> list[tuple[Any, Any]] | None:
        """Normalise a single ``(a, b)`` pair *or* a per-sample sequence of pairs.

        Returns a length-``n_samples`` list of ``(cast(a), cast(b))`` tuples, or
        ``None`` when ``value`` is ``None``. A single ``(a, b)`` (1-D, length 2)
        is broadcast to every sample; an ``(n_samples, 2)`` sequence is taken
        per-sample. This lets ``spacing`` / ``grid_shape`` be set globally or
        adjusted per sample so the resulting grids — and thus frequencies — are
        in the same physical units across heterogeneous samples.
        """
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == 2:
            return [(cast(arr[0]), cast(arr[1]))] * n_samples
        if arr.ndim == 2 and arr.shape[1] == 2:
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"per-sample {name} has {arr.shape[0]} rows but n_samples={n_samples}."
                )
            return [(cast(r[0]), cast(r[1])) for r in arr]
        raise ValueError(
            f"{name} must be a 2-tuple (dy, dx) or an (n_samples, 2) sequence of them; "
            f"got shape {tuple(arr.shape)}."
        )

    # ------------------------------------------------------------------
    def _resolve_chunk_size(
        self, spec: int | str, grid_shapes: list[tuple[int, int]], *, n_jobs: int = 1
    ) -> int:
        """Resolve a chunk-size spec (an int, or ``'auto'``) to an int.

        ``'auto'`` reuses :func:`quadsv.statistics.resolve_chunk_size` — the
        same cache sweet-spot cap (:attr:`_auto_chunk_cap`) and live-memory
        budget (:attr:`_auto_chunk_budget_bytes`) as the Q/R-test chunker — with
        ``per_feat = max(ny·nx) · 8`` bytes (one dense lattice block per gene).
        An int is returned as-is (floored at 1).
        """
        if not isinstance(spec, str):
            return max(1, int(spec))
        if spec != "auto":
            raise ValueError(f"chunk_size must be a positive int or 'auto', got {spec!r}.")
        from quadsv.statistics import resolve_chunk_size

        max_lat = max((ny * nx for (ny, nx) in grid_shapes), default=1)
        return resolve_chunk_size(
            self._auto_chunk_cap,
            max(max_lat, 1) * 8,
            n_jobs=n_jobs,
            budget_bytes=self._auto_chunk_budget_bytes,
        )

    # Rotation-landmark cache budget (bytes). An explicit ``landmark_genes``
    # set in 2d mode needs its full ``(n_landmarks, ny, n_kx)`` spectra cached
    # per sample to estimate angles; if the largest such cache would exceed this
    # we raise rather than risk an OOM. The streamed *geomean* landmark (the
    # default) is a single ``(1, ny, n_kx)`` spectrum and never trips this.
    _landmark_cache_warn_bytes: int = 8 * 1024**3  # 8 GiB

    # ------------------------------------------------------------------
    def _landmark_cache_fits(self, n_landmarks: int) -> bool:
        """Whether caching ``n_landmarks`` full-2D spectra per sample fits budget.

        The explicit-``landmark_genes`` path (2d mode) holds each sample's
        ``(n_landmarks, ny, n_kx)`` landmark spectra in memory to estimate the
        rotation. When the worst-case cache would exceed
        :attr:`_landmark_cache_warn_bytes` this returns ``False`` so the caller
        can fall back to a single streamed geometric-mean landmark (built over
        just the landmark genes) instead of risking an OOM. Returns ``True``
        (cache the per-gene landmark spectra) otherwise.
        """
        if not self._grid_shapes:
            return True
        worst = 0
        for ny, nx in self._grid_shapes:
            n_kx = nx if self._spectrum_fft_solver == "fft2" else nx // 2 + 1
            worst = max(worst, n_landmarks * ny * n_kx * 8)
        if worst > self._landmark_cache_warn_bytes:
            warnings.warn(
                f"Rotation alignment with {n_landmarks} explicit landmark genes would cache "
                f"up to {worst / 1024**3:.1f} GiB of 2D spectra per sample "
                f"(limit {self._landmark_cache_warn_bytes / 1024**3:.1f} GiB). Falling back "
                "to a single streamed geometric-mean of the landmark genes to avoid OOM. "
                "Pass fewer landmark_genes or raise _landmark_cache_warn_bytes to keep the "
                "per-gene landmark alignment.",
                stacklevel=2,
            )
            return False
        return True

    # ------------------------------------------------------------------
    def _resolve_freq_edges(self) -> None:
        """Populate :attr:`freq_edges` (shared physical-frequency grid), if unset.

        Uses the per-sample physical ``self._spacings`` (already known before
        the spectra pass for both backends) to build a common bin grid on
        ``[0, min Nyquist]``. Used by **both** feature modes: the radial path
        bins onto these edges, and the 2-D path resamples its polar radius axis
        onto the same edges so heterogeneous lattices stay cross-sample aligned.
        Idempotent and a no-op when ``freq_edges`` is already set or cannot yet
        be derived.
        """
        if not self._spacings:
            return

        # Request linearly spaced frequency edges if no explicit edges are provided.
        if self.freq_edges is None:
            # Find the minimum Nyquist frequency across all samples.
            nyquists = [1.0 / (2.0 * max(dy, dx)) for (dy, dx) in self._spacings]
            f_max = float(min(nyquists))
            # Create the shared frequency edges.
            self.freq_edges = np.linspace(0.0, f_max * (1.0 + 1e-9), self._n_radial_bins + 1)
            logger.info(
                "Auto-generated %d radial bins on [0, %.4g] cycles per unit length.",
                self._n_radial_bins,
                f_max,
            )

        # For diagnostic purposes, keep a copy of the requested frequency edges.
        if self.freq_edges_requested_ is None and self.freq_edges is not None:
            self.freq_edges_requested_ = np.asarray(self.freq_edges, dtype=float).copy()

        # Optionally, merge empty radial bins to ensure bins are supported in all samples.
        if getattr(self, "_adaptive_freq_edges", False):
            self._adapt_freq_edges()

    def _fft_cell_counts_for_edges(self, edges: np.ndarray | None = None) -> np.ndarray | None:
        """Return per-sample FFT-cell counts for a radial-frequency edge grid."""
        if edges is None:
            edges = self.freq_edges
        if edges is None or not self._grid_shapes or not self._spacings:
            return None
        return np.vstack(
            [
                radial_bin_counts(
                    shape,
                    n_bins=self._n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=spacing,
                    edges=edges,
                )
                for shape, spacing in zip(self._grid_shapes, self._spacings, strict=True)
            ]
        )

    def _adapt_freq_edges(self) -> None:
        """Merge adjacent radial bins until every retained bin has sample support."""
        if self.freq_edges is None:
            return
        if not self._grid_shapes or not self._spacings:
            return

        # Compute number of FFT-cells in each radial bin from the current edges.
        edges = np.asarray(self.freq_edges, dtype=float)
        requested = edges.size - 1
        feature_grid_counts = self._fft_cell_counts_for_edges(edges)
        if feature_grid_counts is None:
            return

        # Accumulate adjacent original bins until all samples have at least one
        # FFT cell in the merged interval. This is equivalent to repeatedly
        # merging unsupported bins, but avoids recomputing per-sample histograms
        # after every edge deletion.
        merged_edges = [float(edges[0])]
        running = np.zeros(feature_grid_counts.shape[0], dtype=float)
        for bin_idx in range(requested):
            running += feature_grid_counts[:, bin_idx]
            if np.all(running > 0):
                merged_edges.append(float(edges[bin_idx + 1]))
                running.fill(0.0)

        if len(merged_edges) == 1:
            merged_edges.append(float(edges[-1]))
        elif merged_edges[-1] != float(edges[-1]):
            # Trailing unsupported bins have no right neighbour, so merge them
            # into the final retained interval by extending its right edge.
            merged_edges[-1] = float(edges[-1])

        # Update the frequency edges and the number of radial bins.
        self.freq_edges = np.asarray(merged_edges, dtype=float)
        self._n_radial_bins = int(self.freq_edges.size - 1)
        collapsed = requested - self._n_radial_bins
        if collapsed:
            logger.info(
                "Adaptive frequency edges collapsed %d empty radial bins "
                "(requested=%d, actual=%d).",
                collapsed,
                requested,
                self._n_radial_bins,
            )

    def _select_supported_frequency_bins(
        self, features: np.ndarray, supported_bins: np.ndarray
    ) -> np.ndarray:
        """Subset radial or 2D polar feature arrays to retained radial bins."""
        if supported_bins.all():
            return features
        supported_bins = np.asarray(supported_bins, dtype=bool)
        n_radius = supported_bins.size
        if self.feature_mode == "radial":
            if features.shape[-1] != n_radius:
                raise ValueError(
                    f"radial feature axis has length {features.shape[-1]}, "
                    f"expected {n_radius} bins before support filtering"
                )
            return features[..., supported_bins]

        # In 2D mode, drop whole angular blocks for unsupported radial bins.
        n_theta = self._n_theta_bins
        expected = n_radius * n_theta
        if features.shape[-1] != expected:
            raise ValueError(
                f"2D feature axis has length {features.shape[-1]}, "
                f"expected {expected} = {n_radius} radial bins x {n_theta} angles"
            )
        reshaped = features.reshape(*features.shape[:-1], n_radius, n_theta)
        kept = reshaped[..., supported_bins, :]
        return kept.reshape(*features.shape[:-1], int(supported_bins.sum()) * n_theta)

    def _record_frequency_bin_metadata(
        self,
        feature_grid_counts: np.ndarray | None,
        requested_grid_counts: np.ndarray | None,
    ) -> np.ndarray | None:
        """Record bin diagnostics and return the feature-grid support mask."""
        if feature_grid_counts is None:
            self.frequency_bin_counts_ = None  # 1D
            self.frequency_bin_intervals_ = None  # 2D, (start, end) per retained bin
            self._supported_frequency_bin_mask = None  # 1D
            return None

        # Identify frequency bins in the current grid that are supported by all samples.
        supported_bins = (feature_grid_counts > 0).all(axis=0)
        if not np.any(supported_bins):
            raise RuntimeError("No radial frequency bins are supported by every sample.")

        self._supported_frequency_bin_mask = supported_bins.copy()
        self.frequency_bin_counts_ = (
            requested_grid_counts if requested_grid_counts is not None else feature_grid_counts
        )

        # Record the start and end of each retained frequency bin.
        if self.freq_edges is not None:
            intervals = np.column_stack(
                [np.asarray(self.freq_edges[:-1]), np.asarray(self.freq_edges[1:])]
            )
            self.frequency_bin_intervals_ = intervals[supported_bins]
        else:
            self.frequency_bin_intervals_ = None

        dropped = int((~supported_bins).sum())
        if dropped:
            logger.info(
                "Dropped %d radial bins with missing sample support " "(retained=%d).",
                dropped,
                int(supported_bins.sum()),
            )
        self._n_radial_bins = int(supported_bins.sum())
        return supported_bins

    # ------------------------------------------------------------------
    def compute_spectra(
        self,
        n_jobs: int = -1,
        landmark_genes: Sequence[str] | None = None,
        progress: bool = True,
    ) -> _ComparatorBase:
        """
        Compute per-sample power spectra and (if ``feature_mode='2d'``) rotation-align.

        Parameters
        ----------
        n_jobs : int, default -1
            Parallelism over samples for the per-sample spectrum pass. When
            ``progress=True`` the outer loop is sequential (so the tqdm bar is
            accurate); finufft / scipy.fft are multi-threaded internally via
            OpenMP so this rarely loses in practice.
        landmark_genes : sequence of str, optional
            Only used in ``feature_mode='2d'``. Names of genes (matched against
            :attr:`gene_names`) whose spectra define the rotation-alignment
            landmarks. Recovered rotations are still applied to every gene in
            :attr:`gene_names`. If None (default), a single streamed geometric
            mean over every gene is used as the landmark.
        progress : bool, default True
            Show tqdm progress bars over the three phases (spectrum compute,
            optional rotation alignment, radial binning).

        Returns
        -------
        self
        """
        logger.info(
            "Computing per-sample spectra (n_samples=%d, mean-centered)...",
            len(self.samples),
        )
        # ``_compute_spectra`` streams and returns the **final per-sample
        # features** for both modes — radial-binned ``(n_genes, n_feature_bins)``
        # (radial), or rotation-aligned physical-frequency polar features
        # ``(n_genes, n_radius_bins · _n_theta_bins)`` (2d). The dense full 2D
        # spectra are never held; rotation angles (2d) are recorded on
        # ``self.rotation_angles_`` by the backend.
        self._raw_2d_spectra = None
        per_sample, self.dc_, self.presence_ = self._compute_spectra(
            n_jobs=n_jobs, progress=progress, landmark_genes=landmark_genes
        )

        # The feature grid may be adaptively merged; diagnostics also preserve
        # support on the originally requested grid.
        feature_grid_counts = self._fft_cell_counts_for_edges(self.freq_edges)
        requested_grid_counts = self._fft_cell_counts_for_edges(self.freq_edges_requested_)
        supported_bins = self._record_frequency_bin_metadata(
            feature_grid_counts,
            requested_grid_counts,
        )

        # self._compute_spectra returns the raw features with potential NA values.
        # Keep only features in the supported frequency bins.
        feats = list(per_sample)
        if supported_bins is not None:
            feats = [self._select_supported_frequency_bins(f, supported_bins) for f in feats]
        n_feature_bins = min(f.shape[-1] for f in feats)
        feats = [f[..., :n_feature_bins] for f in feats]
        self.spectra_ = np.stack(feats, axis=0)
        return self

    # ------------------------------------------------------------------
    def subset(self, indices: Sequence[int] | np.ndarray) -> _ComparatorBase:
        """Return a lightweight comparator view over a subset of samples.

        The returned comparator is a shallow copy of ``self``: sample containers
        and gene-level configuration are shared, while per-sample fitted arrays
        and per-sample metadata are indexed to ``indices``. This is intended for
        running multiple contrasts against one fitted comparator without
        recomputing spectra.
        """
        idx = np.asarray(indices)
        if idx.dtype == bool:
            if idx.ndim != 1 or idx.size != len(self.samples):
                raise IndexError(
                    f"boolean sample mask must have length {len(self.samples)}, got {idx.size}"
                )
            idx = np.flatnonzero(idx)
        else:
            idx = idx.astype(int, copy=False)
        idx = np.ravel(idx)

        sub = copy.copy(self)
        sub.samples = [self.samples[i] for i in idx]
        sub.spectra_ = None if self.spectra_ is None else self.spectra_[idx]
        sub.dc_ = None if self.dc_ is None else self.dc_[idx]
        sub.presence_ = None if self.presence_ is None else self.presence_[idx]
        sub.rotation_angles_ = None if self.rotation_angles_ is None else self.rotation_angles_[idx]
        sub.frequency_bin_counts_ = (
            None if self.frequency_bin_counts_ is None else self.frequency_bin_counts_[idx]
        )
        sub.freq_edges_requested_ = self.freq_edges_requested_
        sub.frequency_bin_intervals_ = self.frequency_bin_intervals_
        sub._supported_frequency_bin_mask = self._supported_frequency_bin_mask
        if self._spacings is not None and len(self._spacings) == len(self.samples):
            sub._spacings = [self._spacings[i] for i in idx]
        if self._grid_shapes and len(self._grid_shapes) == len(self.samples):
            sub._grid_shapes = [self._grid_shapes[i] for i in idx]
        if self._raw_2d_spectra is not None and len(self._raw_2d_spectra) == len(self.samples):
            sub._raw_2d_spectra = [self._raw_2d_spectra[i] for i in idx]
        return sub

    # ------------------------------------------------------------------
    # Post-fit transforms
    # ------------------------------------------------------------------
    def normalize_background(self) -> _ComparatorBase:
        """Apply per-sample geometric-mean background normalization in place."""
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_background().")
        for i in range(self.spectra_.shape[0]):
            self.spectra_[i] = _normalize_background(self.spectra_[i])
        return self

    def normalize_covariates(self, covariates: Sequence[Any]) -> _ComparatorBase:
        """Regress out per-sample covariate spectra from :attr:`spectra_`.

        Two input modes, detected from the first element's type:

        - **Sequence of str** — column-key list shared across samples.
          Each string key is considered as one covariate applied to every
          sample. Implementation depends on the subclass:

            * :class:`~quadsv.ComparatorIrregular` looks each key up in
              ``adata.obs.columns`` first, then ``adata.var_names``
              (preferring obs on collision); the resolved per-spot
              vector is NUFFTed directly onto the sample's k-grid.
            * :class:`~quadsv.ComparatorGrid` rasterizes via
              :func:`spatialdata.rasterize_bins` with the keys forwarded
              as ``value_key`` (any combination of ``.obs`` columns and
              ``var_names``).

        - **Sequence of np.ndarray** — per-sample pre-rasterized images,
          one ``(n_covariates, ny_i, nx_i)`` array per sample. Universal
          path; works on either subclass. Use when you want full control
          over rasterization, or when the covariates aren't already
          attached to the sample containers. These rasters must have
          frequency support compatible with the comparator's retained bins.

        Both modes produce the same downstream behaviour: per-sample
        covariate features are reduced to ``(n_covariates, n_feature_bins)``
        (same trailing axis as :attr:`spectra_`) and passed through
        :func:`~quadsv.comparators.normalization.normalize_covariates` to
        log-space-residualise each gene's spectrum against them.

        Parameters
        ----------
        covariates : sequence of str or sequence of np.ndarray
            See modes above. Strings are interpreted by the subclass;
            ndarrays are used verbatim (one per sample).

        Returns
        -------
        self : _ComparatorBase
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_covariates().")
        items = list(covariates)
        if len(items) == 0:
            raise ValueError("covariates must be a non-empty sequence.")

        first = items[0]
        if isinstance(first, str):
            # Shared key-list mode.
            if not all(isinstance(k, str) for k in items):
                raise TypeError(
                    "Mixed str and non-str entries in `covariates=` — pass either a "
                    "list of column-name strings (shared across samples) or a list "
                    "of per-sample (n_cov, ny, nx) arrays."
                )
            cov_features_per_sample = self._covariate_features_from_keys(items)
            covariate_mode = "key"
        elif isinstance(first, np.ndarray):
            if len(items) != len(self.samples):
                raise ValueError(
                    f"covariates length {len(items)} != n_samples {len(self.samples)}."
                )
            cov_features_per_sample = [
                self._covariate_features_from_array(arr, sample_index=i)
                for i, arr in enumerate(items)
            ]
            covariate_mode = "array"
        else:
            raise TypeError(
                f"covariates[0] is {type(first).__name__}; expected str (column-name "
                "mode) or np.ndarray (per-sample image-array mode)."
            )

        if len(cov_features_per_sample) != len(self.samples):
            raise ValueError(
                f"Subclass returned {len(cov_features_per_sample)} covariate-feature "
                f"sets but the comparator has {len(self.samples)} samples."
            )
        for i, cov_feat in enumerate(cov_features_per_sample):
            if self._supported_frequency_bin_mask is not None:
                # Keep only the frequency bins that are supported by all samples.
                cov_feat = self._select_supported_frequency_bins(
                    cov_feat,
                    self._supported_frequency_bin_mask,
                )
            cov_feat = cov_feat[..., : self.spectra_.shape[-1]]
            self._validate_covariate_features(cov_feat, sample_index=i, mode=covariate_mode)
            self.spectra_[i] = _normalize_covariates(self.spectra_[i], cov_feat)
        return self

    def _validate_covariate_features(
        self, cov_feat: np.ndarray, *, sample_index: int, mode: str
    ) -> None:
        """Validate per-sample covariate features before log-space regression."""
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .normalize_covariates().")
        if cov_feat.shape[-1] != self.spectra_.shape[-1]:
            raise ValueError(
                f"sample {sample_index} covariate features have {cov_feat.shape[-1]} "
                f"bins after frequency-bin filtering, expected {self.spectra_.shape[-1]}."
            )
        if not np.isfinite(cov_feat).all():
            mode_label = "pre-rasterized array" if mode == "array" else "key-based"
            raise ValueError(
                f"sample {sample_index} {mode_label} covariate features contain "
                "non-finite values after frequency-bin filtering. This usually means "
                "the covariate grid lacks Fourier support for at least one retained "
                "bin; use covariates on a compatible grid, reduce n_radial_bins, or "
                "enable adaptive_freq_edges."
            )

    # ------------------------------------------------------------------
    def _covariate_features_from_array(self, cov: np.ndarray, sample_index: int) -> np.ndarray:
        """Image-array → ``(n_covariates, n_feature_bins)`` covariate features.

        Shared by both subclasses for the pre-rasterized
        ``(n_covariates, ny, nx)`` input mode. Mirrors the spectrum +
        radial-binning pipeline used on the gene panel itself. The
        sample index is needed only to look up that sample's physical
        ``spacing`` for radial binning.
        """
        if cov.ndim != 3:
            raise ValueError(f"covariate array must be 3D (n_cov, ny, nx), got {cov.shape}.")
        cov_2d = compute_sample_spectrum(
            cov, fft_solver=self._spectrum_fft_solver, workers=self._workers
        )
        # Use the covariate's own raster shape — for the NUFFT path the
        # sample's internal k-grid (self._grid_shapes[i]) is auto-inferred
        # and may differ from the covariate raster. ``freq_edges`` (shared
        # across samples when ``feature_mode='radial'``) is what aligns the
        # bins, not grid_shape.
        cov_shape = cov.shape[-2:]
        spacing = self._spacings[sample_index] if self._spacings is not None else None
        if self.feature_mode == "radial":
            return radial_bin_spectrum(
                cov_2d,
                grid_shape=cov_shape,
                n_bins=self._n_radial_bins,
                fft_solver=self._spectrum_fft_solver,
                spacing=spacing,
                edges=self.freq_edges,
            )
        if self.freq_edges is None:
            raise RuntimeError("2D covariate features require .compute_spectra() first.")
        angles = self.rotation_angles_
        angle = 0.0 if angles is None else float(angles[sample_index])

        def _cov_chunk(start: int, stop: int) -> np.ndarray:
            return cov_2d[start:stop]

        return stream_polar_features(
            _cov_chunk,
            cov_2d.shape[0],
            cov_shape,
            angle,
            chunk_size=max(1, cov_2d.shape[0]),
            freq_edges=self.freq_edges,
            spacing=spacing,
            n_theta=self._n_theta_bins,
            fft_solver=self._spectrum_fft_solver,
        )

    def _covariate_features_from_keys(self, keys: Sequence[str]) -> list[np.ndarray]:
        """Column-key list → per-sample ``(n_covariates, n_feature_bins)`` features.

        Subclass hook. ``ComparatorIrregular`` reads ``adata.obs[key]``
        per sample (NUFFT directly on per-spot values), and
        ``ComparatorGrid`` forwards the keys as ``value_key`` to
        :func:`spatialdata.rasterize_bins`.

        The base class raises so the user gets a clear error if the
        subclass doesn't implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support column-name covariate input; "
            "pass per-sample (n_cov, ny, nx) arrays instead."
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_diff_freq(  # noqa: C901 — dispatcher over three execution paths
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        statistic: str = "log_l2",
        null: str = "analytic",
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        max_exact_permutations: int = 10000,
        normalize_shape: bool = False,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
    ) -> Any:
        """Differential-frequency (DF) test on :attr:`spectra_`.

        Tests whether each gene's radial-frequency power profile differs
        between conditions / along a contrast. The companion DE test is
        :meth:`test_diff_expr`.

        Dispatches between four execution paths, picked from the
        ``design`` argument's shape and the ``contrast=`` keyword:

        - **Binary, analytic null** (1-D ``design``, ``null="analytic"`` (default),
          ``contrast=None``): Liu mixture-χ² test on the binary indicator
          via :func:`~quadsv.comparators.multisample.compare_two_groups`
          (or its masked variant when any ``presence_`` entry is
          ``False``).
        - **Binary, permutation null** (1-D ``design``,
          ``null="permutation"``, ``contrast=None``): two-group
          label-permutation test on the same dispatch target.
        - **GLM analytic** (multi-column / continuous ``design`` **or**
          explicit ``contrast=``): generalized Liu mixture-χ² test via
          :func:`~quadsv.comparators.multisample.compare_glm` (or
          :func:`~quadsv.comparators.multisample.compare_glm_masked`
          when any ``presence_`` entry is ``False``).

        Supplying ``contrast=`` alongside a 1-D ``design`` switches to
        the GLM path on the single-column DataFrame that wraps the
        binary groups (so the same contrast-resolution rules apply).

        Parameters
        ----------
        design : 1-D array, 2-D ndarray, or pandas.DataFrame
            Sample-level contrast specification. Length / first
            dimension must equal ``len(samples)``. See
            :func:`_validate_design` for the accepted forms; can differ
            across calls on the same fitted comparator (fit once, test
            many).
        contrast : str, dict, or np.ndarray, optional
            Required when ``design`` is multi-column / continuous;
            redundant for 1-D binary ``design`` (the binary indicator
            *is* the contrast).
        statistic : str, default 'log_l2'
            Per-gene statistic. See
            :func:`~quadsv.comparators.multisample.compare_two_groups`
            for the catalog.
        null : {'analytic', 'permutation'}, default 'analytic'
            Null-distribution method. ``'analytic'`` is the analytic
            Liu-approximation null — the default on every dispatch
            path. ``'permutation'`` is available on the binary path
            only (raises on the GLM path).
        n_perm, random_state, max_exact_permutations
            Forwarded to the permutation path; ignored on the analytic
            path.
        freq_weights : np.ndarray, optional
            Per-bin reweighting (same semantics as on the standalone).
        normalize_shape : bool, default False
            If True, divide each per-(sample, gene) spectrum by its sum
            along the trailing (frequency) axis before the statistic is
            computed (delegated to the dispatch target's
            ``normalize_shape=`` keyword). Use to isolate **shape-only**
            redistribution of power across radial frequencies,
            independent of overall amplitude. Non-destructive —
            :attr:`spectra_` is unchanged after the call.
        min_samples_per_group : int, default 2
            Minimum per-group sample count required to keep a gene under
            the masked path (forwarded to
            :func:`~quadsv.comparators.multisample.compare_two_groups_masked`).
            Ignored on the unmasked / GLM paths.
        min_resid_df : int, default 1
            Minimum per-gene residual degrees of freedom required under
            the masked GLM path. Ignored on the unmasked and binary paths.

        Notes
        -----
        Use the ``normalize_shape`` keyword for a one-shot shape-only
        test that leaves :attr:`spectra_` untouched for further
        analysis. For a permanent preprocessing transform that affects
        every downstream operation on the same comparator, use the
        chainable :meth:`normalize_background` and
        :meth:`normalize_covariates` methods (no chainable equivalent
        for sum-1 normalisation — call this kwarg or the standalone
        :func:`~quadsv.comparators.normalization.normalize_shape`).
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .test_diff_freq().")
        if int(min_samples_per_group) < 2:
            raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
        if int(min_resid_df) < 1:
            raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
        groups, design_obj = _validate_design(design, len(self.samples))
        use_masked = self.presence_ is not None and not self.presence_.all()

        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if statistic != "log_l2":
                raise ValueError(
                    "Only statistic='log_l2' is supported when "
                    "contrast= is provided or `design` is a multi-column / "
                    f"continuous design, got statistic={statistic!r}."
                )
            if null != "analytic":
                raise NotImplementedError(
                    "Only null='analytic' is supported when "
                    "contrast= is provided or `design` is a multi-column / "
                    "continuous design. Pass null='analytic' or pass a 1-D "
                    "binary `design` (and omit `contrast=`) to take the "
                    "permutation path."
                )
            if contrast is None:
                raise ValueError(
                    "test_diff_freq() requires `contrast=` when `design` is "
                    "a multi-column / continuous design."
                )
            if use_masked:
                return compare_glm_masked(
                    self.spectra_,
                    design_obj,
                    contrast,
                    self.presence_,
                    gene_names=self.gene_names,
                    freq_weights=freq_weights,
                    normalize_shape=normalize_shape,
                    min_resid_df=int(min_resid_df),
                )
            return compare_glm(
                self.spectra_,
                design_obj,
                contrast,
                gene_names=self.gene_names,
                freq_weights=freq_weights,
                normalize_shape=normalize_shape,
            )

        # Binary path (1-D design, contrast is None).
        if use_masked:
            return compare_two_groups_masked(
                self.spectra_,
                groups,
                self.presence_,
                gene_names=self.gene_names,
                statistic=statistic,
                null=null,
                n_perm=n_perm,
                random_state=random_state,
                min_samples_per_group=int(min_samples_per_group),
                freq_weights=freq_weights,
                max_exact_permutations=max_exact_permutations,
                normalize_shape=normalize_shape,
            )
        return compare_two_groups(
            self.spectra_,
            groups,
            gene_names=self.gene_names,
            statistic=statistic,
            null=null,
            n_perm=n_perm,
            random_state=random_state,
            freq_weights=freq_weights,
            max_exact_permutations=max_exact_permutations,
            normalize_shape=normalize_shape,
        )

    def estimate_null_covariance(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        freq_weights: np.ndarray | None = None,
        normalize_shape: bool = False,
        min_samples_per_group: int = 2,
        min_resid_df: int = 1,
    ) -> _AnalyticNullState:
        """Estimate the analytic ``log_l2`` null covariance for inspection.

        This method follows the same binary-vs-GLM and masked-vs-unmasked
        dispatch rules as :meth:`test_diff_freq`, but returns the covariance
        diagnostics used by the analytic Liu mixture null instead of running a
        test. It is intentionally call-scoped: different designs, contrasts,
        masks, frequency weights, or ``normalize_shape`` choices imply different
        covariance estimates.

        Parameters
        ----------
        design : 1-D array, 2-D ndarray, or pandas.DataFrame
            Same design argument accepted by :meth:`test_diff_freq`.
        contrast : str, dict, or ndarray, optional
            Required for GLM designs, or to force a 1-D binary design through
            the GLM path.
        freq_weights : np.ndarray, optional
            Per-bin weights used to form ``weighted_cov = W^1/2 Sigma W^1/2``.
        normalize_shape : bool, default False
            Apply non-destructive shape normalization before estimating
            covariance, matching :meth:`test_diff_freq(normalize_shape=True)`.
        min_samples_per_group : int, default 2
            Minimum observed samples per arm for masked binary designs.
        min_resid_df : int, default 1
            Minimum per-gene residual df for masked GLM designs.

        Returns
        -------
        dict
            Analytic-null state with common keys ``mode``, ``masked``,
            ``observed``, ``sigma_log``, ``freq_weights``, ``weighted_cov``,
            ``eigenvalues``, ``effective_rank``, ``contrast_scale``,
            ``df_resid``, and ``eligible``.

            Two-group-only metadata: ``n_obs_A`` and ``n_obs_B``.
            GLM-only metadata: ``design_columns``, ``contrast_vector``,
            fitted ``beta`` coefficients, and ``n_obs``.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .estimate_null_covariance().")
        if int(min_samples_per_group) < 2:
            raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
        if int(min_resid_df) < 1:
            raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")

        groups, design_obj = _validate_design(design, len(self.samples))
        use_masked = self.presence_ is not None and not self.presence_.all()
        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if contrast is None:
                raise ValueError(
                    "estimate_null_covariance() requires `contrast=` when "
                    "`design` is a multi-column / continuous design."
                )
            if use_masked:
                return _estimate_glm_masked_null_covariance(
                    self.spectra_,
                    design_obj,
                    contrast,
                    self.presence_,
                    freq_weights=freq_weights,
                    normalize_shape=normalize_shape,
                    min_resid_df=int(min_resid_df),
                )
            return _estimate_glm_null_covariance(
                self.spectra_,
                design_obj,
                contrast,
                freq_weights=freq_weights,
                normalize_shape=normalize_shape,
            )

        if use_masked:
            return _estimate_two_group_masked_null_covariance(
                self.spectra_,
                groups,
                self.presence_,
                freq_weights=freq_weights,
                normalize_shape=normalize_shape,
                min_samples_per_group=int(min_samples_per_group),
            )
        return _estimate_two_group_null_covariance(
            self.spectra_,
            groups,
            freq_weights=freq_weights,
            normalize_shape=normalize_shape,
        )

    def test_diff_expr(
        self,
        design: Any,
        *,
        contrast: str | dict[str, float] | np.ndarray | None = None,
        log_expression: bool = False,
        eps: float = 1e-12,
    ) -> Any:
        """Differential-expression (DE) test on the DC component.

        Per-gene two-sided test on the per-sample DC scalars (the grid
        mean of each sample's per-gene expression), routed through
        :func:`~quadsv.comparators.multisample.compare_two_groups_scalar`
        for 1-D binary designs, or through
        :func:`~quadsv.comparators.multisample.compare_glm_scalar` when
        ``contrast=`` is supplied or ``design`` is a DataFrame / 2-D matrix.
        The binary path uses Welch t statistics with Welch-Satterthwaite
        t-distribution p-values; the GLM path uses OLS contrast t statistics
        with Student t-distribution p-values. The companion DF test is
        :meth:`test_diff_freq`.

        Parameters
        ----------
        design : 1-D array, pandas.Series, pandas.DataFrame, or 2-D ndarray
            Binary group labels for the two-group path, or a GLM design matrix
            when ``contrast=`` is supplied.
        contrast : str, dict, or ndarray, optional
            Linear contrast for a multi-column / continuous design. Required
            unless ``design`` is a 1-D binary vector and the two-group path is
            desired.
        log_expression : bool, default False
            If True, run the scalar test on ``log(dc + eps)`` rather than raw
            DC means. Applies to both the binary two-group path and the GLM
            path.
        eps : float, default 1e-12
            Additive offset used only when ``log_expression=True``.
        """
        if self.dc_ is None:
            raise RuntimeError("Call .compute_spectra() before .test_diff_expr().")
        groups, design_obj = _validate_design(design, len(self.samples))
        use_glm = (contrast is not None) or (groups is None)
        if use_glm:
            if contrast is None:
                raise ValueError(
                    "test_diff_expr() requires `contrast=` when `design` is "
                    "a multi-column / continuous design."
                )
            return compare_glm_scalar(
                self.dc_,
                design_obj,
                contrast,
                gene_names=self.gene_names,
                log_expression=log_expression,
                eps=eps,
            )
        return compare_two_groups_scalar(
            self.dc_,
            groups,
            gene_names=self.gene_names,
            log_expression=log_expression,
            eps=eps,
        )

    def effective_rank(self, *, weights: np.ndarray | None = None) -> np.ndarray:
        """Per-sample effective rank ``K_eff`` of the gene-spectrum covariance.

        Quantifies how concentrated the spatial-frequency content is along
        the eigen-directions of each sample's cross-gene-spectrum covariance matrix.
        ``K_eff = (Σλ)² / Σλ²`` — bounded by 1 (rank-1, all power on a
        single direction) and ``n_feature_bins`` (uniformly spread).

        Parameters
        ----------
        weights : np.ndarray, optional
            Per-bin weights (same semantics as ``freq_weights``). When
            given, returns the effective rank of
            ``W^{1/2} Σ W^{1/2}`` — useful for analysing how a
            frequency-weighted L2 statistic redistributes its power.

        Returns
        -------
        np.ndarray of shape ``(n_samples,)``.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .compute_spectra() before .effective_rank().")
        n_samples = self.spectra_.shape[0]
        return np.array(
            [_gene_pattern_diversity(self.spectra_[i], weights=weights) for i in range(n_samples)]
        )


# ---------------------------------------------------------------------------
# Shared input-validation helpers
# ---------------------------------------------------------------------------


def _validate_design(  # noqa: C901 — dispatch over three input shapes is essential
    design: Any,
    n_samples: int,
) -> tuple[np.ndarray | None, Any]:
    """Validate + normalise the unified ``design`` test-time argument.

    Accepted input forms:

    - **1-D ``np.ndarray`` / ``pd.Series``** — treated as the binary
      contrast. Must contain exactly two distinct labels. Returned
      both as a ``(n_samples,)`` array (the binary-dispatch signal)
      and wrapped in a single-column ``pd.DataFrame({"group": …})``
      so the GLM path can be taken with an explicit ``contrast=`` too.
    - **2-D ``np.ndarray``** — full design matrix of shape
      ``(n_samples, p)``. Used verbatim as the design; binary
      dispatch is disabled.
    - **``pandas.DataFrame``** — passed straight through (patsy
      encoding happens lazily inside ``compare_glm``). Binary
      dispatch disabled.

    Returns
    -------
    groups : np.ndarray | None
        ``(n_samples,)`` 1-D array iff the user supplied a 1-D binary
        input; ``None`` otherwise. Drives the binary two-group dispatch
        in :meth:`test_diff_freq` and :meth:`test_diff_expr`; ``None``
        selects the GLM path.
    design : pandas.DataFrame | np.ndarray
        Normalised design. Always a ``DataFrame`` for the 1-D and
        DataFrame inputs; a 2-D ``ndarray`` is returned as-is.

    Raises
    ------
    ValueError
        Length mismatch, 1-D input without exactly two levels, or
        2-D input with wrong row count.
    TypeError
        ``design`` is not one of the accepted forms.
    """
    if design is None:
        raise ValueError("`design` must be supplied to the test method.")
    # 1-D array-likes → binary groups path.
    arr = np.asarray(design)
    is_1d_array = arr.ndim == 1
    # pandas import is lazy so the array-only path doesn't pay for it.
    try:
        import pandas as _pd
    except ImportError:
        _pd = None
    if _pd is not None and isinstance(design, _pd.Series):
        is_1d_array = True
        arr = design.to_numpy()
    if is_1d_array:
        if arr.shape != (n_samples,):
            raise ValueError(f"1-D design length {arr.shape} does not match n_samples={n_samples}.")
        if np.unique(arr).size != 2:
            raise ValueError(
                "A 1-D design array must contain exactly two distinct labels "
                "(binary contrast). For continuous or multi-column designs, "
                "wrap in a pandas DataFrame: design=pd.DataFrame({'x': arr})."
            )
        if _pd is None:
            raise ImportError(
                "pandas is required to normalise a 1-D design into a "
                "single-column DataFrame; install it or pass a 2-D ndarray."
            )
        return arr, _pd.DataFrame({"group": arr})
    # DataFrame path.
    if _pd is not None and isinstance(design, _pd.DataFrame):
        if len(design) != n_samples:
            raise ValueError(f"design DataFrame length {len(design)} != n_samples={n_samples}.")
        return None, design
    # 2-D ndarray path.
    if isinstance(design, np.ndarray):
        if design.ndim != 2 or design.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), " f"got {design.shape}."
            )
        return None, design
    raise TypeError(
        "design must be a 1-D array/Series (binary contrast), a 2-D ndarray "
        f"(design matrix), or a pandas DataFrame; got {type(design).__name__}."
    )


def _validate_common(
    feature_mode: str,
    fft_solver: str,
    presence_threshold: float,
) -> str:
    if feature_mode not in ("radial", "2d"):
        raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
    if feature_mode == "2d" and fft_solver != "fft2":
        logger.info("feature_mode='2d' works best with fft_solver='fft2'; switching automatically.")
        fft_solver = "fft2"
    if not 0.0 <= float(presence_threshold) <= 1.0:
        raise ValueError(f"presence_threshold must be in [0, 1], got {presence_threshold}.")
    return fft_solver
