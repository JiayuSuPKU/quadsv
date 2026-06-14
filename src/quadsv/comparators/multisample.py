"""
Cross-sample spatial pattern comparison in the frequency domain.

This module implements an alignment-free, frequency-domain approach for ranking genes
by spatial-pattern difference between two groups of spatial-omics samples (e.g.,
*N* healthy vs *M* cancer slides). The key primitive is the 2D power spectrum
:math:`|\\hat{x}(k)|^2` of a rasterized gene image: power spectra are
**translation-invariant**, so samples need not be spatially registered.

Pipeline
--------

1. **Per-sample spectra** — :func:`compute_sample_spectrum` runs
   :func:`quadsv.kernels.fft.power_spectrum_2d` on each sample's ``(n_genes, ny, nx)`` array.
2. **Radial binning (default, rotation-invariant)** — :func:`radial_bin_spectrum`
   collapses the 2D spectrum onto a ``K``-dim vector indexed by normalized radial
   frequency, harmonizing samples with different ``(ny, nx)``.
3. **(Optional) 2D mode with rotation alignment** —
   :func:`align_spectra_by_rotation` rotates each sample's full 2D spectrum to
   maximize similarity to a reference, restoring comparability when directional
   anisotropy matters.
4. **Batch correction** — :func:`normalize_background` cancels per-slide
   gain/sensitivity differences; :func:`normalize_covariates` regresses out
   user-supplied covariate spectra (cell-type proportions, tissue domains, etc.);
   :func:`normalize_shape` projects per-(sample, gene) spectra onto the
   probability simplex along the frequency axis, isolating shape-only
   redistribution from amplitude differences.
5. **Cross-sample comparison per gene** — three dispatch targets share
   the same per-gene output schema:

   - :func:`compare_two_groups` (binary 1-D labels) — permutation or
     analytic Wald (Liu mixture-χ²) null;
   - :func:`compare_two_groups_masked` (binary + per-(sample, gene)
     presence mask) — same nulls, masked per-gene cohort;
   - :func:`compare_glm` (general OLS design + contrast) — Wald
     null only.

   :func:`compare_two_groups_scalar` runs the DC-component DE companion
   on per-(sample, gene) scalars (Welch t analytic or permutation).

This module only contains **array-level primitives**. The two high-level
wrapper classes that drive the pipeline on :class:`anndata.AnnData` /
:class:`spatialdata.SpatialData` containers live in
:mod:`quadsv.comparators` (:class:`~quadsv.ComparatorIrregular` /
:class:`~quadsv.ComparatorGrid`); their ``test_diff_freq(design, ...)``
method dispatches between the three comparison primitives above.

Notes
-----
The default log-L2 statistic is a quadratic form on the log-radial
spectrum: take per-group means in log-space, weight the difference
by the bin weights ``W``, and report ``T² = D' W D``. At typical
study sizes (3–10 slides per group) the exact-permutation test hits
a BH-FDR floor; the analytic Wald null (Liu's χ² mixture
approximation against a pooled within-group Σ) bypasses that floor
while remaining well-calibrated on real data — see
``scripts/comparator_benchmark`` for the calibration battery.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import ks_2samp  # noqa: F401  (exposed for downstream calibration tests)
from scipy.stats import t as _t_dist
from tqdm.auto import tqdm

from quadsv.kernels.fft import power_spectrum_2d
from quadsv.statistics import liu_sf
from quadsv.utils import apply_bh_correction

__all__ = [
    "compute_sample_spectrum",
    "radial_bin_spectrum",
    "align_spectra_by_rotation",
    "estimate_rotations_from_landmarks",
    "apply_rotations_to_spectra",
    "normalize_background",
    "normalize_covariates",
    "normalize_shape",
    "compare_two_groups",
    "compare_two_groups_masked",
    "compare_two_groups_scalar",
    "compare_glm",
]

logger = logging.getLogger(__name__)

_AVAILABLE_STATISTICS = ("log_l2", "welch_t_cauchy")


# ---------------------------------------------------------------------------
# Step 1 — per-sample spectra and radial binning
# ---------------------------------------------------------------------------


def compute_sample_spectrum(
    sample: np.ndarray,
    fft_solver: str = "rfft2",
    workers: int | None = None,
    return_dc: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectrum of every gene in a single sample.

    The spatial signal is **mean-centred per gene** before the FFT so that the
    resulting power spectrum carries only the *AC* component of the pattern —
    i.e. the ``k=0`` (DC) bin is exactly zero and low-``k`` leakage from per-
    sample mean shifts is eliminated. The separated DC scalars (the per-sample
    per-gene grid means) can be returned alongside the spectrum with
    ``return_dc=True`` and are the natural target for a *classical differential
    expression* test complementary to the spectral pattern test.

    Parameters
    ----------
    sample : np.ndarray
        Rasterized expression of shape ``(n_genes, ny, nx)``.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT routine forwarded to :func:`quadsv.kernels.fft.power_spectrum_2d`.
    workers : int, optional
        Parallel workers forwarded to :mod:`scipy.fft`.
    return_dc : bool, default False
        If True, also return a ``(n_genes,)`` array of per-gene grid means (DC
        scalars of the *uncentered* signal).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Power spectra of shape ``(n_genes, ny, n_kx)``. If ``return_dc=True``,
        also returns a ``(n_genes,)`` DC array.

    Raises
    ------
    ValueError
        If ``sample`` is not 3D.
    """
    if sample.ndim != 3:
        raise ValueError(f"sample must be 3D (n_genes, ny, nx), got shape {sample.shape}")

    # DC scalars always come from the *uncentered* grid.
    dc = sample.mean(axis=(1, 2))
    work = sample - dc[:, None, None]

    # Move feature axis to last so power_spectrum_2d treats it as M.
    moved = np.moveaxis(work, 0, -1)
    p = power_spectrum_2d(moved, fft_solver=fft_solver, workers=workers)
    spec = np.moveaxis(p, -1, 0)

    if return_dc:
        return spec, dc
    return spec


def _radial_frequency_grid(
    ny: int,
    nx: int,
    fft_solver: str,
    spacing: tuple[float, float] | None = None,
) -> np.ndarray:
    """Radial frequency for each spectrum bin, shape ``(ny, n_kx)``.

    If ``spacing=(dy, dx)`` is given, frequencies are in **cycles per unit length**
    (e.g., cycles/μm if ``spacing`` is in μm). Otherwise the result is in
    cycles/pixel with both axes normalized by their grid length, i.e.
    :math:`\\sqrt{(k_x/n_x)^2 + (k_y/n_y)^2}`.
    """
    if spacing is None:
        dy = 1.0 / ny
        dx = 1.0 / nx
        # Equivalent: scale fftfreq(..., d=1) by 1/n to get "normalized" frequency.
        ky = np.fft.fftfreq(ny) * (1.0 / dy) * dy  # == np.fft.fftfreq(ny)
        kx_full = np.fft.fftfreq(nx)
        kx_rfft = np.fft.rfftfreq(nx)
    else:
        dy, dx = spacing
        ky = np.fft.fftfreq(ny, d=dy)
        kx_full = np.fft.fftfreq(nx, d=dx)
        kx_rfft = np.fft.rfftfreq(nx, d=dx)
    if fft_solver == "fft2":
        kx = kx_full
    elif fft_solver == "rfft2":
        kx = kx_rfft
    else:
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")
    Kx, Ky = np.meshgrid(kx, ky)
    return np.sqrt(Kx**2 + Ky**2)


def radial_bin_spectrum(
    spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    n_bins: int = 30,
    fft_solver: str = "rfft2",
    exclude_dc: bool = True,
    spacing: tuple[float, float] | None = None,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """
    Bin a 2D power spectrum into ``n_bins`` radial frequency bins.

    By default the binning axis is the **normalized** radial frequency
    :math:`k = \\sqrt{(k_x/n_x)^2 + (k_y/n_y)^2} \\in [0,\\,\\sqrt{0.5}]`, so spectra
    from samples with different ``(ny, nx)`` map onto the same K bins. Passing
    ``spacing=(dy, dx)`` (in physical units, e.g. μm per cell) switches the binning
    axis to **cycles per unit length** (cycles/μm → multiply by 1000 for cycles/mm),
    so bins are directly comparable across samples with different physical
    resolutions. In that case, also pass ``edges`` to enforce a common bin grid
    across samples.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum of shape ``(..., ny, n_kx)``. Leading dims (e.g., genes,
        samples) are preserved.
    grid_shape : tuple[int, int]
        Original ``(ny, nx)`` of the rasterized image (needed because ``rfft2`` only
        stores half of the kx axis).
    n_bins : int, default 30
        Number of radial bins. Ignored when ``edges`` is supplied.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT solver used to produce ``spectrum``. Must match.
    exclude_dc : bool, default True
        If True, drop the zero-frequency (DC) bin from the output.
    spacing : tuple[float, float], optional
        Physical spacing ``(dy, dx)`` per grid cell (e.g., μm). If given, the
        binning axis is physical frequency in cycles per unit length.
    edges : np.ndarray, optional
        Explicit monotonically increasing bin edges (length ``n_bins + 1``) in the
        same frequency units as ``spacing`` (or normalized if ``spacing`` is None).
        When supplied, this overrides ``n_bins`` and gives every sample identical
        bin boundaries — required for cross-sample comparisons in physical units.

    Returns
    -------
    np.ndarray
        Radial spectra of shape ``(..., n_bins)`` (or ``n_bins - 1`` when
        ``exclude_dc=True``).

    Raises
    ------
    ValueError
        If ``spectrum``'s last two dims do not match the expected shape implied by
        ``grid_shape`` and ``fft_solver``.
    """
    ny, nx = grid_shape
    expected_kx = nx if fft_solver == "fft2" else nx // 2 + 1
    if spectrum.shape[-2:] != (ny, expected_kx):
        raise ValueError(
            f"spectrum last two dims {spectrum.shape[-2:]} do not match "
            f"expected ({ny}, {expected_kx}) for fft_solver='{fft_solver}'."
        )

    k = _radial_frequency_grid(ny, nx, fft_solver, spacing=spacing)
    k_max = float(k.max())

    if edges is None:
        # Edges include 0; right edge slightly past k_max so the last bin is closed.
        edges = np.linspace(0.0, k_max * (1.0 + 1e-9), n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be a 1D monotonically increasing array of length >= 2.")
        n_bins = len(edges) - 1
    # Bin index for each spectrum cell (0..n_bins-1).
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, n_bins - 1)

    # For rfft2 the negative-kx half is implicit but corresponds to conjugate
    # entries with identical |X|^2. To make per-bin sums match what fft2 would
    # give, double-count interior columns and single-count DC + Nyquist (if even).
    if fft_solver == "rfft2":
        col_weights = np.full(expected_kx, 2.0)
        col_weights[0] = 1.0
        if nx % 2 == 0:
            col_weights[-1] = 1.0
        weights2d = np.broadcast_to(col_weights, (ny, expected_kx)).ravel()
    else:
        weights2d = np.ones(ny * expected_kx)

    leading = spectrum.shape[:-2]
    flat = spectrum.reshape(-1, ny * expected_kx)  # (B, ny*nkx)
    out = np.zeros((flat.shape[0], n_bins))
    counts = np.zeros(n_bins)
    np.add.at(counts, idx, weights2d)
    counts[counts == 0] = 1.0  # avoid div-by-zero on empty bins
    for b in range(flat.shape[0]):
        np.add.at(out[b], idx, flat[b] * weights2d)
    out /= counts  # bin-mean power

    if exclude_dc:
        out = out[..., 1:]
    return out.reshape(*leading, out.shape[-1])


def stream_radial_features(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    *,
    chunk_size: int,
    n_bins: int,
    fft_solver: str,
    spacing: tuple[float, float] | None,
    edges: np.ndarray | None,
    pbar: Any = None,
) -> np.ndarray:
    """Stream a sample's spectrum gene-chunk-by-gene-chunk into radial features.

    Computes the radial-binned ``(n_genes, K)`` feature matrix without ever
    holding the full ``(n_genes, ny, n_kx)`` 2D spectrum: for each gene-chunk,
    ``spectrum_chunk_fn(start, stop)`` returns that chunk's 2D power spectrum
    ``(stop-start, ny, n_kx)``, which is immediately radial-binned and the dense
    2D block discarded. Peak memory is therefore ``O(chunk · ny · nx)``.

    Parameters
    ----------
    spectrum_chunk_fn : callable
        ``(start, stop) -> np.ndarray`` of shape ``(stop-start, ny, n_kx)``.
    n_genes : int
        Total genes (features) in the sample.
    grid_shape : tuple[int, int]
        ``(ny, nx)`` of the rasterized image.
    chunk_size : int
        Genes per chunk.
    n_bins, fft_solver, spacing, edges
        Forwarded to :func:`radial_bin_spectrum`.
    pbar : tqdm, optional
        Progress bar; ``.update(1)`` is called once per chunk.

    Returns
    -------
    np.ndarray
        Radial features of shape ``(n_genes, K)``.
    """
    parts: list[np.ndarray] = []
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        spec_chunk = spectrum_chunk_fn(start, stop)
        parts.append(
            radial_bin_spectrum(
                spec_chunk,
                grid_shape=grid_shape,
                n_bins=n_bins,
                fft_solver=fft_solver,
                spacing=spacing,
                edges=edges,
            )
        )
        del spec_chunk
        if pbar is not None:
            pbar.update(1)
    return np.concatenate(parts, axis=0)


def stream_geomean_landmark(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    *,
    chunk_size: int,
    gene_subset: np.ndarray | None = None,
    eps: float = 1e-12,
    pbar: Any = None,
) -> np.ndarray:
    """Stream a sample's per-(k) **geometric-mean** spectrum as a single landmark.

    Accumulates the across-gene mean of ``log(power + eps)`` one gene-chunk at a
    time, then exponentiates — yielding ``(1, ny, n_kx)``. The geomean is the
    amplitude-invariant consensus orientation template (per-gene brightness
    becomes an additive log constant that cannot move the angular argmax),
    mirroring :func:`normalize_background`'s cross-gene geometric mean. Peak
    memory is ``O(chunk · ny · nx)`` plus one ``(ny, n_kx)`` accumulator — the
    full ``(n_genes, ny, n_kx)`` stack is never held.

    Parameters
    ----------
    spectrum_chunk_fn : callable
        ``(start, stop) -> (stop-start, ny, n_kx)`` power-spectrum chunk.
    n_genes, grid_shape, chunk_size
        As in :func:`stream_radial_features`.
    gene_subset : np.ndarray, optional
        If given, the geometric mean is taken over only these gene indices
        (into ``0..n_genes-1``), chunked so the subset is never fully cached.
        Used to build a single geomean landmark from an explicit landmark-gene
        set when caching those genes' full 2D spectra would exceed budget.
        ``None`` (default) averages over all ``n_genes``.
    eps : float, default 1e-12
        Floor added before the log (keeps zeros finite).
    pbar : tqdm, optional
        Progress bar; ``.update(1)`` per chunk.

    Returns
    -------
    np.ndarray
        ``(1, ny, n_kx)`` geometric-mean landmark spectrum.
    """
    log_sum: np.ndarray | None = None
    if gene_subset is None:
        indices: Any = range(0, n_genes, chunk_size)
        count = n_genes

        def _chunk(start: int) -> np.ndarray:
            return spectrum_chunk_fn(start, min(start + chunk_size, n_genes))

    else:
        subset = np.asarray(gene_subset, dtype=int)
        count = int(subset.size)
        indices = range(0, count, chunk_size)

        def _chunk(start: int) -> np.ndarray:
            cols = subset[start : min(start + chunk_size, count)]
            # Materialise this slice of the subset gene-by-gene (each is one
            # spectrum) so the full subset is never held at once.
            return np.concatenate([spectrum_chunk_fn(int(j), int(j) + 1) for j in cols], axis=0)

    for start in indices:
        spec_chunk = _chunk(start)
        contrib = np.log(spec_chunk + eps).sum(axis=0)
        log_sum = contrib if log_sum is None else log_sum + contrib
        del spec_chunk, contrib
        if pbar is not None:
            pbar.update(1)
    if log_sum is None:
        raise ValueError("stream_geomean_landmark: must average over >= 1 gene.")
    return np.exp(log_sum / float(count))[None, ...]


def _physical_polar_coords(
    grid_shape: tuple[int, int],
    spacing: tuple[float, float] | None,
    freq_edges: np.ndarray,
    n_theta: int,
) -> tuple[np.ndarray, int]:
    """Pixel sample coords for a **physical-frequency** polar grid.

    Builds the ``(2, n_radius·n_theta)`` ``(yy, xx)`` coordinate stack into the
    fftshifted full ``(ny, nx)`` spectrum, where the radius axis samples the
    *shared physical* frequencies — the bin centres of ``freq_edges`` (cycles
    per unit length) — and ``theta`` spans the half-plane ``[0, π)`` (a real
    signal's power spectrum is centrosymmetric). Because a physical frequency
    ``f`` is converted to a pixel offset using this sample's own
    ``spacing=(dy, dx)`` and ``(ny, nx)`` (``offset_y = f·sin θ·ny·dy``),
    samples with **different lattices map the same physical frequency to the
    same radius bin** — the 2-D analogue of the radial path's shared bin grid.

    Returns ``(coords, n_radius)`` with ``coords`` of shape
    ``(2, n_radius·n_theta)`` in ``(radius, theta)`` row-major order.
    """
    ny, nx = grid_shape
    dy, dx = spacing if spacing is not None else (1.0 / ny, 1.0 / nx)
    edges = np.asarray(freq_edges, dtype=float)
    radii = 0.5 * (edges[:-1] + edges[1:])  # (n_radius,) cycles/unit
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(radii, thetas, indexing="ij")  # (n_radius, n_theta)
    cy, cx = ny // 2, nx // 2  # fftshift puts DC at floor(n/2)
    yy = cy + rr * np.sin(tt) * ny * dy
    xx = cx + rr * np.cos(tt) * nx * dx
    return np.stack([yy.ravel(), xx.ravel()], axis=0), len(radii)


def stream_polar_features(
    spectrum_chunk_fn: Any,
    n_genes: int,
    grid_shape: tuple[int, int],
    angle_deg: float,
    *,
    chunk_size: int,
    freq_edges: np.ndarray,
    spacing: tuple[float, float] | None,
    n_theta: int,
    fft_solver: str,
    pbar: Any = None,
) -> np.ndarray:
    """Stream rotate → **physical-frequency polar resample** per gene-chunk.

    The ``feature_mode='2d'`` path keeps *directional* content, so each gene's
    spectrum is rotation-aligned (not radial-collapsed) and then resampled onto
    a polar grid whose **radius axis is the shared physical-frequency grid
    (``freq_edges`` bin centres, of length ``n_radius``) and whose ``theta`` axis
    carries direction. This is a 2-D generalisation of :func:`radial_bin_spectrum`.
    Because the radius axis is in cycles/unit and shared across samples (see
    :func:`_physical_polar_coords`), different lattices are **cross-sample
    aligned**, exactly as the radial path is.

    For each gene-chunk: 2D power spectrum → rotate by ``angle_deg`` →
    :func:`_to_full_2d` (mirror the rfft half-plane) → fftshift → resample at the
    physical-frequency polar grid → flatten, discarding the dense 2D block. The
    full ``(n_genes, ny, n_kx)`` aligned stack is never held. Returns
    ``(n_genes, n_radius·n_theta)``.
    """
    coords, n_radius = _physical_polar_coords(grid_shape, spacing, freq_edges, n_theta)
    feat_len = n_radius * n_theta
    parts: list[np.ndarray] = []
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        spec_chunk = spectrum_chunk_fn(start, stop)
        if angle_deg != 0.0:
            spec_chunk = apply_rotations_to_spectra(
                [spec_chunk], [grid_shape], np.asarray([angle_deg]), fft_solver=fft_solver
            )[0]
        full = _to_full_2d(spec_chunk, grid_shape, fft_solver)  # (m, ny, nx)
        shifted = np.fft.fftshift(full, axes=(-2, -1))
        m = shifted.shape[0]
        block = np.empty((m, feat_len), dtype=np.float64)
        for j in range(m):
            block[j] = scipy.ndimage.map_coordinates(shifted[j], coords, order=1, mode="reflect")
        parts.append(block)
        del spec_chunk, full, shifted
        if pbar is not None:
            pbar.update(1)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Step 2 — optional 2D rotation alignment
# ---------------------------------------------------------------------------


def _to_full_2d(power: np.ndarray, grid_shape: tuple[int, int], fft_solver: str) -> np.ndarray:
    """Mirror an ``rfft2`` half-spectrum into a full ``(ny, nx)`` spectrum.

    Uses the Hermitian symmetry of the FFT of a real signal: ``|X[ky, kx]|² ==
    |X[(ny - ky) % ny, (nx - kx) % nx]|²``. For ``fft2`` input, returns ``power``
    unchanged.
    """
    if fft_solver == "fft2":
        return power
    ny, nx = grid_shape
    half = power.shape[-1]
    full = np.zeros(power.shape[:-1] + (nx,), dtype=power.dtype)
    full[..., :half] = power

    # Build the (-ky)-flipped version of `power` (axis -2): keep ky=0 fixed,
    # reverse the order of ky=1..ny-1.
    flipped_ky = np.empty_like(power)
    flipped_ky[..., 0, :] = power[..., 0, :]
    if ny > 1:
        flipped_ky[..., 1:, :] = power[..., :0:-1, :]

    # Mirror interior columns. Column j (1 <= j < last_interior) lives at column
    # nx - j with the ky axis reversed. Skip DC (j=0) and Nyquist (j=nx/2 when
    # nx is even) since both are self-conjugate.
    last_interior = half - 1 if nx % 2 == 0 else half
    for j in range(1, last_interior):
        full[..., nx - j] = flipped_ky[..., j]
    return full


def _polar_resample(
    spectrum_2d: np.ndarray,
    n_theta: int,
    n_radius: int,
) -> np.ndarray:
    """
    Resample a 2D spectrum (already shifted so DC is at center) onto a polar grid.

    Returns shape ``(n_theta, n_radius)``.
    """
    ny, nx = spectrum_2d.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_max = min(cy, cx)
    radii = np.linspace(1.0, r_max, n_radius)
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(radii, thetas, indexing="ij")  # (n_r, n_t)
    yy = cy + R * np.sin(T)
    xx = cx + R * np.cos(T)
    coords = np.stack([yy.ravel(), xx.ravel()], axis=0)
    sampled = scipy.ndimage.map_coordinates(spectrum_2d, coords, order=1, mode="reflect")
    return sampled.reshape(n_radius, n_theta).T  # (n_theta, n_radius)


def _build_landmark_polar_stack(
    spectra: np.ndarray,
    grid_shape: tuple[int, int],
    fft_solver: str,
    n_theta: int,
    n_radius: int,
) -> np.ndarray:
    """Build a ``(n_landmarks, n_theta, n_radius)`` polar stack for one sample.

    Each landmark's 2D spectrum is fftshifted (DC at centre), resampled onto
    the polar grid, and zero-meaned along theta so the DC angular component
    doesn't dominate the cross-correlation.
    """
    full = _to_full_2d(spectra, grid_shape, fft_solver)  # (n_landmarks, ny, nx)
    shifted = np.fft.fftshift(full, axes=(-2, -1))
    out = np.empty((shifted.shape[0], n_theta, n_radius), dtype=float)
    for j in range(shifted.shape[0]):
        polar = _polar_resample(shifted[j], n_theta, n_radius)
        out[j] = polar - polar.mean(axis=0, keepdims=True)
    return out


def estimate_rotations_from_landmarks(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = 180,
    n_radius: int = 64,
    progress: bool = False,
) -> np.ndarray:
    """
    Estimate the per-sample rotation that best aligns every landmark
    spectrum to the reference sample's corresponding landmark.

    For each non-reference sample the routine picks a single rotation angle
    that maximises the **sum over landmarks** of the per-landmark circular
    cross-correlation along the polar-angle axis — i.e. each landmark
    aligns to its same-index counterpart in the reference (not to a mean
    template). This is strictly stronger than mean-template alignment
    because it ignores cross-landmark noise (the off-diagonal ``i ≠ j``
    terms that mean-of-means picks up) and picks up anisotropy shared
    across every landmark at a common orientation.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra. Shape ``(n_landmarks, ny, n_kx)`` with
        ``(ny, n_kx)`` following ``fft_solver``. The first dimension
        (``n_landmarks``) must match across samples — landmark ``j`` in
        sample A is compared to landmark ``j`` in sample B.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``landmark_spectra`` — rfft2 spectra are expanded
        to full 2D before resampling to preserve angular content.
    reference_index : int, default 0
        Which sample's landmarks act as the rotation reference (its angle
        is fixed at 0).
    n_theta : int, default 180
        Angular resolution of the polar resampling. Recovered angles are
        accurate to ``180 / n_theta`` degrees.
    n_radius : int, default 64
        Radial resolution of the polar resampling.
    progress : bool, default False
        If True, show a tqdm bar over non-reference samples.

    Returns
    -------
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is exactly 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range or any two samples have
        inconsistent ``n_landmarks``.
    """
    n_samples = len(landmark_spectra)
    if reference_index < 0 or reference_index >= n_samples:
        raise ValueError(f"reference_index {reference_index} out of range [0, {n_samples})")

    n_landmarks = landmark_spectra[reference_index].shape[0]
    for i, s in enumerate(landmark_spectra):
        if s.shape[0] != n_landmarks:
            raise ValueError(
                f"landmark_spectra[{i}] has n_landmarks={s.shape[0]}, "
                f"expected {n_landmarks} (must match across samples)."
            )

    ref_polar = _build_landmark_polar_stack(
        landmark_spectra[reference_index],
        grid_shapes[reference_index],
        fft_solver,
        n_theta,
        n_radius,
    )
    ref_hat = np.fft.fft(ref_polar, axis=1)  # (n_landmarks, n_theta, n_radius)

    angles = np.zeros(n_samples)
    iter_samples: Any = range(n_samples)
    if progress:
        iter_samples = tqdm(iter_samples, total=n_samples, desc="Rotation estimation")
    for i in iter_samples:
        if i == reference_index:
            continue
        cur_polar = _build_landmark_polar_stack(
            landmark_spectra[i], grid_shapes[i], fft_solver, n_theta, n_radius
        )
        cur_hat = np.fft.fft(cur_polar, axis=1)
        # Per-landmark circular cross-correlation along theta; sum across
        # landmarks AND radii → best rotation common to every landmark.
        corr = np.real(np.fft.ifft(ref_hat * np.conj(cur_hat), axis=1))
        total = corr.sum(axis=(0, 2))  # (n_theta,)
        k_best = int(np.argmax(total))
        angles[i] = k_best * 180.0 / n_theta
    return angles


def apply_rotations_to_spectra(
    spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    angles_deg: np.ndarray,
    *,
    fft_solver: str = "fft2",
    progress: bool = False,
) -> list[np.ndarray]:
    """
    Rotate each sample's 2D power spectra by a per-sample angle.

    Parameters
    ----------
    spectra : sequence of np.ndarray
        Per-sample 2D power spectra — any first-axis dimension (e.g. full
        ``n_genes``). Shape ``(n, ny, n_kx)`` with ``(ny, n_kx)`` matching
        ``fft_solver``.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    angles_deg : np.ndarray
        Per-sample rotation angles in degrees (e.g. produced by
        :func:`estimate_rotations_from_landmarks`). Length must equal
        ``len(spectra)``.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``spectra``.
    progress : bool, default False
        Show a tqdm bar across samples.

    Returns
    -------
    rotated : list of np.ndarray
        Per-sample rotated spectra with the same shape as the input.

    Notes
    -----
    Rotation is done on the **2D power spectrum** (fftshifted so DC sits at
    the centre), not back on the spatial image. That is enough for any
    downstream analysis that operates on aligned spectra (radial or 2D-bin
    tests). Samples whose angle is exactly 0 are passed through as-is.
    """
    if len(angles_deg) != len(spectra):
        raise ValueError(
            f"angles_deg length {len(angles_deg)} does not match spectra length {len(spectra)}."
        )
    if len(grid_shapes) != len(spectra):
        raise ValueError(
            f"grid_shapes length {len(grid_shapes)} does not match spectra length {len(spectra)}."
        )
    out: list[np.ndarray] = []
    # strict=False: lengths are already verified above.
    iter_samples: Any = enumerate(zip(spectra, grid_shapes, strict=False))
    if progress:
        iter_samples = tqdm(iter_samples, total=len(spectra), desc="Rotation application")
    for i, (spec_i, shape) in iter_samples:
        angle_deg = float(angles_deg[i])
        if angle_deg == 0.0:
            out.append(np.asarray(spec_i).copy())
            continue
        full = _to_full_2d(spec_i, shape, fft_solver)  # (n, ny, nx)
        full_shift = np.fft.fftshift(full, axes=(-2, -1))
        rot = scipy.ndimage.rotate(
            full_shift, angle=-angle_deg, axes=(-2, -1), reshape=False, order=1, mode="reflect"
        )
        rot = np.fft.ifftshift(rot, axes=(-2, -1))
        if fft_solver == "rfft2":
            ny, nx = shape
            half = nx // 2 + 1
            rot = rot[..., :half]
        out.append(rot)
    return out


def align_spectra_by_rotation(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    target_spectra: Sequence[np.ndarray] | None = None,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = 180,
    n_radius: int = 64,
    progress: bool = False,
) -> tuple[list[np.ndarray] | None, np.ndarray]:
    """
    Two-step rotation alignment: estimate per-sample rotations from
    **landmark** spectra (whose first dimension must match across samples),
    then apply those rotations to a separate set of **target** spectra (the
    full gene panel for each sample, typically a superset of the
    landmarks).

    This is a convenience wrapper around
    :func:`estimate_rotations_from_landmarks` and
    :func:`apply_rotations_to_spectra`. Calling those directly is the
    right pattern when you want to inspect / cache the per-sample angles
    before applying them.

    Implementation
    --------------
    For every non-reference sample:

    1. Expand each landmark's 2D power spectrum to full-fft2 layout,
       fftshift so DC sits at the centre, and resample onto a polar
       ``(n_theta, n_radius)`` grid.
    2. Compute per-landmark circular cross-correlation along the
       polar-angle axis against the reference sample's same-index
       landmark. **Every landmark contributes its own cross-correlation**
       and the per-sample rotation is the angle that maximises the sum
       across landmarks (and across radii). Mean-template alignment —
       what the previous implementation did — was strictly weaker
       because the off-diagonal ``i ≠ j`` pair terms in
       ``corr(mean(a), mean(b))`` are pure noise.
    3. Rotate every entry of ``target_spectra[i]`` (if supplied) by the
       recovered angle.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra, shape ``(n_landmarks, ny, n_kx)``
        per sample. ``n_landmarks`` must match across samples.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    target_spectra : sequence of np.ndarray, optional
        Per-sample spectra to which the recovered rotations are applied.
        Any first-axis dimension (e.g. full gene panel). If ``None``, only
        the angles are returned.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of both inputs. ``fft2`` is recommended so the full
        angular content is present.
    reference_index : int, default 0
    n_theta : int, default 180
    n_radius : int, default 64
    progress : bool, default False

    Returns
    -------
    rotated : list of np.ndarray or None
        Per-sample rotated target spectra (or ``None`` when
        ``target_spectra`` is omitted).
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range, if ``landmark_spectra``
        samples disagree on ``n_landmarks``, or if
        ``target_spectra`` length does not match.
    """
    angles = estimate_rotations_from_landmarks(
        landmark_spectra,
        grid_shapes,
        fft_solver=fft_solver,
        reference_index=reference_index,
        n_theta=n_theta,
        n_radius=n_radius,
        progress=progress,
    )
    if target_spectra is None:
        return None, angles
    if len(target_spectra) != len(landmark_spectra):
        raise ValueError(
            f"target_spectra length {len(target_spectra)} does not match "
            f"landmark_spectra length {len(landmark_spectra)}."
        )
    rotated = apply_rotations_to_spectra(
        target_spectra,
        grid_shapes,
        angles,
        fft_solver=fft_solver,
        progress=progress,
    )
    return rotated, angles


# ---------------------------------------------------------------------------
# Step 3 — batch-effect correction
# ---------------------------------------------------------------------------


def normalize_background(
    spectra: np.ndarray,
    *,
    axis: int = -2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cancel per-sample multiplicative gain via cross-gene geometric-mean centring.

    For each (sample, frequency-bin) pair, every gene's power is divided
    by the geometric mean of the spectrum across the genes axis. Use
    this to correct per-sample multiplicative gain (sequencing depth,
    antibody titre, dewaxing efficiency) that scales every gene's
    spectrum at every frequency by a sample-level factor.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra :math:`P` with shape ``(..., G, K)``
        (:math:`G` along ``axis``, :math:`K` frequency bins on the
        trailing axis). Any leading dimensions (e.g., ``n_samples``)
        are broadcast over.
    axis : int, default -2
        Axis along which the cross-gene geometric mean is taken
        (the genes axis).
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added before the logarithm to keep
        zeros finite.

    Returns
    -------
    np.ndarray
        Background-normalized spectra :math:`\\tilde P`, same shape as
        ``spectra``. Never mutates the input.

    Notes
    -----
    Let :math:`P` denote the input spectrum, :math:`G` the number of
    genes (length of ``axis``), :math:`K` the number of frequency
    bins, and :math:`\\varepsilon` the ``eps`` floor. The per-bin
    geometric-mean background is

    .. math::

        b_{k} = \\exp\\!\\Bigl(
            \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr)
        \\Bigr),

    and the output is the per-bin gene-wise quotient

    .. math::

        \\tilde P_{g,k} = \\frac{P_{g,k}}{b_{k}}.

    Equivalently, in log-space this is per-bin mean centring across
    the genes axis,

    .. math::

        \\log \\tilde P_{g,k}
        = \\log\\bigl(P_{g,k} + \\varepsilon\\bigr)
          - \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr),

    so after the transform :math:`\\prod_{g} \\tilde P_{g,k} = 1` at
    every bin :math:`k` — the cross-gene geometric mean at every
    frequency is unity.

    The operation is equivalent to a per-bin OLS regression of
    :math:`\\log P_{\\cdot,k}` against a constant (the cross-gene
    mean) followed by exponentiation. With a per-sample one-hot
    covariate stacked across all (sample, gene) rows, the residuals
    match :math:`\\log \\tilde P` row-for-row, so running this
    function sample-by-sample is identical to fitting a one-hot
    sample-ID covariate in log-space and residualising.

    Companion functions:

    - :func:`normalize_covariates` removes per-bin bias linear in
      user-supplied covariate spectra (cell-type proportion maps,
      tissue domains, housekeeping templates).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalising along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(2, 5, 8))      # (n_samples, G, K)
    >>> P_tilde = normalize_background(spec)
    >>> P_tilde.shape
    (2, 5, 8)
    >>> # Cross-gene geometric mean at each (sample, bin) is unity:
    >>> bool(np.allclose(np.prod(P_tilde, axis=-2), 1.0))
    True
    """
    log_spec = np.log(spectra + eps)
    bg = log_spec.mean(axis=axis, keepdims=True)
    return np.exp(log_spec - bg)


def normalize_covariates(
    spectra: np.ndarray,
    covariate_spectra: np.ndarray,
    *,
    fit_intercept: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Residualise log-spectra against the log of covariate spectra.

    Each gene's log-spectrum is regressed (per gene, OLS in log-space)
    on the log of the supplied covariate spectra plus an optional
    intercept; the function exponentiates and returns the residual
    spectrum. Use to remove the multiplicative contribution of
    structured per-bin templates (cell-type proportion maps,
    tissue-domain indicators, housekeeping composite expression) from
    every gene's per-frequency power.

    Operating in log-space matches the multiplicative noise model of
    spectral data, keeps the output strictly positive (so the result
    composes cleanly with the downstream ``log_l2`` test), and makes
    this helper commute exactly with :func:`normalize_background`
    (orthogonal projections along orthogonal axes — see Notes).

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative gene spectra :math:`P` of shape ``(G, K)`` to
        residualise.
    covariate_spectra : np.ndarray
        Non-negative covariate spectra :math:`C` of shape
        ``(n_cov, K)``.
    fit_intercept : bool, default True
        If True, prepend a column of ones to the design matrix
        :math:`X` so per-gene log-amplitude offsets along the
        frequency axis are absorbed.
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added inside :math:`\\log(\\cdot)`
        on both ``spectra`` and ``covariate_spectra`` to keep zeros
        finite.

    Returns
    -------
    np.ndarray
        Residual spectra :math:`\\tilde P` of shape ``(G, K)``,
        strictly positive. Never mutates the input.

    Raises
    ------
    ValueError
        If ``covariate_spectra`` has a different last-axis length than
        ``spectra``.

    Notes
    -----
    Let :math:`P \\in \\mathbb{R}_{\\geq 0}^{G \\times K}` denote the
    input spectra (:math:`G` genes, :math:`K` frequency bins) and
    :math:`C \\in \\mathbb{R}_{\\geq 0}^{n_{\\mathrm{cov}} \\times K}`
    the covariate spectra. Build the log-design matrix

    .. math::

        X = \\bigl[\\, \\mathbf{1}_{K} \\;\\big|\\;
                \\log(C^{\\top} + \\varepsilon) \\,\\bigr]
            \\;\\in\\; \\mathbb{R}^{K \\times (n_{\\mathrm{cov}} + 1)},

    dropping the leading column :math:`\\mathbf{1}_{K}` when
    ``fit_intercept=False``. Fit per-gene OLS coefficients via the
    Moore-Penrose pseudoinverse :math:`X^{+}` against the log of the
    response,

    .. math::

        \\hat\\beta_{g} = X^{+}\\,
            \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top}
            \\;\\in\\; \\mathbb{R}^{n_{\\mathrm{cov}} + 1},

    and return the exponentiated residual

    .. math::

        \\tilde P_{g,k}
        = \\exp\\!\\Bigl(
            \\log( P_{g,k} + \\varepsilon )
            - X_{k,\\cdot}\\,\\hat\\beta_{g}
          \\Bigr).

    Equivalently,

    .. math::

        \\log \\tilde P_{g,\\cdot}^{\\top}
        = \\bigl( I_{K} - X X^{+} \\bigr)\\,
          \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top},

    i.e., the orthogonal projection of each gene's **log-spectrum**
    onto the orthogonal complement of the column space of :math:`X`,
    then exponentiated.

    **Commutativity with** :func:`normalize_background`. In log-space
    the two operations are left- vs right-multiplication of the
    :math:`G \\times K` log-spectrum matrix by orthogonal-projection
    matrices on disjoint axes,

    .. math::

        \\mathrm{bg}: \\;\\log P \\;\\mapsto\\;
            \\bigl( I_{G} - \\tfrac{1}{G}\\mathbf{1}_{G}\\mathbf{1}_{G}^{\\top}
            \\bigr)\\,\\log P,
        \\qquad
        \\mathrm{cov}: \\;\\log P \\;\\mapsto\\;
            \\log P \\,\\bigl( I_{K} - X X^{+} \\bigr).

    Left- and right-multiplication trivially commute, so we have the exact identity
    ``normalize_background(normalize_covariates(P)) ==
    normalize_covariates(normalize_background(P))``.

    With ``fit_intercept=True`` and **no** covariates (empty
    ``covariate_spectra``), this reduces to per-gene log-mean centring
    along the frequency axis,

    .. math::

        \\tilde P_{g,k}
        = \\frac{P_{g,k} + \\varepsilon}
                {\\exp\\!\\bigl(\\tfrac{1}{K}
                        \\sum_{k'=1}^{K}\\log(P_{g,k'} + \\varepsilon)
                  \\bigr)},

    i.e., dividing each gene's spectrum by its own cross-bin geometric
    mean — a per-gene companion to :func:`normalize_background`'s
    per-bin cross-gene operation, distinct from
    :func:`normalize_shape`'s arithmetic-mean / sum-1 normalisation.

    Companion functions:

    - :func:`normalize_background` removes per-sample multiplicative
      gain via cross-gene geometric-mean centring in log-space
      (perpendicular axis to this function).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalising along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng  = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(20, 8))      # (G, K)
    >>> cov  = rng.lognormal(size=(2, 8))       # (n_cov, K)
    >>> resid = normalize_covariates(spec, cov)
    >>> resid.shape
    (20, 8)
    >>> bool((resid > 0).all())     # log-space output is strictly positive
    True
    """
    if spectra.shape[-1] != covariate_spectra.shape[-1]:
        raise ValueError(
            f"Last axis must match: spectra has K={spectra.shape[-1]}, "
            f"covariate_spectra has K={covariate_spectra.shape[-1]}."
        )
    K = spectra.shape[-1]
    log_spec = np.log(spectra + eps)  # (G, K)
    log_cov = np.log(covariate_spectra + eps)  # (n_cov, K)
    # Design matrix shape (K, n_covariates [+1]).
    X = log_cov.T
    if fit_intercept:
        X = np.hstack([np.ones((K, 1)), X])
    # Solve OLS in log-space:  log P_g.T = X @ beta_g  →  beta_g = X^+ @ log P_g.T.
    # Closed form via pseudo-inverse (n_cov + 1 columns, small).
    pinv = np.linalg.pinv(X)
    fitted = (X @ pinv @ log_spec.T).T  # (G, K) in log-space
    return np.exp(log_spec - fitted)


def normalize_shape(
    spectra: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project each spectrum onto the probability simplex along ``axis``.

    Each fibre along ``axis`` is divided by its L1 norm, so the result
    is a proper probability distribution over the entries along that
    axis. Two fibres that differ only by a positive scalar produce
    identical outputs — only the **shape** of the power-vs-frequency
    curve survives, the overall scale is removed.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra :math:`P`. Any leading dimensions are
        preserved; normalisation acts along ``axis`` only.
    axis : int, default -1
        Axis to L1-normalise along (typically the trailing
        frequency-bin axis).
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` on the per-fibre sum to avoid
        division by zero.

    Returns
    -------
    np.ndarray
        Shape-normalized spectra :math:`\\tilde P`, same shape as
        ``spectra``, summing to 1 along ``axis``. Never mutates the
        input.

    Notes
    -----
    Let :math:`P` denote the input spectrum with :math:`K` entries
    along ``axis`` and :math:`\\varepsilon` the ``eps`` floor. The
    output is the per-fibre L1 quotient

    .. math::

        \\tilde P_{\\ldots,k}
        = \\frac{P_{\\ldots,k}}
                {\\sum_{k'=1}^{K} P_{\\ldots,k'} + \\varepsilon},

    so :math:`\\sum_{k} \\tilde P_{\\ldots,k} = 1` for every
    leading-index combination (modulo the :math:`\\varepsilon` floor;
    fibres whose total sum is below :math:`\\varepsilon` are
    effectively returned unchanged because the numerator dominates
    the floor).

    Equivalently, in log-space this is per-fibre log-sum centring,

    .. math::

        \\log \\tilde P_{\\ldots,k}
        = \\log P_{\\ldots,k}
          - \\log\\!\\Bigl( \\textstyle\\sum_{k'=1}^{K}
            P_{\\ldots,k'} + \\varepsilon \\Bigr).

    After this transform every fibre is a probability vector over
    frequency bins, so distances such as Jensen-Shannon and
    total-variation are well-defined between fibres.

    Used internally by the spectrum comparison functions
    (:func:`compare_two_groups`, :func:`compare_two_groups_masked`,
    :func:`compare_glm`) when their ``normalize_shape=True``
    keyword argument is set — the differential-frequency test then
    fires only on shape redistribution across radial bins, not on
    overall amplitude changes.

    Companion functions:

    - :func:`normalize_background` removes per-sample multiplicative
      gain via cross-gene geometric-mean centring in log-space.
    - :func:`normalize_covariates` removes per-bin bias linear in
      user-supplied covariate spectra.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    >>> P_tilde = normalize_shape(x, axis=-1)
    >>> bool(np.allclose(P_tilde.sum(axis=-1), 1.0))
    True
    >>> bool(np.allclose(P_tilde[0], P_tilde[1]))  # only the shape survives
    True
    """
    total = spectra.sum(axis=axis, keepdims=True)
    return spectra / (total + eps)


# Internal helper: applied inside the comparison functions when their
# ``normalize_shape: bool`` kwarg is True. Exists only to bridge the
# kwarg-vs-function name overlap inside the function body.
def _normalize_shape_apply(spectra: np.ndarray) -> np.ndarray:
    return normalize_shape(spectra, axis=-1)


# ---------------------------------------------------------------------------
# Step 4 — test statistics
# ---------------------------------------------------------------------------


def _resolve_freq_weights(freq_weights: np.ndarray | None, K: int) -> np.ndarray:
    """Validate / normalize frequency-bin weights; return a length-``K`` array summing to 1.

    Passing None yields uniform weights ``1/K`` — recovering the unweighted
    statistic. Any other input is cast to ``float``, required to be
    non-negative and not all-zero, and rescaled to sum-1. Non-uniform
    weights are how users express a kernel-like frequency preference (e.g.,
    low-pass polynomial vs exponential decay) inside the spectral distance.
    """
    if freq_weights is None:
        return np.full(K, 1.0 / K)
    w = np.asarray(freq_weights, dtype=float).ravel()
    if w.shape != (K,):
        raise ValueError(f"freq_weights must have length K={K}, got shape {w.shape}.")
    if np.any(w < 0):
        raise ValueError("freq_weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("freq_weights must not sum to zero.")
    return w / total


def _stat_log_l2(
    group_a: np.ndarray,
    group_b: np.ndarray,
    freq_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted L2 distance between mean log-spectra. Vectorized over genes.

    The (default) uniform-weight case reduces to the plain L2 distance on
    ``K`` frequency bins — up to an overall ``1/sqrt(K)`` scale that is
    irrelevant under a permutation null. Non-uniform weights (which must be
    non-negative and sum to 1) let the user emphasize low or high
    frequencies the same way a kernel spectrum does (polynomial vs
    exponential decay, etc.).
    """
    eps = 1e-12
    log_a = np.log(np.maximum(group_a, eps)).mean(axis=0)  # (n_genes, K)
    log_b = np.log(np.maximum(group_b, eps)).mean(axis=0)
    diff = log_a - log_b  # (n_genes, K)
    K = diff.shape[-1]
    w = _resolve_freq_weights(freq_weights, K)
    return np.sqrt(np.sum(w * diff**2, axis=-1))


def _welch_t(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Signed Welch t-statistic along axis 0.

    Works for any trailing feature shape — ``(n_samples, n_features)`` gives
    a ``(n_features,)`` result (the DE-test case), ``(n_samples, n_genes, K)``
    gives a ``(n_genes, K)`` result (the per-frequency-bin case).
    """
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1) + 1e-30)
    return (mean_a - mean_b) / se


def _welch_p_two_sided(group_a: np.ndarray, group_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic two-sided Welch t-test along axis 0.

    Returns ``(|t|, p)``; ``p`` uses the Welch–Satterthwaite degrees of
    freedom from the t-distribution tail. This is the sharp-resolution
    per-bin p-value used by the Cauchy-combined pattern test — permutation
    p-values would floor at ``1/(n_perm+1)`` per bin and drag the
    combined gene-level p to ~1e-3, killing BH-FDR power across thousands
    of genes.
    """
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se2_a = var_a / max(n_a, 1)
    se2_b = var_b / max(n_b, 1)
    se2 = se2_a + se2_b + 1e-30
    t_stat = (mean_a - mean_b) / np.sqrt(se2)
    if n_a > 1 and n_b > 1:
        df = (se2**2) / ((se2_a**2) / max(n_a - 1, 1) + (se2_b**2) / max(n_b - 1, 1) + 1e-30)
    else:
        df = np.full_like(mean_a, float(max(n_a + n_b - 2, 1)))
    df = np.maximum(df, 1.0)
    pvals = 2.0 * _t_dist.sf(np.abs(t_stat), df)
    # Clip the floor to the smallest representable positive float so
    # Cauchy's tan(pi(0.5 - p)) stays finite.
    return np.abs(t_stat), np.clip(pvals, np.finfo(float).tiny, 1.0)


def _cauchy_combine(pvals: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Cauchy combination test.

    For p-values :math:`p_1, \\dots, p_K`, forms
    :math:`T = \\frac{1}{K}\\sum_k \\tan(\\pi\\,(0.5 - p_k))` and returns
    the analytic tail probability under the standard Cauchy null,
    :math:`p = 0.5 - \\arctan(T) / \\pi`. Robust to arbitrary dependence
    between the input p-values — that is the whole point of Cauchy
    combination — so it is safe to apply over correlated frequency bins
    without decorrelating them first.

    Parameters
    ----------
    pvals : np.ndarray
        Input p-values in ``[0, 1]``. Values at the exact endpoints are
        clipped away from them to keep :math:`\\tan` finite.
    axis : int, default -1
        Axis along which to combine.

    Returns
    -------
    np.ndarray
        Combined p-value(s); one less axis than ``pvals``.
    """
    eps = np.finfo(float).eps
    clipped = np.clip(pvals, eps, 1.0 - eps)
    T = np.mean(np.tan(np.pi * (0.5 - clipped)), axis=axis)
    return 0.5 - np.arctan(T) / np.pi


_STAT_FNS = {
    "log_l2": _stat_log_l2,
}

# `welch_t_cauchy` lives outside _STAT_FNS because it returns a ``(n_genes, K)``
# per-bin array (not a per-gene scalar) and needs a bespoke runner that turns
# per-bin analytic Welch p-values into a Cauchy-combined gene-level p-value.


# ---------------------------------------------------------------------------
# Step 4b — permutation engine
# ---------------------------------------------------------------------------


def _permutation_indices(
    n_samples: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``(n_perm, n_samples)`` index arrays — random permutations of 0..n-1.

    Retained for back-compatibility; new code should prefer
    :func:`_exchangeable_group_labels`, which returns group-label matrices
    directly and supports the exact-enumeration path for small samples.
    """
    out = np.tile(np.arange(n_samples), (n_perm, 1))
    for i in range(n_perm):
        rng.shuffle(out[i])
    return out


def _exchangeable_group_labels(
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    n_perm_max: int = 10000,
) -> tuple[np.ndarray, bool]:
    """Build a null-distribution set of group relabellings.

    For small samples the total number of distinct two-group label
    assignments (``C(n, n_a)``) can be tiny compared to the user's
    requested ``n_perm``, which means the permutation p-value is floored
    at ``1/(C(n, n_a) + 1)``. In that regime an **exact** enumeration
    of every possible relabelling is both cheaper and strictly more
    accurate (zero Monte-Carlo noise, sharp p-values).

    Parameters
    ----------
    groups : np.ndarray
        Observed group labels, length ``n_samples`` with exactly two
        unique values.
    n_perm : int
        Number of random shuffles to produce when exact enumeration is
        infeasible. Ignored on the exact path.
    rng : np.random.Generator
        RNG for the sampling fallback.
    n_perm_max : int, default 10000
        If ``C(n_samples, n_a)`` is at most this, every distinct relabelling
        is enumerated (``is_exact=True``) and ``n_perm`` is overridden to
        the enumeration count. Otherwise ``n_perm`` random shuffles of
        ``groups`` are returned (``is_exact=False``).

    Returns
    -------
    perm_labels : np.ndarray
        ``(n_used, n_samples)`` int array; each row is a valid relabelling
        (same ``n_a`` / ``n_b`` marginals as ``groups``).
    is_exact : bool
        True if every row is a distinct relabelling and together they
        span every possible partition; False if the rows are independent
        random shuffles.
    """
    groups = np.asarray(groups)
    n = len(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if uniq.size != 2:
        raise ValueError(f"groups must have exactly two unique values, got {uniq}.")
    n_a = int(counts[0])
    total = int(math.comb(n, n_a))
    if total <= n_perm_max:
        perm_labels = np.empty((total, n), dtype=groups.dtype)
        a_val, b_val = uniq[0], uniq[1]
        for i, subset in enumerate(itertools.combinations(range(n), n_a)):
            perm_labels[i] = b_val
            perm_labels[i, list(subset)] = a_val
        return perm_labels, True
    perm_labels = np.empty((n_perm, n), dtype=groups.dtype)
    base = groups.copy()
    for i in range(n_perm):
        rng.shuffle(base)
        perm_labels[i] = base
    return perm_labels, False


def _permutation_pvalue(
    observed: np.ndarray,
    null_samples: np.ndarray,
) -> np.ndarray:
    """One-sided ``Pr(null >= observed)`` with an additive ``+1`` correction."""
    n_perm = null_samples.shape[0]
    ge = (null_samples >= observed[None, :]).sum(axis=0)
    return (ge + 1.0) / (n_perm + 1.0)


def _run_statistic_with_perm(
    stat_name: str,
    spectra: np.ndarray,
    groups: np.ndarray,
    perm_labels: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed statistic + null distribution for one statistic. Internal.

    ``perm_labels`` is a ``(n_perm_used, n_samples)`` matrix of group
    relabellings (as produced by :func:`_exchangeable_group_labels`).

    ``freq_weights`` is forwarded only to statistics that accept it (currently
    ``log_l2``); other statistics ignore it.
    """
    fn = _STAT_FNS[stat_name]
    uniq = np.unique(groups)
    a_val = uniq[0]
    a_mask = groups == a_val

    def _call(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if stat_name == "log_l2":
            return fn(a, b, freq_weights=freq_weights)
        return fn(a, b)

    observed = _call(spectra[a_mask], spectra[~a_mask])
    n_perm = perm_labels.shape[0]
    null = np.empty((n_perm, spectra.shape[1]))
    for p in range(n_perm):
        a = perm_labels[p] == a_val
        null[p] = _call(spectra[a], spectra[~a])
    return observed, null


def _run_welch_t_cauchy_analytic(
    spectra: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin Welch t test + Cauchy-combined gene-level p-value.

    Both the per-bin significance and the gene-level combination are
    **analytic**: per-bin p-values come from the Welch t-distribution (not
    a permutation null) and the gene-level p comes from the Cauchy
    combination, which is valid under arbitrary dependence between
    bins. This is what gives the Cauchy-Welch test
    real power versus the other (permutation-based) statistics in this
    module — permutation p-values are floored at ``1/(n_perm + 1)`` per
    bin, which would cap the combined gene-level p at ~1e-3 for typical
    ``n_perm=500`` and wipe out BH-FDR significance across thousands of
    genes.

    Returns
    -------
    observed_abs_t : np.ndarray
        ``(n_genes, K)`` observed per-bin ``|t|`` — used as the reported
        statistic summary (the max across bins sorts the output table
        sensibly, same convention as before).
    combined_pvals : np.ndarray
        ``(n_genes,)`` Cauchy-combined gene-level p-values built from per-bin
        analytic Welch p-values.
    per_bin_pvals : np.ndarray
        ``(n_genes, K)`` per-bin analytic Welch two-sided p-values.
    """
    a_mask = groups == 0
    abs_t, per_bin_pvals = _welch_p_two_sided(spectra[a_mask], spectra[~a_mask])
    combined = _cauchy_combine(per_bin_pvals, axis=-1)
    return abs_t, combined, per_bin_pvals


# ---------------------------------------------------------------------------
# Step 4b' — analytic Wald-type null for log_l2 (mixture-χ² tail via Liu)
# ---------------------------------------------------------------------------

_NULL_OPTIONS = ("permutation", "wald")


def _resolve_null(null: str) -> str:
    """Validate ``null`` against the supported set. Raises on unknown."""
    if null not in _NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {sorted(_NULL_OPTIONS)}.")
    return null


# Minimum residual df below which we issue a calibration warning for the
# Wald path. At df=1 the σ̂² estimator has a 100% relative noise (var = 2σ⁴),
# at df=2 it's 50%; both can produce occasional anti-conservative spikes.
# df ≥ 3 keeps σ̂² noise under ~67% and matches the conventional small-n
# floor used by limma / DESeq2 / edgeR.
_WALD_MIN_DF_NO_WARN = 3


def _maybe_warn_small_df_wald(df_resid: int) -> None:
    """Warn the user when running the Wald path at very small residual df.

    Suppress with ``warnings.filterwarnings('ignore',
    message='log_l2 + null=.wald.')`` if you accept the calibration risk.
    """
    if df_resid < _WALD_MIN_DF_NO_WARN:
        rel_noise_pct = 100.0 * (2.0 / max(df_resid, 1)) ** 0.5
        warnings.warn(
            f"log_l2 + null='wald' at residual df={df_resid}: "
            f"σ̂² estimator has ~{rel_noise_pct:.0f}% relative noise, "
            f"so the Wald null may be anti-conservative on a per-test basis. "
            f"For n_a + n_b ≤ 4, prefer statistic='welch_t_cauchy' for stricter "
            f"calibration (at the cost of some sensitivity).",
            UserWarning,
            stacklevel=3,
        )


def _pooled_full_within_group_sigma(
    spectra: np.ndarray, g_int: np.ndarray, *, eps: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Pooled-across-genes **full** within-group log-spectrum covariance.

    For each gene ``g`` we centre the log-spectrum at its within-group
    mean to get residuals ``R_g`` of shape ``(n, K)``, form
    ``Σ_g = R_gᵀ R_g / df`` (df = ``n_a + n_b - 2``), then average across
    all ``G`` genes. The result is a ``(K, K)`` symmetric PSD matrix.

    Why pooled and full (vs diagonal)?
    Empirically the bin-bin correlation in real spatial spectra is large
    (mean off-diag ``|r|`` between 0.5 and 0.95 across our three benchmark
    panels), so the rank of the true Σ is far below ``K``. A diagonal
    proxy spreads variance across all ``K`` directions and dramatically
    under-models the tail of the resulting weighted-χ² mixture, which
    causes the Wald null to be 4-15× anti-conservative. Pooling FULL Σ
    across genes (rather than per-gene Σ_g, which is noisy at small
    ``df``) gives a stable rank-correct estimate for the Liu integration.

    Returns
    -------
    Sigma : np.ndarray
        ``(K, K)`` pooled within-group covariance estimate.
    df : int
        Effective residual degrees of freedom ``n_a + n_b - 2``
        (returned for downstream use; ``Sigma`` is already df-normalised).
    """
    a_mask = g_int == 0
    log_a = np.log(np.maximum(spectra[a_mask], eps))  # (n_a, G, K)
    log_b = np.log(np.maximum(spectra[~a_mask], eps))
    n_a = log_a.shape[0]
    n_b = log_b.shape[0]
    res_a = log_a - log_a.mean(axis=0, keepdims=True)
    res_b = log_b - log_b.mean(axis=0, keepdims=True)
    res = np.concatenate([res_a, res_b], axis=0)  # (n, G, K)
    n, G, K = res.shape
    df = max(n_a + n_b - 2, 1)
    # Σ = (1/(G·df)) · Σ_g R_gᵀ R_g  reduces to a single matmul on the
    # flattened (n·G, K) residual matrix.
    res_flat = res.reshape(n * G, K)
    Sigma = (res_flat.T @ res_flat) / (G * df)
    return Sigma, df


def _pooled_full_within_group_sigma_masked(
    spectra: np.ndarray,
    g_int: np.ndarray,
    presence: np.ndarray,
    *,
    min_samples_per_group: int = 2,
    eps: float = 1e-12,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Mask-aware pooled-across-genes full Σ for the Wald masked path.

    For each gene ``g``, we use only the samples with ``presence[:, g] = True``,
    centre per-group, and accumulate ``R_g.T @ R_g`` into a global K×K
    accumulator. The denominator is the total residual df accumulated
    across all eligible genes — i.e. ``Σ_g (n_a_g + n_b_g - 2)``.

    Genes that do not satisfy ``min_samples_per_group`` per arm contribute
    nothing to Σ and are reported as not-tested (NaN p-values) downstream.

    The cross-bin correlation structure of the within-group log-spectrum
    is taken to be **homogeneous across genes** (same A3 assumption F1
    already makes); masking only restricts which (sample, gene) cells
    contribute to the estimator, not the structural assumption.

    Returns
    -------
    Sigma : np.ndarray
        ``(K, K)`` pooled within-group covariance estimate.
    total_df : int
        Sum of per-gene residual df across all eligible genes.
    eligible : np.ndarray
        Boolean ``(n_genes,)`` flag indicating which genes contributed
        and are testable.
    """
    n_samples, n_genes, K = spectra.shape
    a_mask = g_int == 0
    log_spectra = np.log(np.maximum(spectra, eps))
    Sigma_acc = np.zeros((K, K), dtype=np.float64)
    total_df = 0
    eligible = np.zeros(n_genes, dtype=bool)
    for g in range(n_genes):
        ai = np.where(a_mask & presence[:, g])[0]
        bi = np.where(~a_mask & presence[:, g])[0]
        if len(ai) < min_samples_per_group or len(bi) < min_samples_per_group:
            continue
        log_a = log_spectra[ai, g, :]
        log_b = log_spectra[bi, g, :]
        res_a = log_a - log_a.mean(axis=0, keepdims=True)
        res_b = log_b - log_b.mean(axis=0, keepdims=True)
        res = np.concatenate([res_a, res_b], axis=0)  # (n_g, K)
        df_g = max(len(ai) + len(bi) - 2, 1)
        Sigma_acc += res.T @ res
        total_df += df_g
        eligible[g] = True
    if total_df == 0:
        raise ValueError(
            "compare_two_groups_masked + null='wald': no genes meet "
            f"min_samples_per_group={min_samples_per_group} per arm. "
            "Cannot estimate the pooled covariance — drop the wald null "
            "or relax the minimum."
        )
    return Sigma_acc / total_df, total_df, eligible


def _log_l2_wald_pvalues(
    observed_T: np.ndarray,
    lambs: np.ndarray,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """Wald-type analytic p-values for ``log_l2`` via Liu's mixture-χ² tail.

    ``observed_T`` is the ``(n_genes,)`` per-gene statistic returned by
    :func:`_stat_log_l2` — i.e. the *square root* of the underlying
    quadratic form ``D'WD``. Squaring it here gives the H₀ statistic that
    is distributed as ``Σ_k λ_k χ²_1``, which Liu's approximation handles
    directly.
    """
    lambs_safe = np.maximum(np.asarray(lambs, dtype=float), eps)
    T2 = np.asarray(observed_T, dtype=float) ** 2
    return np.asarray(liu_sf(T2, lambs_safe), dtype=float)


def _run_log_l2_wald(
    spectra: np.ndarray,
    g_int: np.ndarray,
    freq_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed log_l2 statistic + analytic Wald p-values per gene.

    Returns ``(observed, pvals)`` both shape ``(n_genes,)``. Internally:
    1. ``observed_T = _stat_log_l2(group_a, group_b, freq_weights)``.
    2. Pooled-across-genes **full** Σ_ℓ via
       :func:`_pooled_full_within_group_sigma`.
    3. Eigenvalues of ``W½ Σ_D W½`` from a single 30×30 eigendecomposition,
       where ``Σ_D = (1/n_a + 1/n_b) · Σ_ℓ``.
    4. ``p = liu_sf(T², λ)`` for the weighted-χ² tail.
    """
    a_mask = g_int == 0
    group_a = spectra[a_mask]
    group_b = spectra[~a_mask]
    n_a = int(a_mask.sum())
    n_b = int((~a_mask).sum())
    K = spectra.shape[-1]
    w = _resolve_freq_weights(freq_weights, K)

    observed = _stat_log_l2(group_a, group_b, freq_weights=freq_weights)  # (n_genes,)
    Sigma_ell, df_resid = _pooled_full_within_group_sigma(spectra, g_int)  # (K, K), df
    _maybe_warn_small_df_wald(df_resid)
    v_c = (1.0 / max(n_a, 1)) + (1.0 / max(n_b, 1))
    sqW = np.sqrt(w)
    M = (sqW[:, None] * Sigma_ell * sqW[None, :]) * v_c  # (K, K)
    lambs = np.maximum(np.linalg.eigvalsh(M), 0.0)  # (K,)
    pvals = _log_l2_wald_pvalues(observed, lambs)
    return observed, pvals


# ---------------------------------------------------------------------------
# Step 4b'' — generalized GLM Wald path for log_l2 (design matrix + contrast)
# ---------------------------------------------------------------------------


def _build_design_matrix(
    design: pd.DataFrame | np.ndarray, n_samples: int
) -> tuple[np.ndarray, list[str]]:
    """Convert ``design`` to a ``(n_samples, p)`` numeric matrix + column labels.

    - ``np.ndarray`` of shape ``(n_samples, p)`` is accepted as-is; columns
      are labelled ``x0, x1, ...``. The caller is responsible for including
      an intercept column if desired.
    - ``pd.DataFrame``: encoded via :func:`patsy.dmatrix` with the formula
      ``~ <col1> + <col2> + ...``, which adds an intercept and one-hot
      encodes categoricals (Treatment contrast against the first level).
      If patsy is not installed, raise ``ImportError`` with an install hint.
    """
    if isinstance(design, np.ndarray):
        X = np.asarray(design, dtype=float)
        if X.ndim != 2 or X.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), " f"got {X.shape}."
            )
        return X, [f"x{i}" for i in range(X.shape[1])]

    if not isinstance(design, pd.DataFrame):
        raise TypeError(
            f"design must be a numpy ndarray or pandas DataFrame, " f"got {type(design).__name__}."
        )
    if len(design) != n_samples:
        raise ValueError(
            f"design DataFrame length {len(design)} does not match " f"n_samples={n_samples}."
        )
    try:
        import patsy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Building a design matrix from a pandas DataFrame requires patsy. "
            "Install via `pip install patsy` or pass a pre-built numpy "
            "design matrix instead."
        ) from e
    formula = "~ " + " + ".join(str(c) for c in design.columns)
    X_pat = patsy.dmatrix(formula, design, return_type="dataframe")
    return X_pat.to_numpy().astype(float), list(X_pat.columns)


def _resolve_contrast(
    contrast: str | dict[str, float] | np.ndarray, design_columns: Sequence[str]
) -> np.ndarray:
    """Map the user-supplied contrast spec to a length-``p`` numeric vector.

    - ``str``: column name in the design matrix. Auto-resolves patsy
      treatment-coded factors (e.g., ``"genotype"`` → ``"genotype[T.TG]"``)
      when there is exactly one matching column. For multi-level factors
      with >1 matching column this raises ``ValueError`` (multi-DOF
      contrasts are out of scope; pass an explicit dict or ndarray).
    - ``dict``: maps column-name → coefficient; missing columns get 0.
    - ``ndarray`` of shape ``(p,)``: used as-is.
    """
    p = len(design_columns)
    if isinstance(contrast, np.ndarray):
        c = np.asarray(contrast, dtype=float)
        if c.shape != (p,):
            raise ValueError(f"contrast ndarray must have length p={p}, got shape {c.shape}.")
        return c
    if isinstance(contrast, str):
        if contrast in design_columns:
            target = contrast
        else:
            matches = [col for col in design_columns if col.startswith(contrast + "[T.")]
            if not matches:
                raise ValueError(
                    f"Contrast '{contrast}' not found in design columns " f"{list(design_columns)}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Contrast '{contrast}' is ambiguous — matches "
                    f"{matches}. Pass an explicit dict or ndarray (multi-DOF "
                    f"contrasts are out of scope)."
                )
            target = matches[0]
        c = np.zeros(p, dtype=float)
        c[list(design_columns).index(target)] = 1.0
        return c
    if isinstance(contrast, dict):
        c = np.zeros(p, dtype=float)
        for k, v in contrast.items():
            if k not in design_columns:
                raise ValueError(
                    f"Contrast key '{k}' not in design columns " f"{list(design_columns)}."
                )
            c[list(design_columns).index(k)] = float(v)
        return c
    raise TypeError(f"Contrast must be a str, dict, or ndarray; got {type(contrast).__name__}.")


def _run_log_l2_glm_wald(
    spectra: np.ndarray,
    X: np.ndarray,
    contrast_vec: np.ndarray,
    freq_weights: np.ndarray | None,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Wald-type analytic ``log_l2`` test for a general design matrix.

    Per gene per bin we fit ``log y = X β + ε`` by OLS, take the linear
    contrast ``θ = cᵀβ``, and test ``H₀: θ = 0`` aggregated across bins
    via the same weighted-L2 quadratic form. Under H₀ the eigenvalues of
    the null distribution are
    ``λ_k = w_k · σ_k² · cᵀ(XᵀX)⁻¹c`` (diagonal because Σ is taken
    diagonal and pooled across genes), and the tail is integrated via
    Liu's mixture-χ² approximation — same machinery as the binary case
    in :func:`_run_log_l2_wald`. The two-group case literally recovers
    the binary path: ``X = [1, 1_A]``, ``c = [0, 1]``, ``v_c = 1/n_a + 1/n_b``.

    Returns
    -------
    observed : np.ndarray
        ``(n_genes,)`` per-gene statistic ``√Σ_k w_k θ_k²``.
    pvals : np.ndarray
        ``(n_genes,)`` Wald p-values via Liu.
    """
    n_samples, n_genes, K = spectra.shape
    p = X.shape[1]
    if X.shape[0] != n_samples:
        raise ValueError(f"design first dim {X.shape[0]} != n_samples {n_samples}.")
    if contrast_vec.shape != (p,):
        raise ValueError(f"contrast length {contrast_vec.shape} != design cols ({p},).")

    Y = np.log(np.maximum(spectra, eps))  # (n, G, K)
    Y_flat = Y.reshape(n_samples, n_genes * K)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta_flat = XtX_inv @ (X.T @ Y_flat)  # (p, G*K)
    res_flat = Y_flat - X @ beta_flat  # (n, G*K)
    beta = beta_flat.reshape(p, n_genes, K)
    res = res_flat.reshape(n_samples, n_genes, K)
    df_resid = max(n_samples - p, 1)
    _maybe_warn_small_df_wald(df_resid)

    theta = np.tensordot(contrast_vec, beta, axes=([0], [0]))  # (n_genes, K)

    # Pooled-across-genes FULL within-gene covariance Σ_ℓ ∈ R^{K×K}.
    # Mirrors :func:`_pooled_full_within_group_sigma` but uses GLM residuals.
    res_2d = res.reshape(n_samples * n_genes, K)
    Sigma_ell = (res_2d.T @ res_2d) / (n_genes * df_resid)
    v_c = float(contrast_vec @ XtX_inv @ contrast_vec)

    w = _resolve_freq_weights(freq_weights, K)
    T2 = (w * theta**2).sum(axis=-1)  # (n_genes,)
    T_obs = np.sqrt(T2)
    sqW = np.sqrt(w)
    M = (sqW[:, None] * Sigma_ell * sqW[None, :]) * v_c
    lambs = np.maximum(np.linalg.eigvalsh(M), 0.0)
    pvals = _log_l2_wald_pvalues(T_obs, lambs)
    return T_obs, pvals


# ---------------------------------------------------------------------------
# Step 4c — public test functions
# ---------------------------------------------------------------------------


def compare_two_groups(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "wald",
    n_perm: int = 1000,
    random_state: int | None = None,
    n_jobs: int = 1,
    freq_weights: np.ndarray | None = None,
    n_perm_max: int = 10000,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """
    Test, for every gene, whether its spatial-pattern spectrum differs between two groups.

    Parameters
    ----------
    spectra : np.ndarray
        Per-sample spectral features of shape ``(n_samples, n_genes, K)``.
    groups : np.ndarray
        Group labels of length ``n_samples`` taking exactly two distinct values
        (mapped internally to 0/1 in sorted order).
    gene_names : sequence of str, optional
        Names for the gene axis. If None, integer indices are used.
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
        Test statistic:

        - ``'log_l2'`` — (optionally weighted) L2 distance between mean
          log-spectra. Global / summary statistic. Pair with
          ``null='wald'`` for an analytic mixture-χ² null that bypasses
          the small-n permutation BH-floor; ``null='permutation'`` (default)
          falls back to label permutations with exact enumeration when
          ``C(n, n_a) ≤ n_perm_max``.
        - ``'welch_t_cauchy'`` — per-bin Welch two-sided t-test with
          **analytic** (t-distribution) p-values combined across bins
          via Cauchy combination. Analytic is the whole point:
          permutation p-values would floor at ``1/(n_perm + 1)`` per
          bin, which would also floor the gene-level combined p-value
          and destroy BH-FDR power across thousands of genes. Yields
          an extra ``P_value_per_bin`` column.
    null : {'wald', 'permutation'}, default 'wald'
        Null-distribution method. ``'wald'`` (the default) uses an analytic Wald-type test for the L2 quadratic
        form: under H₀ the statistic ``T² = D'WD`` is distributed as a
        weighted sum of χ²₁ variables whose tail is integrated via Liu's
        approximation (see :func:`quadsv.statistics.liu_sf`).
        ``'permutation'`` uses the empirical sample-label permutation
        null and is the only option that respects the
        ``n_perm`` / ``random_state`` / ``n_perm_max`` arguments.
        Currently ``null='wald'`` is only supported for
        ``statistic='log_l2'``; raises ``ValueError`` otherwise.
        ``welch_t_cauchy`` ignores this argument.

        **Sample-size guidance** (residual df = ``n_a + n_b - 2``):

        - df ≥ 4 (n_a + n_b ≥ 6): ``'wald'`` recommended — strong
          calibration + sensitivity; sweeps the top of our benchmark.
        - df ≥ 3 (n_a + n_b ≥ 5): ``'wald'`` acceptable.
        - df < 3 (n_a + n_b ≤ 4): ``'wald'`` emits a ``UserWarning``;
          σ̂² has ≥ 67% relative noise so per-test calibration may be
          anti-conservative. Prefer ``statistic='welch_t_cauchy'``
          (per-bin Welch t with proper df-corrected denominator) or
          stay with ``null='permutation'`` if the cohort allows
          enough exact relabellings.
    n_perm : int, default 1000
        Number of label permutations for the null distribution. **Ignored**
        when ``statistic='welch_t_cauchy'`` or ``null='wald'``.
    random_state : int, optional
        Seed for the permutation RNG (ignored for ``'welch_t_cauchy'``).
    n_jobs : int, default 1
        Reserved for future parallelism over genes; currently unused (the per-stat
        implementations are already vectorized over genes).
    freq_weights : np.ndarray, optional
        Only used by ``statistic='log_l2'``. Non-negative weights of length
        ``K`` (the number of frequency bins); internally renormalized to
        sum-1. Lets the user emphasize specific frequencies — e.g., a
        polynomial low-pass shape to mirror a CAR kernel, or an exponential
        high-pass shape to mirror a Gaussian kernel. ``None`` (default)
        means uniform weights.
    n_perm_max : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count. This is both faster and
        strictly more accurate than sampling in the small-sample regime
        (e.g. 6-vs-6 → 924 partitions, 5-vs-5 → 252). Above the threshold
        the test falls back to ``n_perm`` random shuffles.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (i.e., apply :func:`normalize_shape` to ``spectra`` first). Use to
        isolate shape-only / frequency-redistribution signals — the test
        then only fires when the *relative* distribution of power across
        radial frequencies varies with the contrast, independent of
        overall amplitude. Works with every valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``
        (BH-FDR), sorted by descending statistic. When
        ``statistic='welch_t_cauchy'``, the frame also carries a
        ``P_value_per_bin`` object column — each entry is an
        ``(K,)`` numpy array of per-bin permutation p-values for that gene.

    Raises
    ------
    ValueError
        If ``statistic`` is unknown, ``groups`` does not contain exactly two values,
        or shapes are inconsistent.
    """
    _available = set(_STAT_FNS) | {"welch_t_cauchy"}
    if statistic not in _available:
        raise ValueError(f"Unknown statistic '{statistic}'. Options: {sorted(_available)}.")
    null_canon = _resolve_null(null)
    # ``welch_t_cauchy`` carries its own analytic null; the ``null`` argument
    # is documented as ignored. Don't reject ``null='wald'`` (the package
    # default) for that statistic — just no-op.
    if null_canon == "wald" and statistic not in ("log_l2", "welch_t_cauchy"):
        raise ValueError(
            f"null='wald' is only supported with statistic='log_l2', "
            f"got statistic='{statistic}'."
        )
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, K), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)  # 0 = first label sorted, 1 = second

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    rng = np.random.default_rng(random_state)

    if statistic == "welch_t_cauchy":
        if freq_weights is not None:
            logger.debug("freq_weights is ignored by statistic='welch_t_cauchy'.")
        observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(spectra, g_int)
        summary_stat = observed.max(axis=-1)  # reportable scalar per gene
        if gene_names is None:
            gene_names = [str(i) for i in range(n_genes)]
        df = pd.DataFrame(
            {
                "Feature": list(gene_names),
                "Statistic": summary_stat,
                "P_value": combined_p,
                "P_value_per_bin": list(per_bin_p),
            }
        )
        df["P_adj"] = apply_bh_correction(df["P_value"])
        df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
        if n_jobs != 1:  # noqa: PLR2004
            logger.debug("n_jobs ignored: per-statistic implementations are already vectorized.")
        return df

    if null_canon == "wald":
        # Analytic Wald-type test for log_l2 (statistic check above).
        observed, pvals = _run_log_l2_wald(spectra, g_int, freq_weights)
        if gene_names is None:
            gene_names = [str(i) for i in range(n_genes)]
        df = pd.DataFrame({"Feature": list(gene_names), "Statistic": observed, "P_value": pvals})
        df["P_adj"] = apply_bh_correction(df["P_value"])
        df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
        if n_jobs != 1:  # noqa: PLR2004
            logger.debug("n_jobs ignored: per-statistic implementations are already vectorized.")
        return df

    perm_labels, is_exact = _exchangeable_group_labels(g_int, n_perm, rng, n_perm_max=n_perm_max)
    if is_exact:
        logger.info(
            "Exact permutation test: enumerated %d distinct relabellings " "(C(%d, %d)).",
            perm_labels.shape[0],
            n_samples,
            int((g_int == 0).sum()),
        )
    observed, null_dist = _run_statistic_with_perm(
        statistic, spectra, g_int, perm_labels, freq_weights=freq_weights
    )
    pvals = _permutation_pvalue(observed, null_dist)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame({"Feature": list(gene_names), "Statistic": observed, "P_value": pvals})
    df["P_adj"] = apply_bh_correction(df["P_value"])
    df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
    if n_jobs != 1:  # noqa: PLR2004
        logger.debug("n_jobs ignored: per-statistic implementations are already vectorized.")
    return df


# ---------------------------------------------------------------------------
# Step 4d — scalar (DE-style) two-group test
# ---------------------------------------------------------------------------


def compare_two_groups_masked(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "wald",
    n_perm: int = 1000,
    random_state: int | None = None,
    min_samples_per_group: int = 2,
    freq_weights: np.ndarray | None = None,
    n_perm_max: int = 10000,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """
    Per-gene two-group pattern test with **incomplete data** across samples.

    For each gene, only the subset of samples with ``presence[:, g] == True``
    contributes to the observed statistic and to the label-permutation null.
    Genes that fail to reach ``min_samples_per_group`` observations in at
    least one group are reported with ``NaN`` p-values and the number of
    observed samples per group, so the user sees why they were skipped.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, K)``.
    groups : np.ndarray
        ``(n_samples,)``, exactly two distinct labels.
    presence : np.ndarray
        ``(n_samples, n_genes)`` boolean mask. ``True`` = gene is observed
        in that sample (contributes); ``False`` = gene is absent (ignored).
    gene_names : sequence of str, optional
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
    null : {'wald', 'permutation'}, default 'wald'
        Null-distribution method. ``'wald'`` (the default) uses an analytic Wald-type test adapted for the masked
        case via a **mask-aware pooled-Σ** estimator: a single global
        ``(K, K)`` Σ is accumulated across every gene's present
        (sample, gene) cells (each gene contributes ``n_g - 2``
        residual degrees of freedom), and per-gene noncentrality
        scaling ``v_{c,g} = 1/n_a_g + 1/n_b_g`` adjusts the eigenvalues
        for that gene's specific cohort. Cross-bin correlation
        structure is taken to be homogeneous across genes (the same
        A3 assumption used in :func:`compare_two_groups` with Wald).
        Empirical calibration on synthetic missingness up to 50%
        matches the unmasked Wald path. Currently supported only with
        ``statistic='log_l2'``.

        ``'permutation'`` runs a per-gene permutation test,
        exact-enumerated when ``C(n_g, n_a_g) ≤ n_perm_max`` (most
        genes at small samples).
    n_perm : int, default 1000
    random_state : int, optional
    min_samples_per_group : int, default 2
        Minimum observed samples in each group for the gene to be tested.
    freq_weights : np.ndarray, optional
        Only consumed by ``statistic='log_l2'`` (same semantics as
        :func:`compare_two_groups`).
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (same semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals. Works with every
        valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``,
        ``n_obs_A``, ``n_obs_B``. For ``'welch_t_cauchy'`` a
        ``P_value_per_bin`` column is also included (``None`` for skipped
        genes). BH-FDR is computed only over tested genes.
    """
    _available = set(_STAT_FNS) | {"welch_t_cauchy"}
    if statistic not in _available:
        raise ValueError(f"Unknown statistic '{statistic}'. Options: {sorted(_available)}.")
    null_canon = _resolve_null(null)
    # ``welch_t_cauchy`` carries its own analytic null; the ``null`` argument
    # is documented as ignored. Don't reject ``null='wald'`` (the package
    # default) for that statistic — just no-op.
    if null_canon == "wald" and statistic not in ("log_l2", "welch_t_cauchy"):
        raise ValueError(
            f"null='wald' is only supported with statistic='log_l2', "
            f"got statistic='{statistic}'."
        )
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, K = spectra.shape
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    g_int = (groups == uniq[1]).astype(int)

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    rng = np.random.default_rng(random_state)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    # Wald masked path: precompute global pooled Σ + eigvalsh; then per-gene
    # T², v_c-scaled λ, and Liu-tail p-value. ``welch_t_cauchy`` carries its
    # own analytic null and falls through to the per-gene branch below.
    if null_canon == "wald" and statistic == "log_l2":
        Sigma_ell, total_df, _eligible = _pooled_full_within_group_sigma_masked(
            spectra, g_int, presence, min_samples_per_group=min_samples_per_group
        )
        w = _resolve_freq_weights(freq_weights, K)
        sqW = np.sqrt(w)
        M_base = sqW[:, None] * Sigma_ell * sqW[None, :]
        base_lambs = np.maximum(np.linalg.eigvalsh(M_base), 0.0)
        log_spectra = np.log(np.maximum(spectra, 1e-12))

        rows: list[dict[str, Any]] = []
        df_per_gene: list[int] = []
        for g in range(n_genes):
            mask = presence[:, g]
            ga = g_int[mask] == 0
            gb = g_int[mask] == 1
            n_a, n_b = int(ga.sum()), int(gb.sum())
            row: dict[str, Any] = {
                "Feature": gene_names[g],
                "n_obs_A": n_a,
                "n_obs_B": n_b,
                "Statistic": np.nan,
                "P_value": np.nan,
            }
            if n_a < min_samples_per_group or n_b < min_samples_per_group:
                rows.append(row)
                continue
            ai = np.where((g_int == 0) & mask)[0]
            bi = np.where((g_int == 1) & mask)[0]
            D = log_spectra[ai, g, :].mean(axis=0) - log_spectra[bi, g, :].mean(axis=0)
            T = float(np.sqrt(np.sum(w * D**2)))
            T2 = T * T
            v_c = (1.0 / n_a) + (1.0 / n_b)
            lambs = np.maximum(base_lambs * v_c, 1e-30)
            row["Statistic"] = T
            row["P_value"] = float(liu_sf(np.array([T2]), lambs)[0])
            df_per_gene.append(n_a + n_b - 2)
            rows.append(row)

        if df_per_gene:
            _maybe_warn_small_df_wald(int(np.median(df_per_gene)))

        df = pd.DataFrame(rows)
        tested = df["P_value"].notna()
        df["P_adj"] = np.nan
        if tested.any():
            df.loc[tested, "P_adj"] = apply_bh_correction(df.loc[tested, "P_value"])
        return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(
            drop=True
        )

    # Permutation / welch_t_cauchy masked path (per-gene loop).
    rows: list[dict[str, Any]] = []
    for g in range(n_genes):
        mask = presence[:, g]
        ga = g_int[mask] == 0
        gb = g_int[mask] == 1
        n_a, n_b = int(ga.sum()), int(gb.sum())
        row: dict[str, Any] = {
            "Feature": gene_names[g],
            "n_obs_A": n_a,
            "n_obs_B": n_b,
            "Statistic": np.nan,
            "P_value": np.nan,
        }
        if statistic == "welch_t_cauchy":
            row["P_value_per_bin"] = None

        if n_a < min_samples_per_group or n_b < min_samples_per_group:
            rows.append(row)
            continue

        sub = spectra[mask, g : g + 1, :]  # (n_obs, 1, K)
        sub_groups = g_int[mask]

        if statistic == "welch_t_cauchy":
            observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(sub, sub_groups)
            row["Statistic"] = float(observed.max())
            row["P_value"] = float(combined_p[0])
            row["P_value_per_bin"] = per_bin_p[0]
        else:
            # Per-gene exchange set — enumerate exactly when C(n_obs, n_a_obs)
            # is small, otherwise sample. Subsets are typically smaller than
            # the global one so the exact path kicks in more often here.
            perm_labels, _ = _exchangeable_group_labels(
                sub_groups, n_perm, rng, n_perm_max=n_perm_max
            )
            observed, null = _run_statistic_with_perm(
                statistic, sub, sub_groups, perm_labels, freq_weights=freq_weights
            )
            pval = _permutation_pvalue(observed, null)
            row["Statistic"] = float(observed[0])
            row["P_value"] = float(pval[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    # BH-correction over tested (non-NaN) genes only.
    tested = df["P_value"].notna()
    df["P_adj"] = np.nan
    if tested.any():
        df.loc[tested, "P_adj"] = apply_bh_correction(df.loc[tested, "P_value"])
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


def compare_two_groups_scalar(
    values: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    null: str = "wald",
    n_perm: int = 1000,
    random_state: int | None = None,
    n_perm_max: int = 10000,
) -> pd.DataFrame:
    """
    Per-gene two-sample test on scalar per-sample values (classical DE).

    The natural companion to :func:`compare_two_groups`: tested on the DC scalars
    (per-gene grid means) produced by :func:`compute_sample_spectrum`, the
    result is statistically independent of the spectral pattern test because
    DC and AC are orthogonal by construction (the FFT pipeline always mean-
    centres each gene's grid before computing power).

    Two null distributions are supported, chosen via ``null``:

    - ``null='wald'`` (default) — analytic two-sided Welch t-test
      p-value from the Welch-Satterthwaite t-distribution. No
      permutation BH-floor; the natural counterpart to
      :func:`compare_two_groups`'s ``null='wald'`` analytic path on the
      spectral side. The Welch t is itself a Wald-type statistic
      (point estimate / estimated SE under H₁), hence the shared
      kwarg name across the API surface.
    - ``null='permutation'`` — exact / approximate permutation null on
      ``abs(Welch t)``. More conservative at small ``n``; produces
      identical p-values as a Mann-Whitney-style rank test up to ties
      when the permutation pool is exhausted.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)`` — e.g.,
        log-normalized mean expression on each slide.
    groups : np.ndarray
        Group labels of length ``n_samples`` with exactly two distinct values.
    gene_names : sequence of str, optional
        Gene names. Integer indices if None.
    null : {'wald', 'permutation'}, default 'wald'
        Null-distribution method. ``'wald'`` returns analytic
        Welch-Satterthwaite t-distribution p-values; ``'permutation'``
        runs a label-shuffle null on ``abs(Welch t)``.
    n_perm : int, default 1000
        Number of sample-label permutations for ``null='permutation'``.
        Ignored when ``null='wald'``.
    random_state : int, optional
        Seed for the permutation RNG. Ignored when ``null='wald'``.
    n_perm_max : int, default 10000
        Cap on enumerated unique permutations.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(Welch t)``), ``Mean_diff``
        (``mean_groupA − mean_groupB``), ``P_value``, ``P_adj`` (BH-FDR), sorted
        by descending ``Statistic``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``groups`` does not contain exactly two
        distinct values, or ``null`` is unknown.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    if null not in ("wald", "permutation"):
        raise ValueError(f"null must be 'wald' or 'permutation', got {null!r}.")
    n_samples, n_genes = values.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)

    a_vals = values[g_int == 0]
    b_vals = values[g_int == 1]
    mean_diff = a_vals.mean(axis=0) - b_vals.mean(axis=0)

    if null == "wald":
        observed, pvals = _welch_p_two_sided(a_vals, b_vals)
    else:
        rng = np.random.default_rng(random_state)
        perm_labels, is_exact = _exchangeable_group_labels(
            g_int, n_perm, rng, n_perm_max=n_perm_max
        )
        if is_exact:
            logger.info(
                "Exact permutation test (DE): enumerated %d distinct relabellings.",
                perm_labels.shape[0],
            )
        observed = np.abs(_welch_t(a_vals, b_vals))
        null_dist = np.empty((perm_labels.shape[0], n_genes))
        for p in range(perm_labels.shape[0]):
            a_mask = perm_labels[p] == 0
            null_dist[p] = np.abs(_welch_t(values[a_mask], values[~a_mask]))
        pvals = _permutation_pvalue(observed, null_dist)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Mean_diff": mean_diff,
            "P_value": pvals,
        }
    )
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)


def compare_glm(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "wald",
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """Generalized two-group / continuous-covariate test via a design matrix.

    Generalises :func:`compare_two_groups` from binary group labels to an
    arbitrary GLM design matrix and a single-DOF linear contrast. The
    binary case is recovered exactly by passing
    ``design=pd.DataFrame({"group": groups})`` and ``contrast="group"``;
    p-values match :func:`compare_two_groups` to machine precision.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, K)`` spectral features (raw, not logged).
    design : pd.DataFrame or np.ndarray
        Sample-level metadata. ``DataFrame`` columns are auto-encoded via
        :mod:`patsy` (treatment-coded categoricals + intercept);
        ``ndarray`` is passed through as the design matrix verbatim
        (caller responsible for the intercept column).
    contrast : str, dict, or np.ndarray
        Linear-contrast specification:

        - ``str`` — name of a design column. Auto-resolves treatment-coded
          categoricals (e.g., ``"genotype"`` matches ``"genotype[T.TG]"``).
          Multi-DOF (3+ level factor) contrasts must be passed as an
          explicit dict or ndarray.
        - ``dict[str, float]`` — coefficient per design column.
        - ``np.ndarray`` of length ``p`` — raw contrast vector.
    gene_names : sequence of str, optional
    statistic : {'log_l2'}, default 'log_l2'
        Currently only ``log_l2`` is supported in the GLM path.
    null : {'wald'}, default 'wald'
        Only the analytic Wald-type null is supported here. Permutation
        nulls for continuous covariates are intentionally deferred (naive
        row permutation breaks the X-y joint distribution under nuisance
        covariates; correct alternatives like Freedman–Lane add complexity
        without a clear payoff over the analytic Wald). Pass
        ``null="permutation"`` only via :func:`compare_two_groups` (binary
        labels) for permutation-based tests.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the GLM is fit (same
        semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals along the design
        contrast — the test then only fires when the *relative*
        distribution of power across radial frequencies varies with the
        contrast, independent of overall amplitude.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj`` —
        same schema as :func:`compare_two_groups`.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``contrast`` cannot be resolved.
    NotImplementedError
        If ``null='permutation'`` is requested.
    """
    null_canon = _resolve_null(null)
    if null_canon != "wald":
        raise NotImplementedError(
            "compare_glm currently only supports null='wald'. Permutation "
            "null for the GLM path is intentionally deferred — use "
            "compare_two_groups (1-D binary labels) for permutation-based "
            "tests."
        )
    if statistic != "log_l2":
        raise ValueError(
            f"compare_glm currently only supports statistic='log_l2', " f"got '{statistic}'."
        )
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, K), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape

    X, design_columns = _build_design_matrix(design, n_samples)
    contrast_vec = _resolve_contrast(contrast, design_columns)

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    observed, pvals = _run_log_l2_glm_wald(spectra, X, contrast_vec, freq_weights)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame({"Feature": list(gene_names), "Statistic": observed, "P_value": pvals})
    df["P_adj"] = apply_bh_correction(df["P_value"])
    df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
