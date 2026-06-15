"""Spectral feature construction and diagnostics for comparator backends.

The comparator classes operate on per-sample 2-D power spectra before reducing
them to cross-sample features. This module owns those array-level operations:

- :func:`compute_sample_spectrum` builds mean-centered per-gene power spectra
  and optionally returns the separated DC expression means.
- :func:`radial_bin_spectrum` and :func:`stream_radial_features` collapse 2-D
  spectra to rotation-invariant radial profiles.
- :func:`estimate_rotations_from_landmarks` plus the polar streaming helpers
  keep directional information for ``feature_mode="2d"`` while aligning sample
  orientation.
- :func:`power_spectrum_anisotropy` reports a compact second-angular-moment
  diagnostic for raw 2-D spectra before radial collapse.
- :func:`effective_rank` and :func:`gene_pattern_diversity` summarize how many
  independent frequency directions contribute to observed pattern variation.

All helpers are pure array transforms. Container handling, covariate lookup,
and statistical tests live in the comparator backends and
:mod:`quadsv.comparators.multisample`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm.auto import tqdm

from quadsv.kernels.fft import power_spectrum_2d

__all__ = [
    "compute_sample_spectrum",
    "effective_rank",
    "gene_pattern_diversity",
    "power_spectrum_anisotropy",
    "radial_bin_spectrum",
    "stream_radial_features",
    "stream_geomean_landmark",
    "stream_polar_features",
    "estimate_rotations_from_landmarks",
    "apply_rotations_to_spectra",
    "align_spectra_by_rotation",
]


# ---------------------------------------------------------------------------
# Per-sample spectra and radial binning
# ---------------------------------------------------------------------------


def compute_sample_spectrum(
    sample: np.ndarray,
    fft_solver: str = "rfft2",
    workers: int | None = None,
    return_dc: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectrum of every gene in a single sample.

    The spatial signal is **mean-centered per gene** before the FFT so that the
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
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return np.sqrt(kx_grid**2 + ky_grid**2)


def _validate_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Return a validated ``(ny, nx)`` pair."""
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be a pair (ny, nx), got {grid_shape!r}.")
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError(f"grid_shape values must be positive, got {grid_shape!r}.")
    return ny, nx


def _validate_spacing(spacing: tuple[float, float]) -> tuple[float, float]:
    """Return a validated ``(dy, dx)`` pair."""
    if len(spacing) != 2:
        raise ValueError(f"spacing must be a pair (dy, dx), got {spacing!r}.")
    dy, dx = float(spacing[0]), float(spacing[1])
    if dy <= 0 or dx <= 0:
        raise ValueError(f"spacing values must be positive, got {spacing!r}.")
    return dy, dx


def _resolve_power_layout(n_kx: int, nx: int, fft_solver: str) -> tuple[str, int]:
    """Resolve a power-spectrum layout and expected ``kx`` length."""
    rfft_kx = nx // 2 + 1
    if fft_solver == "auto":
        if n_kx == nx:
            return "fft2", nx
        if n_kx == rfft_kx:
            return "rfft2", rfft_kx
        raise ValueError(
            f"power last axis {n_kx} is incompatible with grid width {nx}; "
            f"expected {nx} for fft2 or {rfft_kx} for rfft2."
        )
    if fft_solver not in {"fft2", "rfft2"}:
        raise ValueError(f"fft_solver must be 'auto', 'fft2', or 'rfft2', got {fft_solver!r}.")
    expected_kx = nx if fft_solver == "fft2" else rfft_kx
    if n_kx != expected_kx:
        raise ValueError(
            f"power last axis {n_kx} does not match {fft_solver} layout for "
            f"grid width {nx}; expected {expected_kx}."
        )
    return fft_solver, expected_kx


def _layout_frequency_grid(
    ny: int,
    nx: int,
    expected_kx: int,
    dy: float,
    dx: float,
    fft_solver: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency angle grid plus full-spectrum-equivalent layout weights."""
    ky = np.fft.fftfreq(ny, d=dy)
    if fft_solver == "fft2":
        kx = np.fft.fftfreq(nx, d=dx)
        layout_weights = np.ones((ny, expected_kx), dtype=float)
    else:
        kx = np.fft.rfftfreq(nx, d=dx)
        col_weights = np.full(expected_kx, 2.0)
        col_weights[0] = 1.0
        if nx % 2 == 0:
            col_weights[-1] = 1.0
        layout_weights = np.broadcast_to(col_weights, (ny, expected_kx))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return kx_grid, ky_grid, layout_weights


def power_spectrum_anisotropy(
    power: np.ndarray,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float] = (1.0, 1.0),
    *,
    fft_solver: str = "auto",
) -> pd.DataFrame:
    """Second angular moment of raw 2-D power spectra, excluding DC.

    Each input row is treated as one spectrum ``(ny, n_kx)``. The result
    measures how concentrated that spectrum is around a single orientation
    axis using the normalized second angular moment:
    ``|sum_k P(k) exp(2i theta_k)| / sum_k P(k)``. The value is near ``1`` for
    power concentrated along one axis and near ``0`` when power is angularly
    balanced. Angles are axial, so ``0`` and ``180`` degrees are equivalent;
    the reported ``dominant_angle_deg`` lies in ``[-90, 90]``.

    For ``rfft2`` half-plane spectra, the implicit Hermitian half-plane is
    reconstructed so the result matches the equivalent full ``fft2`` spectrum
    on both even and odd grids. Negative powers are clipped to zero before
    aggregation.

    Parameters
    ----------
    power : np.ndarray
        Raw power spectra of shape ``(n_spectra, ny, n_kx)``. ``n_kx`` may be
        ``nx`` for ``fft2`` output or ``nx // 2 + 1`` for ``rfft2`` output.
    grid_shape : tuple[int, int]
        Original image grid shape ``(ny, nx)``.
    spacing : tuple[float, float], default (1.0, 1.0)
        Grid spacing ``(dy, dx)`` used to compute physical frequency angles.
        Anisotropic spacing changes the frequency-space angle of each bin.
    fft_solver : {'auto', 'fft2', 'rfft2'}, default 'auto'
        Spectrum layout. ``'auto'`` infers the layout from ``power.shape[-1]``;
        pass an explicit value for ambiguous very small grids.

    Returns
    -------
    pandas.DataFrame
        One row per input spectrum with columns ``anisotropy``,
        ``dominant_angle_deg``, and ``total_power``.

    Raises
    ------
    ValueError
        If the input shape, grid shape, spacing, or FFT layout is invalid.
    """
    power = np.asarray(power, dtype=float)
    ny, nx = _validate_grid_shape(grid_shape)
    dy, dx = _validate_spacing(spacing)
    if power.ndim != 3 or power.shape[-2] != ny:
        raise ValueError(f"power must have shape (n_spectra, {ny}, n_kx), got {power.shape}.")

    # Resolve whether the spectrum is a full fft2 grid or an rfft2 half-plane;
    # the downstream angular grid depends on this layout.
    n_kx = int(power.shape[-1])
    resolved_solver, expected_kx = _resolve_power_layout(n_kx, nx, fft_solver)
    if resolved_solver == "rfft2":
        # Expanding avoids subtle even-grid Nyquist sign conventions. In
        # particular, doubling the stored half-plane at the stored angle is not
        # equivalent to the full fftfreq layout on the Nyquist row.
        power = _to_full_2d(power, (ny, nx), "rfft2")
        resolved_solver = "fft2"
        expected_kx = nx
    kx_grid, ky_grid, layout_weights = _layout_frequency_grid(
        ny, nx, expected_kx, dy, dx, resolved_solver
    )

    # The zero-frequency bin describes total expression, not orientation, so it
    # is excluded from the angular moment.
    radius = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    theta = np.arctan2(ky_grid, kx_grid)
    mask = radius > 0

    # Clip small numerical negatives to zero. rfft2 inputs have already been
    # expanded to full fft2 layout, so no half-plane weighting remains here.
    weights = np.maximum(power[:, mask], 0.0) * layout_weights[mask][None, :]
    total = weights.sum(axis=1)

    # The second angular moment is axial: directions separated by 180 degrees
    # are equivalent, which is why the angle enters as 2 * theta.
    cos2 = (weights * np.cos(2.0 * theta[mask])).sum(axis=1)
    sin2 = (weights * np.sin(2.0 * theta[mask])).sum(axis=1)
    anisotropy = np.divide(
        np.sqrt(cos2 * cos2 + sin2 * sin2),
        total,
        out=np.full_like(total, np.nan, dtype=float),
        where=total > 0,
    )
    angle = 0.5 * np.degrees(np.arctan2(sin2, cos2))
    angle = np.where(total > 0, angle, np.nan)
    return pd.DataFrame(
        {
            "anisotropy": anisotropy,
            "dominant_angle_deg": angle,
            "total_power": total,
        }
    )


def effective_rank(
    cov: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    r"""Effective rank (participation ratio) of a covariance matrix.

    Computes

    .. math::
        K_\mathrm{eff} \;=\; \frac{\big(\sum_k \lambda_k\big)^2}{\sum_k \lambda_k^2}

    where :math:`\lambda_k` are the eigenvalues of ``cov`` (or, when
    ``weights`` is given, of :math:`W^{1/2} \mathrm{cov}\, W^{1/2}` with
    :math:`W = \mathrm{diag}(w)`). The result is bounded by 1 (rank-1,
    all variance on a single direction) and ``K = cov.shape[0]``
    (uniformly spread, all eigenvalues equal). It coincides with the
    standard inverse Herfindahl index of the (normalised) eigenvalue
    distribution and quantifies the "effective number of independent
    components" of a quadratic-form statistic
    :math:`T^2 = X^\top \mathrm{cov} X` where :math:`X \sim \mathcal{N}(0,I)`.

    Parameters
    ----------
    cov : np.ndarray
        Symmetric ``(K, K)`` covariance matrix. Negative eigenvalues
        from numerical noise are clipped to 0.
    weights : np.ndarray, optional
        Non-negative weights of length ``K``. When provided, returns the
        effective rank of the weighted form :math:`W^{1/2} \mathrm{cov}\,W^{1/2}`,
        useful for analysing how a frequency-weighted L2 statistic
        actually distributes its sensitivity across eigen-directions.

    Returns
    -------
    float
        Effective rank in ``[1, K]``. Returns ``nan`` when the trace
        is non-positive (degenerate covariance).

    Examples
    --------
    >>> import numpy as np
    >>> effective_rank(np.eye(10))
    10.0
    >>> # Rank-1 outer product
    >>> v = np.zeros(10); v[0] = 1.0
    >>> effective_rank(np.outer(v, v))
    1.0
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square 2D matrix, got shape {cov.shape}.")
    n_bins = cov.shape[0]
    if weights is None:
        eigvals = np.linalg.eigvalsh(cov)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_bins,):
            raise ValueError(f"weights must have length n_bins={n_bins}, got shape {w.shape}.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        sqW = np.sqrt(w)
        M = (sqW[:, None] * cov) * sqW[None, :]
        eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 0.0)
    s = float(eigvals.sum())
    if s <= 0:
        return float("nan")
    return float(s * s / float((eigvals**2).sum()))


def gene_pattern_diversity(
    spectra: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> float:
    r"""Spatial-pattern diversity across genes within a single sample.

    Quantifies how heterogeneous the per-gene spatial-frequency profiles
    are within one sample. Computes the cross-gene covariance of the
    log-spectra,

    .. math::
        \hat\Sigma_\mathrm{genes} \;=\; \frac{1}{G - 1}
            \sum_g \big(\ell_g - \bar\ell\big)\big(\ell_g - \bar\ell\big)^\top

    where :math:`\ell_g \in \mathbb{R}^K` is gene :math:`g`'s
    radially-binned log-spectrum and :math:`\bar\ell` the mean across
    genes, then returns ``effective_rank(Σ_genes, weights)``.

    Interpretation:

    - **Low diversity** (:math:`K_\mathrm{eff} \approx 1`): most genes
      share the same spatial-frequency profile — the sample's spatial
      patterns collapse onto a single dominant mode (e.g. all "smooth"
      or all "punctate").
    - **High diversity** (:math:`K_\mathrm{eff} \approx K`): genes vary
      widely in their spatial structure — the sample carries a rich
      mix of spatial scales.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_genes, n_bins)`` raw spectrum matrix (typically a single sample's
        radially-binned spectrum). Log is taken internally with an ``eps``
        floor.
    weights : np.ndarray, optional
        Per-bin weights, same semantics as :func:`effective_rank`.
    eps : float, default 1e-12
        Floor for ``log(spectra)`` to handle exact-zero bins.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be (n_genes, n_bins), got shape {spectra.shape}.")
    log_s = np.log(np.maximum(spectra, eps))
    centred = log_s - log_s.mean(axis=0, keepdims=True)
    G = log_s.shape[0]
    cov = (centred.T @ centred) / max(G - 1, 1)
    return effective_rank(cov, weights=weights)


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
    from samples with different ``(ny, nx)`` map onto the same radial bin grid. Passing
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
        If True, exclude only the zero-frequency (DC) cell before bin averaging.
        The number of radial bins in the output is unchanged.
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
        Radial spectra of shape ``(..., n_bins)``.

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
    k_flat = k.ravel()
    idx = np.clip(np.digitize(k_flat, edges) - 1, 0, n_bins - 1)

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

    if exclude_dc:
        # Remove only the true zero-frequency cell; keep nonzero low frequencies
        # in the first radial interval.
        keep = k_flat > 0.0
        idx = idx[keep]
        weights2d = weights2d[keep]

    leading = spectrum.shape[:-2]
    flat = spectrum.reshape(-1, ny * expected_kx)  # (n_items, ny * n_kx)
    if exclude_dc:
        flat = flat[:, keep]
    out = np.zeros((flat.shape[0], n_bins))
    counts = np.zeros(n_bins)
    np.add.at(counts, idx, weights2d)
    counts[counts == 0] = 1.0  # avoid div-by-zero on empty bins
    for b in range(flat.shape[0]):
        np.add.at(out[b], idx, flat[b] * weights2d)
    out /= counts  # bin-mean power
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

    Computes the radial-binned ``(n_genes, n_feature_bins)`` feature matrix without ever
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
        Radial features of shape ``(n_genes, n_feature_bins)``.
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
    mirroring :func:`quadsv.comparators.normalization.normalize_background`'s
    cross-gene geometric mean. Peak
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
# Optional 2D rotation alignment
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
