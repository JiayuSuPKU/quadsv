from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.fft
import scipy.sparse as sp
from scipy.special import gamma, kv
from scipy.stats import chi2, norm

from quadsv.kernels.base import Kernel

__all__ = ["FFTKernel", "power_spectrum_2d"]


def power_spectrum_2d(
    x: np.ndarray,
    fft_solver: str = "fft2",
    workers: int | None = None,
) -> np.ndarray:
    """
    Compute the 2D power spectrum :math:`|\\hat{x}(k)|^2` of one or more grid signals.

    The result is *translation-invariant*: shifting the input image leaves the power
    spectrum unchanged. This makes the spectrum a natural alignment-free representation
    of a spatial pattern. Use :func:`quadsv.comparators.features.radial_bin_spectrum`
    to further reduce the 2D spectrum to a 1D radial-binned vector that is also
    rotation-invariant.

    Parameters
    ----------
    x : np.ndarray
        Grid signal of shape ``(ny, nx)`` for a single feature, or ``(ny, nx, M)``
        for ``M`` stacked features sharing the grid.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT routine. ``'rfft2'`` returns the half-spectrum of shape
        ``(ny, nx // 2 + 1)`` and roughly halves memory.
    workers : int, optional
        Number of parallel workers forwarded to :mod:`scipy.fft`. ``None`` uses the
        SciPy default.

    Returns
    -------
    np.ndarray
        Power spectrum. Shape ``(ny, n_kx)`` if input was 2D, or ``(ny, n_kx, M)``
        if input was 3D, where ``n_kx = nx`` for ``fft2`` and ``nx // 2 + 1`` for
        ``rfft2``. Layout matches the corresponding :mod:`scipy.fft` routine
        (zero-frequency bin at ``[0, 0]``, no fftshift applied).

    Raises
    ------
    ValueError
        If ``fft_solver`` is not one of ``'fft2'`` or ``'rfft2'``.

    Examples
    --------
    >>> img = np.random.randn(32, 32)
    >>> P = power_spectrum_2d(img, fft_solver='rfft2')
    >>> P.shape
    (32, 17)
    """
    if fft_solver not in ("fft2", "rfft2"):
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")

    squeeze = x.ndim == 2
    if squeeze:
        x = x[..., np.newaxis]

    if fft_solver == "fft2":
        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=workers)
    else:
        x_hat = scipy.fft.rfft2(x, axes=(0, 1), workers=workers)

    power = np.abs(x_hat) ** 2

    if squeeze:
        power = power[..., 0]
    return power


class FFTKernel(Kernel):
    """
    FFT-accelerated spatial kernel for dense grid data.

    Operates on evenly-spaced grid data (raster data) with spectral decomposition
    via FFT under periodic (torus) boundary conditions.

    Attributes
    ----------
    ny, nx : int
        Grid dimensions (number of rows and columns).
    n_grid : int
        Total number of grid points (``ny * nx``).
    topology : {'square', 'hex'}
        Grid topology. ``'hex'`` mirrors 10x Visium hexagonal layouts.
    method : str
        Kernel method (``'gaussian'``, ``'matern'``, ``'moran'``, ``'graph_laplacian'``,
        ``'car'``).
    params : dict
        Resolved kernel parameters (e.g. ``bandwidth``, ``nu``, ``neighbor_degree``,
        ``rho``) after defaults are merged with user overrides.
    fft_solver : {'fft2', 'rfft2'}
        FFT routine in use. ``'rfft2'`` stores roughly half the spectrum.
    n_rfft : int
        Length of the flattened spectrum: ``ny * nx`` for ``fft2`` and
        ``ny * (nx // 2 + 1)`` for ``rfft2``.
    workers : int or None
        Number of parallel workers forwarded to :mod:`scipy.fft`.
    spectrum : np.ndarray
        Flattened (row-major) eigenvalues of the kernel matrix, shape ``(n_rfft,)``.
        Eagerly computed in ``__init__``. See :meth:`eigenvalues` for a sorted /
        full-FFT-layout accessor.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(  # noqa: C901
        self,
        shape: tuple[int, int],
        spacing: tuple[float, float] = (1.0, 1.0),
        topology: str = "square",
        method: str = "matern",
        workers: int | None = None,
        fft_solver: str = "fft2",
        *,
        centering: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize FFT-accelerated spatial kernel for grid data.

        Parameters
        ----------
        shape : tuple of int
            Grid dimensions (ny, nx).
        spacing : tuple of float, default (1.0, 1.0)
            Physical distance between pixels (dy, dx).
        topology : {'square', 'hex'}, default 'square'
            Grid topology. 'hex' is for Visium-like hexagonal layouts.
        method : str, default 'matern'
            Kernel method: 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
        workers : Optional[int], default None
            Number of parallel workers for fft computations.
        fft_solver : {'fft2', 'rfft2'}, default 'fft2'
            FFT solver to use. 'fft2' (full FFT) or 'rfft2' (real FFT, ~50% memory).
            Default is 'fft2' for better compatibility and robustness on most architectures.
        **kwargs : dict
            Kernel parameters (bandwidth, nu, neighbor_degree, rho).

        Examples
        --------
        >>> kernel = FFTKernel((64, 64), method='gaussian', bandwidth=2.0)
        >>> kernel = FFTKernel((64, 64), topology='hex', method='matern')
        """
        super().__init__(centering=centering)
        ny, nx = shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Grid dimensions must be >= 2, got ({ny}, {nx})")
        self.ny: int = ny
        """Number of grid rows."""
        self.nx: int = nx
        """Number of grid columns."""
        self._dy, self._dx = spacing
        self.n_grid: int = self.ny * self.nx
        """Total number of grid points (``ny * nx``)."""
        self.n: int = self.n_grid
        """Alias for ``n_grid`` to satisfy the :class:`~quadsv.kernels.Kernel` interface."""

        # FFT solver selection
        if fft_solver not in ("fft2", "rfft2"):
            raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")
        self.fft_solver: str = fft_solver
        """FFT routine in use (``'fft2'`` or ``'rfft2'``)."""
        self.n_rfft: int = (
            self.ny * self.nx if fft_solver == "fft2" else self.ny * (self.nx // 2 + 1)
        )
        """Length of the flattened spectrum buffer (``ny*nx`` for ``fft2``, ``ny*(nx//2+1)`` for ``rfft2``)."""

        # Sanity Checks
        if topology not in ("square", "hex"):
            raise ValueError(f"topology must be 'square' or 'hex', got '{topology}'")
        if method not in self._available_kernels:
            raise ValueError(f"method must be one of {self._available_kernels}, got '{method}'")

        self.topology: str = topology
        """Grid topology (``'square'`` or ``'hex'``)."""
        self.method: str = method
        """Kernel method name."""

        # Update kernel parameters from defaults. Graph kernels accept an
        # additional ``k_neighbors`` as a convenience (k-NN semantic): it's
        # converted to the closest ``neighbor_degree`` (FFT-ring semantic)
        # based on the grid topology — see _k_neighbors_to_degree below.
        params = self._get_default_params(method).copy()
        k_neighbors_user = None
        if kwargs:
            for key, value in kwargs.items():
                if key == "k_neighbors" and method in ("moran", "graph_laplacian", "car"):
                    k_neighbors_user = value
                elif key in params:
                    params[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        self.params: dict = params
        """Resolved kernel parameters after defaults are merged with user overrides."""
        self.workers: int | None = workers
        """Number of parallel workers forwarded to :mod:`scipy.fft`, or ``None`` for the library default."""

        # 1. Precompute Distances
        # For Periodic: Distances wrap around (min(d, L-d)).
        if self.topology == "hex":
            self._min_dist_sq = self._precompute_hex_torus()
        else:
            self._min_dist_sq = self._precompute_square_dists()

        # Resolve k_neighbors → neighbor_degree now that the distance grid is built.
        if k_neighbors_user is not None:
            if "neighbor_degree" in kwargs:
                raise ValueError(
                    "Pass either 'k_neighbors' (k-NN semantic) or "
                    "'neighbor_degree' (FFT-ring semantic), not both."
                )
            self.params["neighbor_degree"] = self._k_neighbors_to_degree(k_neighbors_user)

        # 2. Precompute Kernel spectrum
        self.spectrum: np.ndarray = self._compute_eigenvalues()
        """Flattened (row-major) eigenvalues of the kernel matrix, shape ``(n_rfft,)``."""

    def _unique_ring_distances(self) -> np.ndarray:
        """Tolerance-grouped unique squared distances on the grid.

        Groups numerically-close values into a single "ring" so hex and other
        irrational-coordinate topologies report physical shells consistently.
        Returns sorted ascending, starting with 0 (self). Internal helper.
        """
        flat = np.sort(self._min_dist_sq.ravel())
        tol = 1e-6 * max(1.0, float(flat[-1]))
        # Take the first element of each tolerance-gap-separated group.
        diffs = np.diff(flat)
        keep = np.concatenate([[True], diffs > tol])
        return flat[keep]

    def _k_neighbors_to_degree(self, k_neighbors: int) -> int:
        """Translate a k-NN-style ``k_neighbors`` into an FFT-ring ``neighbor_degree``.

        Returns the smallest ``neighbor_degree`` whose cumulative count of
        grid cells (excluding self) is ≥ ``k_neighbors``. Topology-aware via
        :meth:`_unique_ring_distances`:

        - Square: k=4 → 1 (N/S/E/W), k=8 → 2 (+diagonals), k=12 → 3.
        - Hex:    k=6 → 1, k=12 → 2, k=18 → 3.
        """
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be ≥ 1, got {k_neighbors}")
        unique_dists = self._unique_ring_distances()
        tol = 1e-6 * max(1.0, float(unique_dists[-1]))
        for deg_order in range(1, len(unique_dists)):
            cutoff_sq = unique_dists[deg_order]
            count = int((self._min_dist_sq <= cutoff_sq + tol).sum()) - 1
            if count >= k_neighbors:
                return deg_order
        return len(unique_dists) - 1

    def _format_params(self) -> str:
        """Format kernel params safely without dumping large arrays/matrices."""
        if not self.params:
            return "None"
        parts = []
        for k, v in self.params.items():
            try:
                if isinstance(v, np.ndarray):
                    parts.append(f"{k}=array(shape={v.shape}, dtype={v.dtype})")
                elif sp.issparse(v):
                    parts.append(f"{k}=sparse(shape={v.shape}, nnz={v.nnz})")
                else:
                    parts.append(f"{k}={v}")
            except Exception:
                parts.append(f"{k}=?")
        return ", ".join(parts)

    def __repr__(self) -> str:
        """
        Return a detailed, machine-readable representation of the FFTKernel.

        Returns
        -------
        str
            String representation in angle-bracket format.
        """
        spectrum_info = (
            f"spectrum shape={self.spectrum.shape}"
            if self.spectrum is not None
            else "spectrum=None"
        )
        return (
            f"<FFTKernel method={self.method} shape=({self.ny}, {self.nx}) topology={self.topology} "
            f"fft_solver={self.fft_solver} {spectrum_info} params={{ {self._format_params()} }}>"
        )

    def __str__(self) -> str:
        """
        Return a human-friendly, multi-line representation of the FFTKernel.

        Returns
        -------
        str
            Multi-line string summary.
        """
        lines = [
            "FFTKernel",
            f"- Method: {self.method}",
            f"- Grid shape: ({self.ny}, {self.nx})",
            f"- Topology: {self.topology}",
            f"- Spacing: ({self._dy}, {self._dx})",
            f"- FFT Solver: {self.fft_solver}",
        ]

        if self.spectrum is not None:
            lines.append(
                f"- Spectrum: shape={self.spectrum.shape}, min={np.min(self.spectrum):.4g}, max={np.max(self.spectrum):.4g}"
            )
        else:
            lines.append("- Spectrum: None")

        lines.append(f"- Params: {self._format_params()}")
        return "\n".join(lines)

    def _get_default_params(self, method: str) -> dict[str, Any]:
        """
        Returns default parameters for specific kernel methods.

        Parameters
        ----------
        method : str
            Kernel method name. Should be one of _available_kernels.

        Returns
        -------
        dict[str, Any]
            Method defaults: bandwidth (gaussian/matern), nu (matern), neighbor_degree (moran/graph_laplacian/car), rho (car).
        """
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"nu": 1.5, "bandwidth": 2.0},
            "moran": {"neighbor_degree": 1},
            "graph_laplacian": {"neighbor_degree": 1},
            "car": {"rho": 0.9, "neighbor_degree": 1},
        }
        return method_defaults.get(method, {})

    def _precompute_square_dists(self):
        """Computes wrap-around torus distances from (0,0) to (y,x)."""
        y = np.arange(self.ny) * self._dy
        x = np.arange(self.nx) * self._dx

        # Wrap-around distance for periodic boundaries
        y = np.minimum(y, (self.ny * self._dy) - y)
        x = np.minimum(x, (self.nx * self._dx) - x)

        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy**2 + xx**2

    def _precompute_hex_torus(self):
        """Squared torus distances on a hexagonal grid (Visium convention).

        Spot ``(r, c)`` lies at physical ``(y, x) = (r * sqrt(3)/2, c + 0.5 * (r%2))``
        in units of the center-to-center horizontal step — i.e., odd rows are shifted
        half a step in +x, matching the 10x Visium ``array_row`` / ``array_col``
        layout.

        Returns an ``(ny, nx)`` array consistent with the ``(ny, nx)`` signal shape
        expected by :meth:`xtKx`. (The previous implementation returned ``(nx, ny)``
        and scrambled the spectrum for non-square grids, silently breaking anisotropic
        signals on any real Visium slide — Visium is never square. Tests covered only
        square hex grids so the bug was invisible. Fixed.)

        Periodicity in the y direction is well-defined only when ``ny`` is even (so
        the row-parity shift is preserved under wrap-around); callers feeding odd
        ``ny`` will get a near-periodic but slightly off torus and a warning is
        emitted.
        """
        if self.ny % 2 != 0:
            warnings.warn(
                f"Hex topology expects an even number of rows (ny); got ny={self.ny}. "
                "Periodic boundary conditions are approximate for odd ny.",
                UserWarning,
                stacklevel=2,
            )
        r = np.arange(self.ny)  # row index, first (ny) axis
        c = np.arange(self.nx)  # col index, second (nx) axis
        rr, cc = np.meshgrid(r, c, indexing="ij")  # both shape (ny, nx)

        y_phys = rr * (np.sqrt(3) / 2.0)
        x_phys = cc + 0.5 * (rr % 2)
        coords_grid = np.stack([y_phys, x_phys], axis=-1)  # (ny, nx, 2)

        # Torus periods: width in x is nx, height in y is ny * sqrt(3)/2.
        P_y = np.array([self.ny * (np.sqrt(3) / 2.0), 0.0])
        P_x = np.array([0.0, float(self.nx)])

        min_d2 = np.full((self.ny, self.nx), np.inf)
        for k in (-1, 0, 1):
            for m in (-1, 0, 1):
                shift = k * P_y + m * P_x
                shifted = coords_grid + shift.reshape(1, 1, 2)
                d2 = np.sum(shifted**2, axis=-1)
                min_d2 = np.minimum(min_d2, d2)
        return min_d2

    def _compute_eigenvalues(self):  # noqa: C901
        """Spectral decomposition of the kernel using fft2 or rfft2.

        Returns eigenvalues in the selected FFT layout.
        """

        # --- Continuous Kernels ---
        if self.method == "gaussian":
            bw = self.params["bandwidth"]
            K_img = np.exp(-0.5 * (self._min_dist_sq / bw**2))
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        elif self.method == "matern":
            bw = self.params["bandwidth"]
            nu = self.params["nu"]
            d = np.sqrt(self._min_dist_sq)
            mask_zero = d == 0
            d[mask_zero] = 1.0  # dummy value, overwritten below
            factor = (np.sqrt(2 * nu) * d) / bw
            K_img = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
            K_img[mask_zero] = 1.0  # correct limit: K(x, x) = 1
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        # --- Graph-based Kernels (Moran / Graph Laplacian / CAR) ---
        elif self.method in ["moran", "graph_laplacian", "car"]:
            degree_order = self.params["neighbor_degree"]

            # Tolerance-based ring grouping: hex distances like 1.0 arise from
            # sqrt(3)/2 products and split into several numerical clusters
            # (e.g. 0.9999998 and 1.0000003) that are *physically the same
            # shell*. Without this, degree_order=1 on hex returns only 2 cells
            # instead of the full 6-neighbour ring.
            unique_dists = self._unique_ring_distances()

            if degree_order < len(unique_dists):
                cutoff_sq = unique_dists[degree_order]
            else:
                cutoff_sq = unique_dists[-1]

            # Inclusive cutoff with tolerance so we catch the whole ring.
            tol = 1e-6 * max(1.0, float(unique_dists[-1]))
            W_img = (self._min_dist_sq <= cutoff_sq + tol).astype(float)
            W_img[0, 0] = 0.0

            # Row-Normalization Factor
            # For Periodic: Exact constant degree.
            degree = np.sum(W_img)

            if degree == 0:
                return np.ones(self.n_grid)

            # Compute Spectrum of Normalized W
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(W_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(W_img, workers=self.workers)
            lam_W = np.real(spectrum_2d).ravel() / degree

            if self.method == "moran":
                return lam_W
            elif self.method == "graph_laplacian":
                return 1.0 - lam_W
            elif self.method == "car":
                rho = self.params["rho"]
                # Cap rho to prevent singularity if rho is too close to 1
                if rho >= 1.0:
                    warnings.warn(
                        f"rho={rho} >= 1.0 causes singularity in CAR kernel; clamping to 0.99",
                        UserWarning,
                        stacklevel=2,
                    )
                    rho = 0.99
                return 1.0 / (1.0 - rho * lam_W)

        else:
            raise ValueError("Unknown method")

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """
        Compute the quadratic form x^T K x efficiently using FFT.

        Uses Parseval's theorem to compute the result in frequency domain
        for O(n log n) complexity instead of O(n²).

        Parameters
        ----------
        x : np.ndarray
            Input data tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.

        Returns
        -------
        float or np.ndarray
            Quadratic form value(s). Scalar if input was 2D, shape (M,) if input was 3D.
        """
        if x.ndim == 2:
            x = x[..., np.newaxis]

        ny, nx, M = x.shape

        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Data shape ({ny}, {nx}) does not match kernel ({self.ny}, {self.nx})"
            )

        # HKH quadratic form on z-scored input equals raw K on the centered
        # input; subtracting per-feature mean is cheap on the grid.
        if self.centering:
            x = x - x.mean(axis=(0, 1), keepdims=True)

        # Transform using selected FFT solver via the shared power-spectrum helper.
        x_power = power_spectrum_2d(x, fft_solver=self.fft_solver, workers=self.workers)

        if self.fft_solver == "fft2":
            # Reshape spectrum for full fft2: (ny, nx, 1)
            lam = self.spectrum.reshape(self.ny, self.nx, 1)

            # Weighted Sum (Parseval's Theorem)
            weighted_power = np.sum(x_power * lam, axis=(0, 1))

        else:
            # Reshape spectrum for rfft2: (ny, nx//2+1, 1)
            lam = self.spectrum.reshape(self.ny, self.nx // 2 + 1, 1)

            # Weighted Sum (Parseval's Theorem) with correction for rfft2
            weighted = x_power * lam
            weighted_power = 2.0 * np.sum(weighted, axis=(0, 1))

            # Correction: Subtract the first column (fx=0) once
            # because we added it twice in the line above, but it only exists once.
            weighted_power -= np.sum(weighted[:, 0, :], axis=0)

            # Correction: If width is even, the last column is Nyquist (fx=N/2).
            # It is also unique (real-valued in full spectrum), so subtract it once.
            if nx % 2 == 0:
                weighted_power -= np.sum(weighted[:, -1, :], axis=0)

        # FFT is unnormalized: Parseval requires 1/n normalization
        Q = weighted_power / (ny * nx)

        # Unwrap if M=1
        return Q.item() if M == 1 else Q.ravel()

    def _ones_stats(self) -> tuple[float, float]:
        """Return ``(s1, s2) = (𝟏ᵀ K 𝟏, ‖K·𝟏‖²)`` analytically from the DC mode.

        On a torus the constant vector ``𝟏`` is the ``k = (0, 0)`` Fourier
        mode, so ``K · 𝟏 = λ₀ · 𝟏`` where ``λ₀ = spectrum[0]`` (DC).
        ``s1 = λ₀ · n_grid``, ``s2 = λ₀² · n_grid``. Zero FFT work.
        """
        lam0 = float(self.spectrum.ravel()[0])
        return lam0 * self.n_grid, (lam0**2) * self.n_grid

    def Kx(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the kernel operator to ``x`` via FFT in O(n log n).

        Implemented as ``K x = F^{-1}(λ · F(x))`` where ``λ`` is the full
        eigenvalue spectrum on the torus. The result is returned on the
        spatial grid (not the feature-axis first layout used by
        :class:`NUFFTKernel`); callers that want a quadratic / bilinear form
        should prefer :meth:`xtKx` / :meth:`xtKy`, which avoid the inverse
        FFT by using Parseval's theorem.

        Parameters
        ----------
        x : np.ndarray
            Grid signal of shape ``(ny, nx)`` or ``(ny, nx, M)``.

        Returns
        -------
        np.ndarray
            ``K @ x`` with the same shape as ``x``.
        """
        x = np.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[..., np.newaxis]

        ny, nx, M = x.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Data shape ({ny}, {nx}) does not match kernel ({self.ny}, {self.nx})"
            )

        # HKH x = H · (K · (H x)). Subtract per-feature spatial mean both
        # before and after the FFT round-trip. Equivalent to zeroing the
        # DC coefficient directly in the frequency domain.
        if self.centering:
            x = x - x.mean(axis=(0, 1), keepdims=True)

        if self.fft_solver == "fft2":
            x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
            lam = self.spectrum.reshape(ny, nx, 1)
            out = np.real(scipy.fft.ifft2(lam * x_hat, axes=(0, 1), workers=self.workers))
        else:
            x_hat = scipy.fft.rfft2(x, axes=(0, 1), workers=self.workers)
            lam = self.spectrum.reshape(ny, nx // 2 + 1, 1)
            out = scipy.fft.irfft2(lam * x_hat, s=(ny, nx), axes=(0, 1), workers=self.workers)

        if self.centering:
            out = out - out.mean(axis=(0, 1), keepdims=True)

        return out[..., 0] if squeeze else out

    def xtKy(self, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """
        Bilinear form ``x^T K y`` on the grid via Parseval's theorem.

        Parameters
        ----------
        x, y : np.ndarray
            Grid signals of shape ``(ny, nx)`` or ``(ny, nx, M)``. Both must
            have the same shape.

        Returns
        -------
        float or np.ndarray
            Scalar if inputs are 2D; shape ``(M,)`` if 3D.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}.")
        squeeze = x.ndim == 2
        if squeeze:
            x = x[..., np.newaxis]
            y = y[..., np.newaxis]
        ny, nx, _ = x.shape
        # x^T HKH y = (H x)^T K (H y); both sides need centering.
        if self.centering:
            x = x - x.mean(axis=(0, 1), keepdims=True)
            y = y - y.mean(axis=(0, 1), keepdims=True)
        R_sum = _spectral_cross_product(x, y, self, ny, nx)
        R = R_sum / (ny * nx)
        if squeeze:
            return float(R.item()) if R.size == 1 else R.ravel()
        return R.ravel()

    def eigenvalues(self, k: int | None = None, return_full_layout: bool = False) -> np.ndarray:
        """
        Eigenvalues of the kernel matrix.

        When ``self.centering`` is True (default) the ``k=(0, 0)`` DC
        component is zeroed before returning — this is exactly the
        spectrum of ``HKH`` on a torus, since the constant vector ``𝟏``
        is the DC Fourier mode. Set ``centering=False`` at construction
        to recover the raw ``K`` spectrum.

        Parameters
        ----------
        k : int, optional
            Number of largest eigenvalues to return. If None, returns all.
        return_full_layout : bool, default False
            Only for ``fft_solver='rfft2'``. If True, returns eigenvalues
            in full FFT layout (ny, nx) flattened.
        """
        if self.spectrum is None:
            self.spectrum = self._compute_eigenvalues()

        # Resolve the raw spectrum array (respecting rfft2 vs fft2).
        if (self.fft_solver == "fft2") or (not return_full_layout):
            spec = self.spectrum
        else:  # fft_solver == 'rfft2' and return_full_layout=True
            # Convert rfft2 layout to full FFT layout.
            full_fft = np.zeros((self.ny, self.nx), dtype=self.spectrum.dtype)
            rfft_size = self.nx // 2 + 1
            full_fft[:, :rfft_size] = self.spectrum.reshape(self.ny, rfft_size)
            for i in range(self.ny):
                for j in range(1, rfft_size - 1):
                    full_fft[i, self.nx - j] = full_fft[i, j].conj()
            spec = full_fft.ravel()

        if self.centering:
            # Drop the constant-mode eigenvalue (DC entry).
            spec = spec.copy()
            spec[0] = 0.0

        if k is None:
            return spec
        idx = np.argsort(-spec)[:k]
        return spec[idx]

    def trace(self) -> float:
        """``trace(K)`` (raw) or ``trace(HKH)`` (centered).

        Closed-form ``Σ_k λ(k)`` — FFT diagonalizes ``K`` in the
        Fourier basis, so the trace is an ``O(n)`` sum over the
        spectrum. No stochastic path.
        """
        return float(np.sum(self.eigenvalues(return_full_layout=True)))

    def square_trace(self) -> float:
        """``trace(K²)`` (raw) or ``trace((HKH)²)`` (centered).

        Closed-form ``Σ_k λ(k)²`` — FFT diagonalization gives the
        spectrum directly.
        """
        return float(np.sum(self.eigenvalues(return_full_layout=True) ** 2))


def _q_test_fft(  # noqa: C901
    Xn: np.ndarray,
    kernel: FFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    FFT-accelerated spatial Q-test for grid data.

    Tests whether a spatial variable exhibits significant clustering or dispersion
    using FFT-based spectral decomposition. Parseval's theorem reduces the
    quadratic form to an elementwise spectral weighting.

    Parameters
    ----------
    Xn : np.ndarray
        Input data tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.
        Order follows kernel dimensions. Will be automatically reshaped to 3D if 2D.
    kernel : FFTKernel
        Pre-constructed FFT kernel object for grid data.
    null_params : dict, optional
        Pre-computed null distribution parameters from
        :func:`quadsv.statistics.compute_null_params`. When supplied, the
        cached ``eigenvalues`` / ``mean_Q`` / ``var_Q`` entries are reused
        in the p-value stage to avoid recomputing the spectrum on every
        call — useful when running the same kernel against many features
        (e.g., in :class:`quadsv.DetectorGrid`). If None, the
        spectrum and moments are computed on the fly.
    return_pval : bool, default True
        If True, returns (Q, pval) tuple; if False, returns Q only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally. Otherwise standardizes
        per-feature (mean 0, std 1) across spatial dimensions.

    Returns
    -------
    Q : float or np.ndarray
        Test statistic. Scalar if input was 2D; array of shape (M,) if 3D.
    pval : float or np.ndarray, optional
        Tail probability under null hypothesis. Only returned if return_pval=True.
        Uses Liu's method for most kernels; Normal approximation for Moran's I.

    Raises
    ------
    ValueError
        If ``Xn`` spatial dimensions don't match kernel shape ``(ny, nx)``.

    Notes
    -----
    Under H₀: data is spatially independent.
    Under H₁: mean-shift present.

    Computationally: ``Q = zᵀ K z`` where ``z`` is standardized data.
    Uses FFT via Parseval's theorem to compute :math:`Q = \\sum_{i,j} \\lambda_{i,j} Z^2_{i,j}`
    in O(n' log n') time instead of O(n'³) dense methods.

    For Moran's I kernel (which has negative eigenvalues), uses Normal approximation
    based on asymptotic theory. For other kernels, uses Liu's chi-squared mixture approximation.

    Examples
    --------
    >>> ny, nx = 32, 32
    >>> kernel = FFTKernel((ny, nx), method='gaussian', bandwidth=1.0)
    >>> data = np.random.randn(ny, nx)
    >>> Q, pval = spatial_q_test(data, kernel)
    """
    Xn = np.asarray(Xn).astype(float)
    if Xn.ndim == 2:
        Xn = Xn[..., np.newaxis]

    ny, nx, M = Xn.shape
    if ny != kernel.ny or nx != kernel.nx:
        raise ValueError(
            f"Data shape ({ny}, {nx}) does not match kernel ({kernel.ny}, {kernel.nx})"
        )

    # 1. Standardization (Z-score across spatial dimensions)
    if is_standardized:
        z = Xn
    else:
        # Mean/Std per feature slice
        means = np.mean(Xn, axis=(0, 1), keepdims=True)
        stds = np.std(Xn, axis=(0, 1), keepdims=True, ddof=1)

        # Handle constant features (std=0)
        # Create result array
        z = np.zeros_like(Xn)

        # Mask where std > 0 (shape 1,1,M broadcastable)
        valid = stds > 1e-12

        # Safe division
        np.divide(Xn - means, stds, out=z, where=valid)

    # 2. Compute Q statistic: z^T K z
    # Helper returns (M,) array or scalar if input was 2D
    Q = kernel.xtKx(z)

    if not return_pval:
        return Q

    # 3. P-value approximation. Dispatch on the user-selected null method
    # (``null_params['method']``) mirroring the MatrixKernel path in
    # :func:`quadsv.statistics.spatial_q_test`: any of 'clt' / 'welch' / 'liu'.
    # Default: CLT for Moran (indefinite K → Welch/Liu degenerate),
    # Liu for everything else. When `null_params` is supplied the caller's
    # cached moments are reused so we don't retraverse the spectrum per feature.
    Q_arr = np.atleast_1d(Q).astype(float).ravel()

    if null_params is not None and "method" in null_params:
        null_approx = str(null_params["method"])
    else:
        null_approx = "clt" if kernel.method == "moran" else "liu"
    if kernel.method == "moran" and null_approx != "clt":
        raise ValueError(
            f"Moran's I kernel is indefinite; only null_method='clt' is "
            f"supported for the Q-test. Got method={null_approx!r}."
        )

    def _get_mean_var() -> tuple[float, float]:
        if null_params is not None and "mean_Q" in null_params and "var_Q" in null_params:
            return float(null_params["mean_Q"]), float(null_params["var_Q"])
        # Fall back to compute_null_params so we get H-centered moments
        # with the finite-n ratio correction — raw trace(K), 2·trace(K²)
        # would inflate the null variance (see compute_null_params docstring).
        from quadsv.statistics import compute_null_params

        p = compute_null_params(kernel, method="clt")
        return float(p["mean_Q"]), float(p["var_Q"])

    if null_approx == "clt":
        mean_Q, var_Q = _get_mean_var()
        sigma = float(np.sqrt(var_Q))
        if sigma <= 1e-12:
            pvals = np.ones_like(Q_arr)
        else:
            z_scores = (Q_arr - mean_Q) / sigma
            pvals = chi2.sf(z_scores**2, df=1)

    elif null_approx == "welch":
        if null_params is not None and "scale_g" in null_params and "df_h" in null_params:
            g = float(null_params["scale_g"])
            h = float(null_params["df_h"])
        else:
            mean_Q, var_Q = _get_mean_var()
            if mean_Q <= 0 or var_Q <= 0:
                pvals = np.ones_like(Q_arr)
                g = h = None
            else:
                g = var_Q / (2.0 * mean_Q)
                h = 2.0 * mean_Q**2 / var_Q
        if g is not None:
            pvals = chi2.sf(Q_arr / g, df=h)

    elif null_approx == "liu":
        from quadsv.statistics import _liu_apply, _liu_prepare, _liu_prepare_from_cumulants

        # Dirichlet(1/2) variance correction: pass ``n`` so ``sigma_Q``
        # uses ``2·(m·c_2 − c_1²)/(m+2)`` rather than the large-n limit
        # ``2·c_2``. Matters on broad-spectrum PSD kernels where
        # ``c_1² ≈ m·c_2`` (e.g. CAR on a dense regular grid).
        n_kernel = int(kernel.n)
        coef = None if null_params is None else null_params.get("liu_coef")
        if coef is None and null_params is not None and "cumulants" in null_params:
            coef = _liu_prepare_from_cumulants(null_params["cumulants"], n=n_kernel)
        if coef is None:
            # No cached coef and no user-supplied cumulants — auto-build
            # from the kernel's own full spectrum (cheap for FFT: O(n)).
            if null_params is not None and null_params.keys() - {"method"}:
                raise ValueError(
                    "null_params with method='liu' must contain either "
                    "'liu_coef' (preferred) or 'cumulants'. Build via "
                    "compute_null_params(kernel, method='liu')."
                )
            evals = kernel.eigenvalues(return_full_layout=True)
            if evals.min() < -0.1:
                raise ValueError(
                    "Kernel has significant negative eigenvalues; Liu's method may be invalid."
                )
            sig_evals = evals[evals > 1e-9]
            coef = _liu_prepare(sig_evals, n=n_kernel)
        pvals = np.atleast_1d(_liu_apply(Q_arr, coef))

    else:
        raise ValueError(f"Unknown null approximation method: {null_approx!r}")

    # Unwrap to scalar if the caller passed a 2D grid for a single feature.
    if np.ndim(Q) == 0:
        return Q, float(pvals[0])
    return Q, pvals


def _standardize_grid(X: np.ndarray) -> np.ndarray:
    """Z-score standardize a grid tensor along spatial dims (0, 1)."""
    m = np.mean(X, axis=(0, 1), keepdims=True)
    s = np.std(X, axis=(0, 1), keepdims=True, ddof=1)
    Z = np.zeros_like(X)
    np.divide(X - m, s, out=Z, where=(s > 1e-12))
    return Z


def _spectral_cross_product(
    Zx: np.ndarray, Zy: np.ndarray, kernel: FFTKernel, ny: int, nx: int
) -> np.ndarray:
    """Compute sum of conj(Zx_hat) * lambda * Zy_hat in frequency domain."""
    if kernel.fft_solver == "fft2":
        Zx_hat = scipy.fft.fft2(Zx, axes=(0, 1), workers=kernel.workers)
        Zy_hat = scipy.fft.fft2(Zy, axes=(0, 1), workers=kernel.workers)
        lam = kernel.eigenvalues().reshape(ny, nx, 1)
        spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)
        return np.sum(spectral_prod, axis=(0, 1))

    # rfft2 case with symmetry correction
    Zx_hat = scipy.fft.rfft2(Zx, axes=(0, 1), workers=kernel.workers)
    Zy_hat = scipy.fft.rfft2(Zy, axes=(0, 1), workers=kernel.workers)
    lam = kernel.eigenvalues().reshape(ny, nx // 2 + 1, 1)
    spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)
    R_sum = 2.0 * np.sum(spectral_prod, axis=(0, 1))
    R_sum -= np.sum(spectral_prod[:, 0, :], axis=0)
    if nx % 2 == 0:
        R_sum -= np.sum(spectral_prod[:, -1, :], axis=0)
    return R_sum


def _r_test_fft(
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: FFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    FFT-accelerated spatial R-test (bivariate) for grid data.

    Tests for spatial co-variation between two variables using the specified kernel.
    Computes the cross-variance statistic ``R = xᵀ K y`` via FFT-based spectral methods.

    Parameters
    ----------
    Xn : np.ndarray
        First input tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.
        Will be automatically reshaped to 3D if 2D.
    Yn : np.ndarray
        Second input tensor. Must have the same shape as Xn.
    kernel : FFTKernel
        Pre-constructed FFT kernel object for grid data.
    null_params : dict, optional
        Pre-computed null parameters from
        :func:`quadsv.statistics.compute_null_params`. Only the
        ``var_R = trace(K²)`` entry is consumed here; when None, it is
        computed on the fly from ``kernel.square_trace()``.
    return_pval : bool, default True
        If True, returns (R, pval) tuple; if False, returns R only.
    is_standardized : bool, default False
        If True, skips standardization. Otherwise standardizes each variable
        independently (mean 0, std 1) across spatial dimensions.

    Returns
    -------
    R : float or np.ndarray
        Test statistic (cross-variance). Scalar if input was 2D; array of shape (M,) if 3D.
    pval : float or np.ndarray, optional
        Two-tailed p-value under null hypothesis (no spatial co-variation).
        Based on Normal approximation: :math:`z = R / \\sqrt{\\text{Trace}(K^2)}`.
        Only returned if return_pval=True.

    Raises
    ------
    ValueError
        If ``Xn`` and ``Yn`` shapes don't match, or spatial dimensions don't match kernel.

    Notes
    -----
    Under H₀: x and y are spatially independent.
    Under H₁: spatial co-clustering or co-dispersion present.

    Computationally: R = z_x^T K z_y where z_x, z_y are standardized data.
    Uses FFT via Parseval's theorem: :math:`R = \\frac{1}{N} \\sum_{i,j} \\overline{Z_{x}}_{i,j} \\lambda_{i,j} Z_{y_{i,j}}`

    P-value calculation assumes asymptotic Normality with variance estimated from
    kernel trace: :math:`\\text{Var}(R) \\approx \\text{Trace}(K^2) / N^2`.
    Returns two-tailed probability: :math:`p = 2 P(|Z| > |\\text{z-score}|)`.

    Examples
    --------
    >>> ny, nx = 32, 32
    >>> kernel = FFTKernel((ny, nx), method='gaussian', bandwidth=1.0)
    >>> x_data = np.random.randn(ny, nx)
    >>> y_data = np.random.randn(ny, nx)
    >>> R, pval = spatial_r_test(x_data, y_data, kernel)
    """
    Xn = np.asarray(Xn).astype(float)
    Yn = np.asarray(Yn).astype(float)

    if Xn.ndim == 2:
        Xn = Xn[..., np.newaxis]
    if Yn.ndim == 2:
        Yn = Yn[..., np.newaxis]

    ny, nx, M = Xn.shape
    if Xn.shape != Yn.shape:
        raise ValueError(f"Xn and Yn shapes must match, got {Xn.shape} and {Yn.shape}")
    if ny != kernel.ny or nx != kernel.nx:
        raise ValueError(
            f"Data shape ({ny}, {nx}) does not match kernel ({kernel.ny}, {kernel.nx})"
        )

    # 1. Standardization
    if is_standardized:
        Zx, Zy = Xn, Yn
    else:
        Zx = _standardize_grid(Xn)
        Zy = _standardize_grid(Yn)

    # 2. Compute R = Zx^T K Zy via FFT (Parseval's theorem)
    R_sum = _spectral_cross_product(Zx, Zy, kernel, ny, nx)

    # Apply Parseval's 1/n normalization
    n_pixels = ny * nx
    R = R_sum / n_pixels

    # Unwrap if M=1
    if M == 1 and R.size == 1:
        R = R.item()

    if not return_pval:
        return R

    # 3. P-values (Normal Approximation). R is Normal under H₀ with
    # variance ``trace((HKH)²)`` — NOT ``trace(K²)``, since both X and Y
    # are z-scored before R = Zₓᵀ K Zᵧ is formed. Honor a precomputed
    # ``var_R`` if the caller supplied one via ``compute_null_params``.
    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        # kernel.square_trace() returns trace((HKH)²) by default (centering=True).
        var_R = float(kernel.square_trace())
    sigma = np.sqrt(var_R)

    if sigma > 1e-12:
        z_scores = R / sigma
        pval = 2 * norm.sf(np.abs(z_scores))
    else:
        pval = np.ones_like(R) if isinstance(R, np.ndarray) else 1.0

    return R, pval
