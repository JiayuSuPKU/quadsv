"""
Built-in concrete matrix kernel.

:class:`MatrixKernel` is the standard subclass of
:class:`~quadsv.kernels.base.MatrixKernelBase`. It carries the
construction logic for turning a coordinate cloud or a precomputed
matrix into the underlying ``_K`` storage; the algorithm itself
(Kx / xtKx / trace / etc.) is inherited unchanged from
:class:`~quadsv.kernels.base.MatrixKernelBase`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv
from sklearn.neighbors import NearestNeighbors

from quadsv.kernels.base import MatrixKernelBase

__all__ = ["MatrixKernel"]


class MatrixKernel(MatrixKernelBase):
    """
    Built-in matrix kernel constructed from coordinates or a precomputed matrix.

    Carries only the **construction** logic on converting coordinates ``S`` or a
    user-supplied matrix into ``_K`` on top of :class:`MatrixKernelBase`.

    If you want a bespoke kernel builder (e.g. a custom distance decay, a
    cross-modality covariance, a learnt operator) subclass
    :class:`MatrixKernelBase` directly and implement :meth:`_build_kernel`.

    See Also
    --------
    MatrixKernel.from_coordinates
        Recommended entry point when working from raw sample coordinates.
    MatrixKernel.from_matrix
        Recommended entry point when a kernel or precision matrix is already
        available.
    MatrixKernelBase
        Base class to inherit from for custom kernel constructions.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self, data: np.ndarray, mode: str = "coords", method: str = "matern", **kwargs
    ) -> None:
        """
        Construct a spatial kernel from already-prepared input data.

        This constructor is public but low-level; most users should prefer the
        factory methods :meth:`from_coordinates` or :meth:`from_matrix`, which
        dispatch to this constructor with the appropriate ``mode``.

        Parameters
        ----------
        data : np.ndarray or scipy.sparse matrix
            Input data whose interpretation is controlled by ``mode``:
            an ``(n, D)`` coordinate array when ``mode='coords'``, an ``(n, n)``
            kernel matrix when ``mode='precomputed'``, or an ``(n, n)`` precision
            matrix when ``mode='precomputed_inverse'``.
        mode : {'coords', 'precomputed', 'precomputed_inverse'}, default 'coords'
            How ``data`` should be interpreted.
        method : str, default 'matern'
            Kernel method. Must be one of ``'gaussian'``, ``'matern'``, ``'moran'``,
            ``'graph_laplacian'``, ``'car'``, or ``'precomputed'``.
        **kwargs : dict
            Kernel-specific parameters (e.g., ``bandwidth``, ``nu``, ``rho``,
            ``k_neighbors``). Unknown keys raise :class:`ValueError`.

        Raises
        ------
        ValueError
            If ``mode`` or ``method`` is unknown, or any parameter fails validation.
        """
        self._data = data
        self._mode = mode
        if mode not in ("coords", "precomputed", "precomputed_inverse"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'coords', 'precomputed', or 'precomputed_inverse'."
            )

        if method not in self._available_kernels + ["precomputed"]:
            raise ValueError(f"Unknown kernel method: {method}.")

        # Pop the ``centering`` flag out of kwargs before validating — it
        # is a Kernel-ABC-level argument, not a per-method hyper-parameter.
        centering = kwargs.pop("centering", True)

        # Update kernel parameters from defaults
        defaults = self._get_default_params(method).copy()
        if kwargs:
            for key, value in kwargs.items():
                if key in defaults:
                    defaults[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        n = data.shape[0]
        if mode == "coords":
            self._validate_coords_params(n, method, defaults)

        super().__init__(n, method=method, centering=centering, **defaults)

    @staticmethod
    def _validate_coords_params(n: int, method: str, params: dict) -> None:
        """Validate parameters for coordinate-based kernel construction."""
        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}")
        if method in ("gaussian", "matern"):
            bw = params.get("bandwidth", None)
            if bw is not None and bw <= 0:
                raise ValueError(f"bandwidth must be positive, got {bw}")
        if method == "matern":
            nu = params.get("nu", None)
            if nu is not None and nu <= 0:
                raise ValueError(f"nu must be positive, got {nu}")
        if method in ("moran", "graph_laplacian", "car"):
            k = params.get("k_neighbors", None)
            if k is not None and (k < 1 or k >= n):
                raise ValueError(f"k_neighbors must be in [1, {n - 1}], got {k}")
        if method == "car":
            rho = params.get("rho", None)
            if rho is not None and rho < 0:
                raise ValueError(f"rho must be non-negative, got {rho}")

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
            Method defaults: bandwidth (gaussian/matern), nu (matern), k_neighbors (moran/graph_laplacian/car), rho (car).
        """
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"bandwidth": 2.0, "nu": 1.5},
            "moran": {"k_neighbors": 4},
            "graph_laplacian": {"k_neighbors": 4},
            "car": {"rho": 0.9, "k_neighbors": 4, "standardize": False},
        }
        return method_defaults.get(method, {})

    @classmethod
    def from_coordinates(cls, coords: np.ndarray, method: str = "matern", **kwargs) -> MatrixKernel:
        """
        Build kernel from spatial coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Array of spatial coordinates, shape (n, D).
        method : str, default 'matern'
            Kernel method. Must be one of 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
        **kwargs : dict
            Additional kernel parameters (bandwidth, nu, rho, k_neighbors, etc.).

        Returns
        -------
        MatrixKernel
            Initialized kernel object.

        Raises
        ------
        ValueError
            If ``method`` is not one of :attr:`_available_kernels`.

        Examples
        --------
        >>> coords = np.random.randn(100, 2)
        >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian', bandwidth=1.0)
        """
        if method not in cls._available_kernels:
            raise ValueError(f"Unknown kernel method for coordinates: {method}.")

        return cls(coords, mode="coords", method=method, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray | sp.spmatrix,
        is_precision: bool = False,
        method: str = "precomputed",
        **kwargs,
    ) -> MatrixKernel:
        """
        Build kernel from a precomputed kernel matrix or its inverse.

        Parameters
        ----------
        matrix : np.ndarray or scipy.sparse matrix
            Kernel matrix ``(n, n)`` or its inverse (precision matrix).
        is_precision : bool, default False
            If True, ``matrix`` is treated as the inverse (precision) matrix ``K^{-1}``.
        method : str, default 'precomputed'
            The logical kernel method (e.g., 'car' for precision matrices).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        MatrixKernel
            Initialized kernel object.

        Examples
        --------
        >>> K = np.array([[2, -1], [-1, 2]])  # kernel matrix
        >>> kernel = MatrixKernel.from_matrix(K, is_precision=False)
        """
        mode = "precomputed_inverse" if is_precision else "precomputed"
        return cls(matrix, mode=mode, method=method, **kwargs)

    def _build_kernel(self):  # noqa: C901
        method = self.method

        # ==========================================
        # 1. PREPARE RAW INPUTS (Dists or Weights)
        # ==========================================

        # Case A: Coordinates provided -> Compute Dists or W from scratch
        if self._mode == "coords":
            coords = self._data
            if method in ["gaussian", "matern"]:
                # Compute dense distance matrix
                dists = squareform(pdist(coords, metric="euclidean"))
                W = None
            elif method in ["moran", "graph_laplacian", "car"]:
                # Compute sparse adjacency graph
                k = self.params["k_neighbors"]
                nbrs = NearestNeighbors(
                    n_neighbors=k + 1, algorithm="auto", metric="euclidean"
                ).fit(coords)
                W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)

                # Mutual neighbors: keep only edges where both spots list each other
                W_mut = W + W.T
                W_mut.data = (W_mut.data > 1).astype(float)
                W_mut.setdiag(0)

                # Handle isolated nodes: add self-loop to avoid division-by-zero
                row_sums = np.asarray(W_mut.sum(axis=1)).ravel()
                isolated = row_sums == 0
                if isolated.any():
                    W_mut.setdiag(isolated.astype(float))
                W_mut.eliminate_zeros()

                W = W_mut
                dists = None
            else:
                raise ValueError(f"Unknown method for coordinates: {method}")

        # Case B: Precomputed Kernel provided
        elif self._mode == "precomputed":
            return self._data

        # Case C: Precomputed Inverse Kernel provided
        elif self._mode == "precomputed_inverse":
            M = self._data
            standardize = self.params.get("standardize", False)

            # If small, realize dense K; else keep implicit precision
            if self.n <= self._implicit_threshold:
                try:
                    M_dense = M.toarray() if sp.issparse(M) else M
                    K_dense = inv(M_dense)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "Precision matrix is singular; using pseudo-inverse.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    M_dense = M.toarray() if sp.issparse(M) else M
                    K_dense = np.linalg.pinv(M_dense)

                if standardize:
                    diag_K = np.diag(np.asarray(K_dense)).copy()
                    diag_K[diag_K <= 0] = 1e-12
                    s = 1.0 / np.sqrt(diag_K)
                    S = np.diag(s)
                    K_dense = S @ K_dense @ S

                return 0.5 * (K_dense + K_dense.T)
            else:
                self.stores_precision = True
                if standardize:
                    M = self._standardize_precision(M)
                return M

        # ==========================================
        # 2. CONSTRUCT KERNEL FROM INPUTS
        # ==========================================

        # --- Distance Based ---
        if method in ["gaussian", "matern"]:
            bw = self.params["bandwidth"]

            if method == "gaussian":
                K = np.exp(-(dists**2) / (2 * bw**2))
            elif method == "matern":
                nu = self.params["nu"]
                length_scale = bw
                # Mask zero distances; evaluate Bessel K only at non-zero distances
                mask_zero = dists == 0
                dists_safe = dists.copy()
                dists_safe[mask_zero] = 1.0  # dummy value, overwritten below
                factor = (np.sqrt(2 * nu) * dists_safe) / length_scale
                K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
                K[mask_zero] = 1.0  # correct limit: K(x, x) = 1
            return K

        # --- Graph Based ---
        elif method in ["moran", "graph_laplacian", "car"]:
            # Symmetrize and apply symmetric normalization: D^{-1/2} W D^{-1/2}
            if W is None:
                raise ValueError("Graph weights (W) required for graph kernels.")

            # Ensure float
            W = W.astype(float)

            # Symmetrize first
            W_sym = 0.5 * (W + W.T)

            # Zero out self-loops
            W_sym.setdiag(0)

            # Degree-based symmetric normalization
            row_sums = np.array(W_sym.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_D_sqrt = sp.diags(1.0 / np.sqrt(row_sums))
            W_norm = inv_D_sqrt @ W_sym @ inv_D_sqrt

            if method == "moran":
                # Already symmetric and normalized
                return W_norm

            elif method == "graph_laplacian":
                I = sp.eye(self.n, format="csr")
                return I - W_norm

            elif method == "car":
                rho = self.params["rho"]
                if rho >= 1.0:
                    warnings.warn(
                        f"rho={rho} >= 1.0 causes singularity in CAR kernel; clamping to 0.99",
                        UserWarning,
                        stacklevel=2,
                    )
                    rho = 0.99
                    self.params["rho"] = rho
                standardize = self.params["standardize"]
                I = sp.eye(self.n, format="csc")
                # M = (I - rho * W_norm) is the inverse of the CAR kernel
                M = I - rho * W_norm

                if self.n > self._implicit_threshold:
                    self.stores_precision = True
                    if standardize:
                        M = self._standardize_precision(M)
                    return M
                else:
                    try:
                        K_dense = inv(M.toarray())
                    except np.linalg.LinAlgError:
                        warnings.warn(
                            "CAR precision matrix is singular; using pseudo-inverse. "
                            "Consider reducing rho or changing k_neighbors.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        K_dense = np.linalg.pinv(M.toarray())

                    if standardize:
                        diag_K = np.diag(np.asarray(K_dense)).copy()
                        diag_K[diag_K <= 0] = 1e-12
                        s = 1.0 / np.sqrt(diag_K)
                        S = np.diag(s)
                        K_dense = S @ K_dense @ S

                    return 0.5 * (K_dense + K_dense.T)

        else:
            raise ValueError(f"Unknown kernel method: {method}")

    def __repr__(self):
        # Describe input data succinctly
        if self._mode == "coords":
            coords = self._data
            data_desc = f"coords shape={getattr(coords, 'shape', '?')}"
        elif self._mode == "precomputed":
            M = self._data
            if sp.issparse(M):
                data_desc = f"matrix shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"matrix shape={getattr(M, 'shape', '?')} dense"
        elif self._mode == "precomputed_inverse":
            M = self._data
            if sp.issparse(M):
                data_desc = f"precision shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"precision shape={getattr(M, 'shape', '?')} dense"
        else:
            data_desc = "data=?"

        return (
            f"<MatrixKernel method={self.method} mode={self._mode} n={self.n} "
            f"implicit={self.stores_precision} data={data_desc} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        # Human-friendly multi-line summary
        lines = [
            "MatrixKernel",
            f"- Method: {self.method}",
            f"- Mode: {self._mode}",
            f"- Samples: {self.n}",
            f"- Implicit: {self.stores_precision} (threshold={self._implicit_threshold})",
        ]

        # Add a brief data description
        try:
            if self._mode == "coords":
                coords = self._data
                lines.append(f"- Data: coords shape={getattr(coords, 'shape', '?')}")
            else:
                M = self._data
                if sp.issparse(M):
                    kind = "precision" if self._mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={M.shape} sparse nnz={M.nnz}")
                else:
                    kind = "precision" if self._mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={getattr(M, 'shape', '?')} dense")
        except Exception:
            lines.append("- Data: ?")

        lines.append(f"- Params: {self._format_params()}")
        return "\n".join(lines)
