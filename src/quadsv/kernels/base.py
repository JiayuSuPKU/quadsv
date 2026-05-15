"""
Abstract bases for the spatial-kernel layer.

This module hosts the two ABCs:

- :class:`Kernel`: the universal interface every kernel implements
  (``Kx`` / ``xtKx`` / ``xtKy`` / ``trace`` / ``square_trace`` /
  ``eigenvalues``). Subclass this for a custom kernel that can be
  expressed through a single ``self._K`` buffer.
- :class:`MatrixKernelBase`: matrix-form kernels with dense / sparse
  / sparse-precision auto-switching. Subclass this for a new matrix
  backend; :class:`~quadsv.MatrixKernel` is the standard concrete
  subclass.

Concrete classes live in sibling modules: :mod:`quadsv.kernels.matrix`,
:mod:`quadsv.kernels.fft`, and :mod:`quadsv.kernels.nufft`.
"""

from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv, lu_factor, lu_solve
from scipy.sparse.linalg import splu
from tqdm import tqdm

__all__ = ["Kernel", "MatrixKernelBase"]


class Kernel(ABC):
    """
    Abstract base class shared by all spatial kernels in :mod:`quadsv`.

    Concrete backends:

    - :class:`MatrixKernel` — explicit n×n kernel or its sparse precision matrix.
    - :class:`quadsv.fft.FFTKernel` — grid kernel via its eigenvalue spectrum.
    - :class:`quadsv.nufft.NUFFTKernel` — irregular-point kernel evaluated through a
      type-1 / type-2 NUFFT round-trip.

    Required interface
    ------------------
    Every concrete kernel exposes:

    - ``n``, ``method``, ``params``, ``centering`` — integer / string /
      dict / bool attributes.
    - :meth:`xtKx`, :meth:`xtKy`, :meth:`Kx` — quadratic / bilinear /
      apply primitives.

    .. note::

        The empirical data centering (z-scoring) inside
        :func:`quadsv.spatial_q_test` / :func:`~quadsv.spatial_r_test`
        breaks independence across spatial obervations. As a result,
        the null distribution of the test statistic ``Q = Zᵀ K Z = Xᵀ (H K H) X / σ²``
        with ``H = I - 𝟏𝟏ᵀ/n`` should inspect the spectrum of a centered kernel ``HKH``.
        Every :class:`Kernel` carries a ``centering`` flag (default ``True``).
        Set ``centering=False`` to recover the raw ``K`` moments
        (useful for diagnostics or theoretical comparison).

    """

    n: int
    method: str
    params: dict
    centering: bool

    def __init__(self, *args, centering: bool = True, **kwargs):
        # Concrete subclasses handle ``n``/``method``/``params`` in their own
        # __init__; this just sets the centering flag and chains up so an
        # explicit ``super().__init__(centering=...)`` can flip it. Extra
        # args/kwargs are ignored here so multiple-inheritance-style init
        # chains remain safe.
        self.centering = bool(centering)

    # When ``self.centering`` is True, each of the six public methods
    # below returns the quantity for the centered operator
    # ``HKH = (I − 𝟏𝟏ᵀ/n) K (I − 𝟏𝟏ᵀ/n)`` — the operator that actually
    # acts on z-scored data inside :func:`spatial_q_test` and
    # :func:`spatial_r_test`. Set ``centering=False`` to expose the raw
    # ``K`` primitives (diagnostics, comparisons to literature).

    @abstractmethod
    def xtKx(self, x):
        """``xᵀ K x`` (raw) or ``xᵀ HKH x`` (centered)."""

    @abstractmethod
    def xtKy(self, x, y):
        """``xᵀ K y`` (raw) or ``xᵀ HKH y`` (centered)."""

    @abstractmethod
    def Kx(self, x):
        """``K x`` (raw) or ``HKH x`` (centered)."""

    @abstractmethod
    def trace(self) -> float:
        """``trace(K)`` or ``trace(HKH) = trace(K) − s₁/n``.

        Concrete backends are free to add kwargs for backend-specific
        options (e.g. :class:`MatrixKernelBase` exposes ``n_probes``
        for its precision-stored Hutchinson path).
        """

    @abstractmethod
    def square_trace(self) -> float:
        """``trace(K²)`` or ``trace((HKH)²) = trace(K²) − 2·s₂/n + s₁²/n²``.

        Same contract as :meth:`trace` re: backend-specific kwargs.
        """

    @abstractmethod
    def eigenvalues(self, k: int | None = None) -> np.ndarray:
        """Eigenvalues of ``K`` (raw) or ``HKH`` (centered)."""

    # ------------------------------------------------------------------
    # Optional helpers used by concrete backends. Kept non-abstract so
    # each kernel can override or ignore them — they exist to factor the
    # common centering arithmetic, not to constitute an interface.
    # ------------------------------------------------------------------
    @staticmethod
    def _center_vec(x: np.ndarray) -> np.ndarray:
        """``H x`` — subtract the column mean. Works on 1-D or 2-D arrays."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x - x.mean()
        return x - x.mean(axis=0, keepdims=True)

    def _ones_stats(self) -> tuple[float, float]:
        """``(s1, s2) = (𝟏ᵀ K 𝟏, ‖K·𝟏‖²)`` via one raw ``K·𝟏`` application.

        Requires a transient ``centering=False`` view — we need the raw
        ``K·𝟏``, not the centered ``HKH·𝟏 = 0``.
        """
        prev = self.centering
        self.centering = False
        try:
            ones = np.ones(self.n, dtype=float)
            c = np.asarray(self.Kx(ones)).ravel()
        finally:
            self.centering = prev
        return float(c.sum()), float(c @ c)


class MatrixKernelBase(Kernel):
    """
    Concrete base for kernels backed by an explicit or implicit ``n × n`` matrix.

    Subclass this when you want a custom way to construct ``K`` (e.g. a bespoke
    function of coordinates, a learnt kernel, or an ad-hoc adjacency matrix) while
    inheriting every downstream primitive ready-to-use.

    Subclasses must implement :meth:`_build_kernel` to return either the
    ``(n, n)`` kernel matrix (dense ``np.ndarray`` or ``scipy.sparse``) or
    its precision matrix ``M = K^{-1}``. If a precision matrix is returned,
    set ``self.stores_precision = True`` before calling ``super().__init__``,
    or flip it inside ``_build_kernel`` while the buffer is being built; the
    :class:`MatrixKernel` CAR / precomputed-inverse path shows a worked example.

    Everything else — cached sparse LU for implicit solves, Hutchinson
    trace estimation when ``stores_precision=True``, automatic sparse-vs-dense
    dispatch in ``xtKx`` / ``Kx``, standardized-quadratic-form cache
    (``_K_column_sums`` / :meth:`xtKx_standardized`), pickle safety for the
    non-picklable LU factor — is already implemented here.

    Handles dense, sparse, and implicit (operator-based) kernels. Switches between
    an explicit representation (the kernel matrix ``K`` is stored and used directly)
    and an implicit representation (the precision matrix ``M = K^{-1}`` is stored and
    linear systems are solved on demand) based on problem size.

    Attributes
    ----------
    n : int
        Number of observations (``n``).
    method : str
        Kernel method name (free-form; used only for diagnostics / provenance).
    params : dict
        Resolved kernel parameters — subclasses decide what goes here.
    stores_precision : bool
        If ``True``, ``_K`` holds the precision matrix ``M = K^{-1}`` and linear
        solves (via a cached LU) are used for :meth:`xtKx` and trace estimation.
        If ``False``, ``_K`` is the realized kernel matrix (dense or sparse).

    Notes
    -----
    The internal buffer ``_K`` stores the kernel matrix when ``stores_precision=False``
    and the precision matrix ``K^{-1}`` when ``stores_precision=True``. Public methods
    (:meth:`xtKx`, :meth:`trace`, :meth:`square_trace`, :meth:`eigenvalues`)
    transparently handle both cases; callers should not access ``_K`` directly.
    """

    def __init__(
        self, n: int, method: str = "gaussian", *, centering: bool = True, **kwargs
    ) -> None:
        """
        Initialize the Kernel.

        Parameters
        ----------
        n : int
            Number of observations.
        method : str, default 'gaussian'
            Kernel method to use.
        centering : bool, default True
            If ``True``, :meth:`trace`, :meth:`square_trace`, and
            :meth:`eigenvalues` return the moments of the centered
            operator ``HKH`` — the one :func:`spatial_q_test` /
            :func:`spatial_r_test` actually apply after z-scoring. Set
            ``False`` to recover the raw ``K`` moments (diagnostic /
            theoretical comparison only).
        **kwargs : dict
            Additional kernel-specific parameters stored in ``self.params``.
        """
        super().__init__(centering=centering)
        self.n: int = n
        """Number of observations (samples)."""
        self.method: str = method
        """Kernel method name."""
        self.params: dict = kwargs
        """Resolved kernel parameters after defaults are merged with user overrides."""

        # Threshold (in samples) for switching to the implicit representation.
        self._implicit_threshold = 5000
        self.stores_precision: bool = False
        """Whether the kernel is stored in precision form (``True``) or as the realized kernel matrix (``False``)."""
        self._lu = None  # Cache for sparse LU factorization if needed
        self._lu_lock = threading.Lock()  # Thread safety for lazy LU init

        # _K stores the kernel matrix when stores_precision=False and the precision
        # matrix K^{-1} when stores_precision=True (see class Notes).
        self._K = self._build_kernel()
        # Lazy per-mode spectrum caches; populated by :meth:`eigenvalues`.
        self._spectrum_raw: np.ndarray | None = None
        self._spectrum_centered: np.ndarray | None = None

    @abstractmethod
    def _build_kernel(self):
        """Constructs the kernel matrix or its inverse operator."""
        pass

    def _format_params(self):
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

    def __repr__(self):
        return (
            f"<Kernel method={self.method} n={self.n} implicit={self.stores_precision} "
            f"threshold={self._implicit_threshold} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        return (
            "Kernel\n"
            f"- Method: {self.method}\n"
            f"- Samples: {self.n}\n"
            f"- Implicit: {self.stores_precision} (threshold={self._implicit_threshold})\n"
            f"- Params: {self._format_params()}"
        )

    def realization(self) -> np.ndarray | sp.spmatrix:
        """
        Return the realized ``(n, n)`` kernel matrix.

        When ``self.centering`` is False, returns ``K`` as stored (dense
        ndarray or sparse matrix). When ``self.centering`` is True,
        returns the centered operator ``HKH`` with ``H = I − 𝟏𝟏ᵀ/n`` —
        the operator that :meth:`xtKx` / :meth:`Kx` / :meth:`trace` /
        :meth:`eigenvalues` all actually apply. ``HKH`` is dense even
        when ``K`` is sparse (each row gets a column-mean subtracted), so
        the centered path always materializes a dense result.

        Notes
        -----
        If ``stores_precision`` is True, this forces expensive dense inversion of the
        precision matrix. Prefer :meth:`xtKx` and :meth:`trace` for implicit kernels.
        """
        # 1) Resolve the raw kernel matrix.
        if self.stores_precision:
            # _K is M = K^-1. We need to invert it.
            if sp.issparse(self._K):
                K = inv(self._K.toarray())
            else:
                K = inv(self._K)
        else:
            K = self._K

        if not self.centering:
            return K

        # 2) Centered view: HKH = K − 𝟏rᵀ − c𝟏ᵀ + m·𝟏𝟏ᵀ, where
        #    r = row-mean(K), c = col-mean(K), m = grand-mean(K). Always
        #    dense output (row-mean subtraction destroys sparsity).
        K_dense = K.toarray() if sp.issparse(K) else np.asarray(K)
        row_mean = K_dense.mean(axis=1, keepdims=True)
        col_mean = K_dense.mean(axis=0, keepdims=True)
        grand_mean = float(K_dense.mean())
        return K_dense - row_mean - col_mean + grand_mean

    def eigenvalues(self, k: int | None = None) -> np.ndarray:  # noqa: C901
        """
        Eigenvalues of ``K`` or ``HKH`` (according to ``self.centering``).

        Three paths:

        - **Dense ``K``** — form
          ``HKH = K − 𝟏 r^T − c 𝟏^T + m · 𝟏𝟏^T``
          in closed form (``r`` / ``c`` = row / col means, ``m`` = grand
          mean) and call :func:`numpy.linalg.eigvalsh`. Same O(n³) cost
          as the raw case, no extra memory.
        - **Sparse explicit ``K``** — wrap ``H K H · v`` as a
          :class:`scipy.sparse.linalg.LinearOperator` and call
          :func:`eigsh`. Preserves ``K``'s sparsity — we never densify.
        - **Implicit precision (sparse ``M = K⁻¹``)** — wrap
          ``H K H · v = H (M⁻¹ (H v))`` where ``M⁻¹·`` is the cached
          sparse-LU solve. Same sparsity benefit as the raw-inverse path.

        Results are cached per centering mode; switching ``self.centering``
        invalidates and recomputes.

        Parameters
        ----------
        k : int, optional
            Number of largest-magnitude eigenvalues to return. If None,
            returns all (dense) or ``max(6, n − 2)`` (sparse/implicit —
            limited by what ``eigsh`` can extract).

        Returns
        -------
        np.ndarray
            Eigenvalues sorted in descending order.
        """
        # Per-mode cache (raw vs centered spectra differ).
        cache_key = "_spectrum_centered" if self.centering else "_spectrum_raw"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            if k is None and len(cached) == self.n:
                return cached
            if k is not None and len(cached) >= k:
                return cached[:k]

        k_orig = k
        centered = self.centering

        if self.stores_precision:
            # Implicit sparse precision ``M`` — solve systems instead of
            # densifying. Raw spectrum via ``eigsh(M, which='SM')`` of the
            # precision (inverted). Centered via an ``HKH`` LinearOperator
            # backed by the same sparse-LU solver.
            from scipy.sparse.linalg import LinearOperator, eigsh

            k = k if k is not None else max(6, self.n - 2)
            if not centered:
                vals, _ = eigsh(self._K, k=k, which="SM")
                vals = np.real(1.0 / vals)
            else:
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = splu(self._K.tocsc()) if sp.issparse(self._K) else None
                lu = self._lu

                def _hkh_matvec(v: np.ndarray) -> np.ndarray:
                    v_c = v - v.mean()
                    if lu is not None:
                        kv = lu.solve(v_c)
                    else:  # dense precision fallback
                        kv = lu_solve(lu_factor(self._K), v_c)
                    return kv - kv.mean()

                op = LinearOperator(shape=(self.n, self.n), matvec=_hkh_matvec, dtype=float)
                vals, _ = eigsh(op, k=k, which="LM")
                vals = np.real(vals)
        else:
            # Explicit ``K`` buffer (dense or sparse).
            if sp.issparse(self._K):
                from scipy.sparse.linalg import LinearOperator, eigsh

                k = k if k is not None else max(6, self.n - 2)
                if not centered:
                    vals, _ = eigsh(self._K, k=k, which="LM")
                else:
                    K_sparse = self._K

                    def _hkh_matvec(v: np.ndarray) -> np.ndarray:
                        v_c = v - v.mean()
                        kv = K_sparse @ v_c
                        if sp.issparse(kv):
                            kv = np.asarray(kv.todense()).ravel()
                        else:
                            kv = np.asarray(kv).ravel()
                        return kv - kv.mean()

                    op = LinearOperator(shape=(self.n, self.n), matvec=_hkh_matvec, dtype=float)
                    vals, _ = eigsh(op, k=k, which="LM")
                vals = np.real(vals)
            else:
                if not centered:
                    vals = np.linalg.eigvalsh(self._K)
                else:
                    # ``HKH = K − 𝟏rᵀ − c𝟏ᵀ + m·𝟏𝟏ᵀ`` where r = row-mean
                    # of K, c = col-mean of K, m = grand-mean. Same
                    # asymptotic cost as eigvalsh(K); no dense HKH copy
                    # beyond a few n×1 means.
                    K = self._K
                    row_mean = K.mean(axis=1, keepdims=True)  # (n, 1)
                    col_mean = K.mean(axis=0, keepdims=True)  # (1, n)
                    grand_mean = float(K.mean())
                    HKH = K - row_mean - col_mean + grand_mean
                    vals = np.linalg.eigvalsh(HKH)

        spectrum = np.sort(vals)[::-1]  # descending
        setattr(self, cache_key, spectrum)
        return spectrum if k_orig is None else spectrum[:k_orig]

    # ------------------------------------------------------------------
    # Internal primitive: compute ``K @ x`` as a dense 2D block
    # ------------------------------------------------------------------
    def _apply_K_dense(self, x_2d: np.ndarray) -> np.ndarray:
        """Compute ``K @ x_2d`` and return a dense ``(n, M)`` ndarray.

        Used as the shared kernel of :meth:`Kx`, :meth:`xtKx`, :meth:`xtKy`.
        Expects a dense ``(n, M)`` input; sparse inputs must be densified by
        the caller. Implicit precision solves go through a cached LU; explicit
        kernels dispatch to the underlying sparse / dense matmul.
        """
        if self.stores_precision:
            if sp.issparse(self._K):
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = splu(self._K.tocsc())
                y = self._lu.solve(x_2d)
            else:
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = lu_factor(self._K)
                y = lu_solve(self._lu, x_2d)
            return np.asarray(y)
        # Explicit: sparse K → K.dot(dense) returns dense. Dense K → dense @ dense.
        y = self._K.dot(x_2d)
        if sp.issparse(y):  # pragma: no cover — current scipy always returns dense
            y = np.asarray(y.todense())
        return np.asarray(y)

    @staticmethod
    def _to_2d(x: np.ndarray | sp.spmatrix) -> tuple[Any, bool]:
        """Normalize ``x`` to a 2D ``(n, M)`` (sparse or dense) and report whether
        the caller passed a 1D vector. Does *not* densify sparse input.
        """
        if sp.issparse(x):
            if x.ndim == 1 or x.shape[1] == 1 and x.shape[0] == 1:
                # scipy rarely exposes 1D sparse; reshape if someone slipped it in.
                return x.reshape(-1, 1), True
            if x.shape[1] == 1 and x.shape[0] > 1:
                return x, False  # already a (n, 1) sparse column
            return x, False
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.reshape(-1, 1), True
        return arr, False

    def Kx(self, x: np.ndarray | sp.spmatrix) -> np.ndarray:
        """
        Apply the kernel operator to ``x``, returning ``K @ x``.

        Single public primitive for kernel–vector products. Handles explicit
        (dense or sparse ``K``) and implicit (precision matrix + cached LU) cases
        uniformly.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``. Sparse inputs are densified internally because
            ``scipy.linalg.lu_solve`` / ``splu.solve`` require dense RHS and
            ``K @ x`` typically returns dense anyway.

        Returns
        -------
        np.ndarray
            ``(n,)`` if ``x`` was 1D, else ``(n, M)``.

        Examples
        --------
        >>> import numpy as np
        >>> from quadsv import MatrixKernel
        >>> rng = np.random.default_rng(0)
        >>> coords = rng.standard_normal((40, 2))
        >>> kernel = MatrixKernel.from_coordinates(coords, method="matern")
        >>> kernel.Kx(rng.standard_normal(40)).shape
        (40,)
        """
        x_2d, squeeze = self._to_2d(x)
        if sp.issparse(x_2d):
            x_2d = x_2d.toarray()
        # Centered path: HKH x = H · (K · (H x)). Apply H before and after K.
        if self.centering:
            x_2d = x_2d - x_2d.mean(axis=0, keepdims=True)
        y = self._apply_K_dense(x_2d)
        if self.centering:
            y = y - y.mean(axis=0, keepdims=True)
        return y.ravel() if squeeze else y

    def _xtKy_from_Ky(
        self,
        x: np.ndarray | sp.spmatrix,
        Ky: np.ndarray,
        n_cols: int,
    ) -> float | np.ndarray:
        """Given sparse-or-dense ``x`` (``(n, M)``) and dense ``Ky`` (``(n, M)``),
        return the paired diagonal ``sum(x_i * Ky_i, axis=0)``.

        Preserves sparsity of ``x`` — ``x.multiply(Ky).sum(axis=0)`` iterates only
        x's non-zeros. Falls back to ``np.sum(x * Ky, axis=0)`` when ``x`` is dense.
        """
        if sp.issparse(x):
            result = np.asarray(x.multiply(Ky).sum(axis=0)).ravel()
        else:
            result = np.sum(x * Ky, axis=0)
        if n_cols == 1:
            return float(result.item())
        return result

    def xtKy(self, x: np.ndarray | sp.spmatrix, y: np.ndarray | sp.spmatrix) -> float | np.ndarray:
        """
        Bilinear form ``x^T K y`` (paired diagonal for batched inputs).

        For ``(n, M)`` batches returns ``(M,)`` — the diagonal of ``X^T K Y``
        in the same column order, matching :func:`quadsv.spatial_r_test`.
        Sparse ``x`` is preserved; only ``K @ y`` is densified.

        Parameters
        ----------
        x, y : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)`` (must share the M).

        Returns
        -------
        float or np.ndarray
            Scalar if 1D inputs; ``(M,)`` if batched.

        Examples
        --------
        >>> import numpy as np
        >>> from quadsv import MatrixKernel
        >>> rng = np.random.default_rng(0)
        >>> coords = rng.standard_normal((40, 2))
        >>> kernel = MatrixKernel.from_coordinates(coords, method="matern")
        >>> x = rng.standard_normal(40)
        >>> y = rng.standard_normal(40)
        >>> isinstance(kernel.xtKy(x, y), float)
        True
        """
        x_2d, x_squeeze = self._to_2d(x)
        y_2d, y_squeeze = self._to_2d(y)
        squeeze = x_squeeze and y_squeeze
        n_cols = x_2d.shape[1]
        y_dense = y_2d.toarray() if sp.issparse(y_2d) else y_2d
        # x^T HKH y = (H x)^T K (H y). Densify & center both sides; the
        # contraction with x is unchanged.
        if self.centering:
            x_dense = x_2d.toarray() if sp.issparse(x_2d) else x_2d
            x_2d = x_dense - x_dense.mean(axis=0, keepdims=True)
            y_dense = y_dense - y_dense.mean(axis=0, keepdims=True)
        Ky = self._apply_K_dense(y_dense)
        return self._xtKy_from_Ky(x_2d, Ky, 1 if squeeze else n_cols)

    def xtKx(self, x: np.ndarray | sp.spmatrix) -> float | np.ndarray:
        """
        Quadratic form ``x^T K x`` (paired diagonal for batched inputs).

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``. Sparse ``x`` is preserved through the
            final ``x^T (K x)`` contraction — only the right side ``K @ x``
            needs a dense RHS for the solver / BLAS call.

        Returns
        -------
        float or np.ndarray
            Scalar if 1D input, ``(M,)`` if batched.
        """
        x_2d, squeeze = self._to_2d(x)
        n_cols = x_2d.shape[1]
        x_dense = x_2d.toarray() if sp.issparse(x_2d) else x_2d
        # x^T HKH x = (H x)^T K (H x) — center, then reuse the raw
        # x^T K x machinery. The contraction is with the centered x.
        if self.centering:
            x_dense = x_dense - x_dense.mean(axis=0, keepdims=True)
            x_2d = x_dense  # dense post-centering; sparse view no longer valid
        Kx = self._apply_K_dense(x_dense)
        return self._xtKy_from_Ky(x_2d, Kx, 1 if squeeze else n_cols)

    # ------------------------------------------------------------------
    # Sparsity-preserving standardized quadratic form
    # ------------------------------------------------------------------
    def _K_column_sums(self) -> tuple[np.ndarray, float]:
        """Return (``K @ 1_N``, ``1_N^T K 1_N``), computed once and cached.

        Used by :meth:`xtKx_standardized` to evaluate the mean-centering
        correction without densifying sparse inputs.
        """
        cache = getattr(self, "_K_col_sum_cache", None)
        if cache is not None:
            return cache
        ones = np.ones((self.n, 1))
        K_sum = self._apply_K_dense(ones).ravel()  # (n,)
        K_total = float(K_sum.sum())
        self._K_col_sum_cache = (K_sum, K_total)
        return self._K_col_sum_cache

    def xtKx_standardized(
        self,
        x: np.ndarray | sp.spmatrix,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """
        Compute ``z^T K z`` where ``z = (x - means) / stds`` *without* densifying
        sparse ``x``.

        Sparse-aware expansion using the raw-``K`` primitives::

            z^T K z = (x^T K x - 2·μ·(K·𝟏)^T·x + μ²·(𝟏^T K 𝟏)) / σ²

        and the cached row sums of ``K``. This is the fast path for
        standardizing large sparse feature matrices (e.g. scRNA-seq counts)
        before a Q-test.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``. Columns correspond to features.
        means, stds : np.ndarray
            ``(M,)`` per-feature mean and std (``ddof=1`` to match
            :func:`quadsv.statistics.spatial_q_test`).

        Returns
        -------
        np.ndarray
            ``(M,)`` standardized quadratic form values. Columns with
            ``std <= 0`` are returned as zero.
        """
        x_2d, _ = self._to_2d(x)
        n_cols = x_2d.shape[1]
        means = np.asarray(means, dtype=float).reshape(-1)
        stds = np.asarray(stds, dtype=float).reshape(-1)
        if means.shape[0] != n_cols or stds.shape[0] != n_cols:
            raise ValueError(
                f"means/stds shape {means.shape}/{stds.shape} "
                f"inconsistent with x columns ({n_cols})."
            )

        K_sum, K_total = self._K_column_sums()  # K_sum: (n,), K_total: scalar

        # Term 1: RAW ``x^T K x``. Apply K via the dense primitive directly
        # (skips the public :meth:`xtKx` centering wrapper), then contract
        # through :meth:`_xtKy_from_Ky` which preserves x's sparsity.
        x_dense = x_2d.toarray() if sp.issparse(x_2d) else x_2d
        Kx_dense = self._apply_K_dense(x_dense)
        q_raw = np.atleast_1d(np.asarray(self._xtKy_from_Ky(x_2d, Kx_dense, n_cols))).astype(float)

        # Term 2: (K·1)^T x → (M,). x^T @ K_sum, preserving x's sparsity.
        if sp.issparse(x_2d):
            ksum_x = np.asarray(x_2d.T @ K_sum).ravel()
        else:
            ksum_x = x_2d.T @ K_sum

        # Standardized quadratic form
        q_centered = q_raw - 2.0 * means * ksum_x + (means**2) * K_total
        valid = stds > 1e-12
        out = np.zeros(n_cols, dtype=float)
        out[valid] = q_centered[valid] / (stds[valid] ** 2)
        return out

    def _get_rvs_trace_cache(self, n_vectors=15):
        """Generate random vectors for trace estimation caching."""
        if not self.stores_precision:
            raise RuntimeError("Trace caching is only for implicit kernels.")

        # Check if cache exists
        if hasattr(self, "_trace_rvs_cache"):
            if self._trace_rvs_cache["n_vectors"] == n_vectors:
                return self._trace_rvs_cache
            else:
                warnings.warn(
                    "Updating trace random vectors cache with different n_vectors.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Hutchinson's trick random vectors. Use a dedicated, seeded RNG so
        # the estimator is deterministic per-instance (matches NUFFTKernel's
        # convention) and does not perturb numpy's global RNG state — which
        # previously caused flaky order-dependent failures under pytest.
        rvs = np.random.default_rng(0).choice([-1.0, 1.0], size=(self.n, n_vectors)).astype(float)
        # Batched Solve: Solve M * Y = rvs
        # spsolve can handle multiple RHS if passed as dense 2D array
        if sp.issparse(self._K):
            if self._lu is None:
                self._lu = splu(self._K.tocsc())

            Y = self._lu.solve(rvs)
            # Ensure Y is 2D even if n_vectors=1
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
        else:
            if self._lu is None:
                self._lu = lu_factor(self._K)
            Y = lu_solve(self._lu, rvs)

        # Cache for future use
        self._trace_rvs_cache = {"n_vectors": n_vectors, "rvs": rvs, "Y": Y}
        return self._trace_rvs_cache

    def trace(self, n_probes: int | None = None) -> float:
        """``trace(K)`` (raw) or ``trace(HKH)`` (centered).

        Dispatch is automatic based on :attr:`stores_precision`:

        - **Explicit ``K``** (dense or sparse): diagonal sum of the
          stored ``K`` — exact, ``O(n)`` (sparse) or ``O(n)`` (dense).
        - **Precision-stored** (CAR in the implicit regime, ``K`` only
          available through the LU solve on ``K⁻¹``): Hutchinson
          estimator ``(1/m) Σᵢ vᵢᵀ (K vᵢ)`` with cached ``±1`` probes
          solved through the precision.

        ``-s₁/n`` is applied on top when ``centering=True``.

        Parameters
        ----------
        n_probes : int, optional
            Probe count for the Hutchinson path (default 15). **Ignored**
            on the analytic / explicit-``K`` path (no probes involved).
            On the Hutchinson path the cache is keyed on ``n_probes``;
            requesting a different count recomputes the LU-solve cache
            and emits a :class:`RuntimeWarning`.

        Returns
        -------
        float
        """
        if self.stores_precision:
            m = int(n_probes) if n_probes else 15
            cache = self._get_rvs_trace_cache(m)
            raw = float(np.sum(cache["rvs"] * cache["Y"]) / m)
        elif sp.issparse(self._K):
            raw = float(self._K.diagonal().sum())
        else:
            raw = float(np.trace(self._K))
        if not self.centering:
            return raw
        s1, _ = self._ones_stats()
        return raw - s1 / self.n

    def square_trace(self, n_probes: int | None = None) -> float:
        """``trace(K²)`` (raw) or ``trace((HKH)²)`` (centered).

        Dispatch is automatic based on :attr:`stores_precision`:

        - **Explicit ``K``**: Frobenius norm of the stored ``K`` —
          ``Σᵢⱼ Kᵢⱼ²`` in ``O(nnz)`` for sparse, ``O(n²)`` for dense.
        - **Precision-stored**: ``(1/m) Σᵢ ‖K vᵢ‖²`` with the same
          cached ``±1`` probes as :meth:`trace` (single LU solve
          shared across both moments).

        ``-2·s₂/n + s₁²/n²`` is applied on top when ``centering=True``.

        Parameters
        ----------
        n_probes : int, optional
            Probe count for the Hutchinson path (default 15). **Ignored**
            on the analytic / explicit-``K`` path. Shared cache with
            :meth:`trace`.

        Returns
        -------
        float
        """
        if self.stores_precision:
            m = int(n_probes) if n_probes else 15
            cache = self._get_rvs_trace_cache(m)
            raw = float(np.sum(cache["Y"] ** 2) / m)
        elif sp.issparse(self._K):
            raw = float(self._K.power(2).sum())
        else:
            raw = float(np.sum(self._K**2))
        if not self.centering:
            return raw
        s1, s2 = self._ones_stats()
        return max(raw - 2.0 * s2 / self.n + s1**2 / (self.n**2), 0.0)

    def _compute_inv_diag(self, M):
        """Compute diagonal of K = M^{-1} using batched solves to save memory.

        For sparse M, uses batched splu solves on chunks of the identity matrix.
        This avoids allocating a dense n x n inverse matrix.
        """
        n = self.n

        if sp.issparse(M):
            # Factorize once
            if self._lu is None:
                self._lu = splu(M.tocsc())
            lu = self._lu

            # Result array for the diagonal
            diag_vals = np.zeros(n)

            # Determine batch size (100-1000 is usually optimal for cache locality)
            batch_size = 128
            with tqdm(
                total=n,
                desc="Computing diagonal of K",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as pbar:
                for i in range(0, n, batch_size):
                    end = min(i + batch_size, n)
                    current_batch_size = end - i

                    # Create the RHS block: A slice of the Identity matrix.
                    # This corresponds to columns i through end of I.
                    # Shape: (n, current_batch_size)
                    # We construct it directly to save memory.
                    b = np.zeros((n, current_batch_size))

                    # Fill the specific rows that correspond to the diagonal 1s
                    # For the k-th column in this batch (which corresponds to global column i+k),
                    # the 1 is at row i+k.
                    b[i:end, :] = np.eye(current_batch_size)

                    # Solve M * x = b  =>  x = M^{-1} * b
                    x = lu.solve(b)

                    # We only need the diagonal elements of M^{-1}.
                    # In the result x (shape n, batch), the diagonal elements of M^{-1}
                    # are located at x[i, 0], x[i+1, 1], ..., x[end-1, end-1-i]
                    # This corresponds to the diagonal of the square block starting at row i.
                    diag_vals[i:end] = x[i:end, :].diagonal()

                    # Update the progress bar by the actual number of items processed in this batch
                    pbar.update(current_batch_size)

            return diag_vals

        else:
            # Dense case: Direct inversion (slow, should work with kernel matrix directly)
            warnings.warn(
                "Dense precision inversion invoked. Consider using the kernel matrix directly.",
                RuntimeWarning,
                stacklevel=2,
            )
            Minv = inv(M)
            return np.diag(Minv)

    def _standardize_precision(self, M):
        """Scale precision M so that covariance K has unit diagonal, without forming dense K when implicit."""
        diag_K = self._compute_inv_diag(M).copy()
        diag_K[diag_K <= 0] = 1e-12
        s = 1.0 / np.sqrt(diag_K)
        if sp.issparse(M):
            S_inv = sp.diags(1.0 / s)
            return S_inv @ M @ S_inv
        else:
            S_inv = np.diag(1.0 / s)
            return S_inv @ M @ S_inv

    def __getstate__(self):
        """
        Custom pickling behavior: exclude unpicklable SuperLU objects and locks.
        """
        state = self.__dict__.copy()
        # Remove the cached LU factorization because SuperLU objects cannot be pickled.
        # Workers will re-compute this locally.
        state["_lu"] = None
        # Locks cannot be pickled; will be recreated in __setstate__
        state.pop("_lu_lock", None)
        return state

    def __setstate__(self, state):
        """
        Restore state and ensure _lu is reset to None.
        """
        self.__dict__.update(state)
        # Ensure _lu is explicitly None upon restoration
        self._lu = None
        self._lu_lock = threading.Lock()
