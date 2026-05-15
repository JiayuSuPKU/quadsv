"""
Shared base class for single-sample spatial pattern detectors.

Concrete detectors follow a three-step workflow:

1. **Construction** — :meth:`Detector.__init__` takes kernel method + backend
   configs + kernel hyperparameters. No data is attached.
2. **Data setup** — :meth:`Detector.setup_data` takes the input container
   (:class:`anndata.AnnData` for :class:`DetectorIrregular`,
   :class:`spatialdata.SpatialData` for :class:`DetectorGrid`), performs
   preprocessing (feature filtering, coordinate / obsp extraction, or
   rasterization), and builds the kernel.
3. **Computation** — :meth:`Detector.compute_qstat` and
   :meth:`Detector.compute_rstat` take feature selections + compute-time knobs
   (``n_jobs``, ``chunk_size``, etc.) and return per-feature results.

The base class owns the attribute contract (``kernel_method_``,
``kernel_params_``, ``kernel_``, ``n``, ``_data_ready``) and enforces the
workflow via :meth:`_require_setup`. Concrete subclasses implement
:meth:`_merge_kernel_defaults`, :meth:`setup_data`, :meth:`compute_qstat`,
and :meth:`compute_rstat`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quadsv.kernels import Kernel

__all__ = ["Detector"]


class Detector(ABC):
    r"""
    Abstract base for single-sample pattern detectors.

    Attributes
    ----------
    kernel_method\_ : str
        Kernel method name (e.g. ``'matern'``, ``'car'``). Set at construction.
    kernel_params\_ : dict
        Resolved kernel parameters after backend-specific defaults are merged
        with user overrides. Set at construction.
    kernel\_ : :class:`~quadsv.kernels.Kernel` or None
        Kernel object built in :meth:`setup_data`. ``None`` before data setup.
    n : int or None
        Effective number of observations after preprocessing. ``None`` before
        data setup.
    """

    _available_kernels: tuple[str, ...] = (
        "gaussian",
        "matern",
        "moran",
        "graph_laplacian",
        "car",
    )

    def __init__(self, kernel_method: str, **kernel_params: Any) -> None:
        if kernel_method not in self._available_kernels:
            raise ValueError(
                f"kernel_method must be one of {self._available_kernels}, "
                f"got {kernel_method!r}."
            )
        self.kernel_method_: str = kernel_method
        self.kernel_params_: dict = self._merge_kernel_defaults(kernel_method, dict(kernel_params))
        self.kernel_: Kernel | None = None
        self.n: int | None = None
        self._data_ready: bool = False

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _merge_kernel_defaults(self, method: str, user_params: dict) -> dict:
        """Return ``{default: value}`` after merging user overrides.

        Subclasses must validate unknown keys and fill in backend-specific
        defaults (e.g. ``fft_solver`` / ``spacing`` / ``topology`` for
        :class:`DetectorGrid`; ``k_neighbors`` vs ``neighbor_degree`` for
        the matrix vs NUFFT backends of :class:`DetectorIrregular`).
        """

    @abstractmethod
    def setup_data(self, data: Any, **kwargs: Any) -> Detector:
        """Attach ``data``, preprocess features, and build :attr:`kernel_`.

        Must set :attr:`kernel_`, :attr:`n`, and ``self._data_ready = True``
        before returning ``self``.
        """

    @abstractmethod
    def compute_qstat(self, features: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Univariate Q-test across ``features``."""

    @abstractmethod
    def compute_rstat(self, **kwargs: Any) -> pd.DataFrame:
        """Bivariate R-test. Signature / feature selection are subclass-specific."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _require_setup(self) -> None:
        """Raise if :meth:`setup_data` has not been called."""
        if not self._data_ready or self.kernel_ is None:
            raise RuntimeError(f"{type(self).__name__}: call setup_data(...) before compute_*().")

    def __repr__(self) -> str:
        state = "ready" if self._data_ready else "not set up"
        return (
            f"<{type(self).__name__} kernel_method={self.kernel_method_!r} "
            f"state={state} n={self.n}>"
        )
