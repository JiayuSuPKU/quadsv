from __future__ import annotations

import warnings

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing anything else.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.fft
import spatialdata as sd
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from quadsv.detectors.base import Detector
from quadsv.kernels.fft import FFTKernel
from quadsv.statistics import spatial_q_test
from quadsv.utils import apply_bh_correction

__all__ = ["DetectorGrid"]

logger = logging.getLogger(__name__)


def _qstat_worker_fft(
    raster_layer, feature_batch: list[str], kernel: FFTKernel, return_pval: bool
) -> list[dict]:
    """
    Worker function for parallel Q-statistic computation with FFT kernels.

    Parameters
    ----------
    raster_layer : xarray.DataArray
        Rasterized data layer (lazy Dask array).
    feature_batch : List[str]
        Feature names to process in this batch.
    kernel : FFTKernel
        Pre-constructed FFT kernel object.
    return_pval : bool
        Whether to compute p-values.

    Returns
    -------
    List[dict]
        Results for each feature: {'Feature': str, 'Q': float, 'P_value': float, 'Z_score': float}
    """
    results = []

    # Load data to memory for batch: shape (M, ny, nx)
    data_chunk = raster_layer.sel(c=feature_batch).values
    # Transpose to (ny, nx, M) for kernel
    data_chunk_transposed = np.moveaxis(data_chunk, 0, -1)

    # Compute statistics
    if return_pval:
        stats, pvals = spatial_q_test(data_chunk_transposed, kernel, return_pval=True)
    else:
        stats = spatial_q_test(data_chunk_transposed, kernel, return_pval=False)
        pvals = None

    # Ensure array semantics for iteration (handle 0-d arrays)
    stats = np.atleast_1d(np.asarray(stats))
    if pvals is not None:
        pvals = np.atleast_1d(np.asarray(pvals, dtype=object))
    else:
        pvals = np.array([None] * len(feature_batch), dtype=object)

    # Null parameters
    mu = kernel.trace()
    sigma = np.sqrt(2.0 * kernel.square_trace())
    z_scores = (stats - mu) / sigma if sigma > 1e-12 else np.zeros_like(stats)

    # Format batch results
    for j, gene in enumerate(feature_batch):
        results.append(
            {
                "Feature": gene,
                "Q": float(stats[j]),
                "P_value": float(pvals[j]) if pvals[j] is not None else None,
                "Z_score": float(z_scores[j]),
            }
        )

    return results


class DetectorGrid(Detector):
    r"""
    Detect spatial patterns on **regular grids** (SpatialData bins) with
    FFT-accelerated kernel tests.

    Univariate (Q-test) and bivariate (R-test) kernel-based spatial statistics
    on rasterized :class:`spatialdata.SpatialData` bins.

    Workflow
    --------
    1. **Construct** with kernel method + kernel hyperparameters / grid controls.
    2. **Setup** with :meth:`setup_data` passing the :class:`spatialdata.SpatialData`
       plus the bin / table / col / row keys. Setup rasterizes the table and
       builds the :class:`~quadsv.FFTKernel` at the resulting grid shape.
    3. **Compute** with :meth:`compute_qstat` / :meth:`compute_rstat`.

    Parameters
    ----------
    kernel_method : str, default ``'car'``
        One of ``'gaussian'``, ``'matern'``, ``'moran'``, ``'graph_laplacian'``,
        ``'car'``.
    **kernel_params
        Kernel hyperparameters plus grid controls (``spacing``, ``topology``,
        ``fft_solver``, ``workers``). See :class:`~quadsv.FFTKernel`.

    Attributes
    ----------
    sdata : :class:`spatialdata.SpatialData` or None
        Input container set by :meth:`setup_data`.
    min_count : int or None
        Feature count threshold; set by :meth:`setup_data`.
    kernel\_ : :class:`~quadsv.FFTKernel` or None
        Built in :meth:`setup_data` once the grid shape is known.
    kernel_method\_, kernel_params\_, n
        See :class:`Detector`.

    Examples
    --------
    >>> det = DetectorGrid(kernel_method='car', rho=0.8)
    >>> det.setup_data(sdata, bins='grid', table_name='table',
    ...                col_key='col_idx', row_key='row_idx')  # doctest: +SKIP
    >>> q = det.compute_qstat(features=['Gene_1', 'Gene_2'])  # doctest: +SKIP
    """

    def __init__(self, kernel_method: str = "car", **kernel_params: Any) -> None:
        super().__init__(kernel_method, **kernel_params)

        # Data-state attrs (populated by setup_data):
        self.sdata: sd.SpatialData | None = None
        """Reference to the input :class:`spatialdata.SpatialData`, set by :meth:`setup_data`."""
        self.min_count: int | None = None
        """Minimum total count per feature applied in :meth:`setup_data`."""

        # Rasterization keys (populated by setup_data):
        self._img_key: str | None = None
        self._table_name: str | None = None
        self._bins: str | None = None
        self._col_key: str | None = None
        self._row_key: str | None = None

    def _merge_kernel_defaults(self, method: str, user_params: dict) -> dict:
        """Merge grid-level + per-method FFTKernel defaults with user overrides."""
        general_defaults = {
            "spacing": (1.0, 1.0),
            "topology": "square",
            "fft_solver": "fft2",
            "workers": None,
        }
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"nu": 1.5, "bandwidth": 2.0},
            "moran": {"neighbor_degree": 1},
            "graph_laplacian": {"neighbor_degree": 1},
            "car": {"rho": 0.9, "neighbor_degree": 1},
        }
        defaults = {**general_defaults, **method_defaults.get(method, {})}
        for key, value in user_params.items():
            if key not in defaults:
                raise ValueError(
                    f"Unknown parameter {key!r} for method {method!r}. "
                    f"Allowed: {sorted(defaults)}."
                )
            defaults[key] = value
        return defaults

    def setup_data(
        self,
        sdata: sd.SpatialData,
        *,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        min_count: int | None = None,
    ) -> DetectorGrid:
        """
        Attach ``sdata``, rasterize the chosen bins table, and build the FFTKernel.

        Parameters
        ----------
        sdata : :class:`spatialdata.SpatialData`
            Input container.
        bins : str
            Name of the SpatialElement (Shape) defining the grid-like bins.
        table_name : str
            Name of the table annotating the SpatialElement in ``sdata.tables``.
        col_key, row_key : str
            ``.obs`` columns holding integer column / row indices for the bins.
        value_key : str, optional
            Value column in ``.obs`` to rasterize. ``None`` uses counts / presence.
        min_count : int, optional
            Minimum total count for a feature to pass filtering. ``None`` disables.

        Returns
        -------
        self : DetectorGrid
        """
        self.sdata = sdata
        self.min_count = min_count
        self._bins = bins
        self._table_name = table_name
        self._col_key = col_key
        self._row_key = row_key

        # Rasterize once, store the resulting image key.
        self._img_key = self._rasterize_bins(
            bins=bins,
            table_name=table_name,
            col_key=col_key,
            row_key=row_key,
            value_key=value_key,
        )
        raster_layer = self.sdata[self._img_key]
        _, ny, nx = raster_layer.shape
        self.n = ny * nx

        logger.info(
            "Building FFTKernel (%s) for grid shape (%d, %d)...",
            self.kernel_method_,
            ny,
            nx,
        )
        self.kernel_ = FFTKernel(shape=(ny, nx), method=self.kernel_method_, **self.kernel_params_)
        self._data_ready = True
        return self

    def _rasterize_bins(
        self,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        return_region_as_labels: bool = False,
    ) -> str:
        """
        Wrapper for spatialdata.rasterize_bins with format validation.

        Converts sparse table into a rasterized (grid) image. Ensures CSC sparse
        format for efficient processing and stores result in sdata.images.

        Parameters
        ----------
        bins : str
            Name of the SpatialElement (Shape) which defines the grid-like bins.
        table_name : str
            Name of the table annotating the SpatialElement in sdata.tables.
        col_key : str
            Column in sdata[table_name].obs containing column indices (integers) for bins.
        row_key : str
            Column in sdata[table_name].obs containing row indices (integers) for bins.
        value_key : str, optional
            Column in sdata[table_name].obs to use as pixel values. If None, uses counts/presence.
        return_region_as_labels : bool, default False
            If True, returns bin region masks as integer labels. If False, returns aggregated values.

        Returns
        -------
        img_key : str
            Key under which the rasterized image is stored in sdata.images.
            Format: 'rasterized_{table_name}'.

        Notes
        -----
        This method ensures the underlying matrix is in CSC sparse format for efficient
        column-wise operations required by rasterize_bins.
        """
        from quadsv._rasterize import rasterize_table

        img_key = f"rasterized_{table_name}"
        logger.info("Rasterizing %s into %s...", table_name, img_key)
        rasterized = rasterize_table(
            self.sdata,
            bins=bins,
            table_name=table_name,
            col_key=col_key,
            row_key=row_key,
            value_key=value_key,
            return_region_as_labels=return_region_as_labels,
        )
        self.sdata[img_key] = rasterized
        return img_key

    def _filter_features(self, features, table_name):
        """Helper to validate and filter features based on min_count."""
        # Extract all features from table
        all_features = self.sdata.tables[table_name].var_names.to_list()

        if features is None:
            # Use all features
            valid = all_features
        else:
            valid = [f for f in features if f in all_features]
            if not valid:
                raise ValueError("No valid features found.")

        valid = np.array(valid)

        # Apply min_count filter
        if self.min_count is not None:
            counts = np.asarray(self.sdata.tables[table_name][:, valid].X.sum(axis=0)).ravel()
            valid = valid[counts >= self.min_count]
            if len(valid) == 0:
                raise ValueError(f"No features passed min_count={self.min_count}")

        return valid

    # ------------------------------------------------------------------
    # Auto-tuning helpers
    # ------------------------------------------------------------------
    def _auto_chunk_size(self, budget_bytes: int = 2 * (1 << 30)) -> int:
        """Thin wrapper around :func:`quadsv.statistics.auto_chunk_size`.

        Delegates to the shared helper so the FFT chunk-size policy
        (cache sweet spot of 32, per-feature ``~24·n`` bytes) is kept
        in one place — see :func:`~quadsv.statistics.auto_chunk_size`
        for the full model.
        """
        from quadsv.statistics import auto_chunk_size

        return auto_chunk_size(self.kernel_, budget_bytes=budget_bytes)

    def _auto_schedule(
        self, n_batches: int, n_jobs: int | str, workers: int | str | None
    ) -> tuple[int, int | None]:
        """Balance joblib ``n_jobs`` and scipy.fft ``workers`` to the CPU count.

        Both ``n_jobs`` and ``workers`` parallelize, and stacking them thrashes
        cores. ``'auto'`` policy:

        - If ``n_batches >= cpu_count``: parallelize across batches
          (``n_jobs=cpu_count``), let each FFT call be single-threaded (``workers=1``).
        - Otherwise (few batches, big grids): cap ``n_jobs`` at ``n_batches`` and
          give each worker ``cpu_count / n_jobs`` FFT threads.

        Concrete integers passed by the caller are respected.
        """
        import os

        cpu = os.cpu_count() or 1
        if n_jobs == "auto" or n_jobs == -1:
            if n_batches >= cpu:
                n_jobs_resolved = cpu
                workers_resolved = 1 if workers == "auto" else workers
            else:
                n_jobs_resolved = max(1, n_batches)
                workers_resolved = (
                    max(1, cpu // max(1, n_batches)) if workers == "auto" else workers
                )
        else:
            n_jobs_resolved = int(n_jobs)
            workers_resolved = (
                max(1, cpu // max(1, n_jobs_resolved)) if workers == "auto" else workers
            )
        return n_jobs_resolved, workers_resolved

    def compute_qstat(
        self,
        features: list[str] | None = None,
        n_jobs: int | str = "auto",
        workers: int | str | None = "auto",
        return_pval: bool = True,
        chunk_size: int | str = "auto",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the spatial Q-statistic across features in parallel.

        Requires :meth:`setup_data` to have been called; rasterization and
        kernel construction happen there. This method pulls the rasterized
        feature tensor from :attr:`sdata` and runs per-feature FFT Q-tests.

        Parameters
        ----------
        features : list of str, optional
            Feature names to analyze. ``None`` uses all features that pass
            the ``min_count`` filter from :meth:`setup_data`.
        n_jobs : int or ``'auto'``, default ``'auto'``
            Joblib workers over feature batches. ``'auto'`` balances against
            ``workers`` — see :meth:`_auto_schedule`. ``-1`` is also accepted
            and behaves like ``'auto'``.
        workers : int, ``'auto'``, or None, default ``'auto'``
            Threads for scipy.fft inside each worker. ``'auto'`` co-balances with
            ``n_jobs``; ``None`` defers to scipy's default.
        return_pval : bool, default True
            Whether to compute p-values + Benjamini–Hochberg–adjusted p-values.
        chunk_size : int or ``'auto'``, default ``'auto'``
            Features per worker batch. ``'auto'`` resolves to ``~256 MB / (ny·nx·24)``
            via :meth:`_auto_chunk_size` and clips to ``[16, 1024]``.
        show_progress : bool, default True
            Show a tqdm progress bar over worker chunks.

        Returns
        -------
        pandas.DataFrame
            Indexed by feature. Columns: ``Q``, ``Z_score``, and (if
            ``return_pval=True``) ``P_value``, ``P_adj``. Sorted by ``Q`` desc.
        """
        self._require_setup()
        raster_layer = self.sdata[self._img_key]
        features = self._filter_features(features, self._table_name)

        if isinstance(chunk_size, str):
            if chunk_size != "auto":
                raise ValueError(f"chunk_size must be 'auto' or int, got {chunk_size!r}.")
            chunk_size = self._auto_chunk_size()

        feature_batches = [
            features[i : i + chunk_size] for i in range(0, len(features), chunk_size)
        ]
        n_jobs, workers = self._auto_schedule(len(feature_batches), n_jobs, workers)
        # Let the FFT path pick up the balanced workers setting.
        self.kernel_.workers = workers

        logger.info(
            "Q-test on %d features — %d batches, n_jobs=%d, workers=%s, chunk_size=%d",
            len(features),
            len(feature_batches),
            n_jobs,
            workers,
            chunk_size,
        )

        batch_iter = feature_batches
        if show_progress:
            batch_iter = tqdm(
                feature_batches,
                desc=f"Q ({self.kernel_method_})",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )
        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_qstat_worker_fft)(raster_layer, batch, self.kernel_, return_pval)
            for batch in batch_iter
        )

        # 7. Flatten results
        results = [item for sublist in results_list for item in sublist]

        # 5. Compile results
        df = pd.DataFrame(results).set_index("Feature")
        if not return_pval:
            df = df.drop(columns=["P_value"])

        # 6. Multiple testing correction (Benjamini-Hochberg)
        if return_pval:
            df["P_adj"] = apply_bh_correction(df["P_value"])

        return df.sort_values(by="Q", ascending=False)

    def _compute_batch_spectral_embeddings(self, raster_layer, feature_names):
        """
        Helper: Loads data, standardizes, and computes weighted spectral components.
        Returns matrix of shape (n_features, n_spectral_components).
        """
        # 1. Load Data (IO Bound)
        # Shape: (N_features, Y, X)
        data = raster_layer.sel(c=feature_names).values
        n_feats, ny, nx = data.shape

        # 2. Standardize (In-place to save memory)
        # Mean/Std per feature
        means = np.mean(data, axis=(1, 2), keepdims=True)
        stds = np.std(data, axis=(1, 2), keepdims=True, ddof=1)

        # Avoid div by zero
        stds[stds < 1e-12] = 1.0
        data = (data - means) / stds

        # 3. FFT and Spectral Weighting
        # Use selected FFT solver
        if self.kernel_.fft_solver == "fft2":
            freq_data = scipy.fft.fft2(data, axes=(1, 2), workers=self.kernel_.workers)
            rfft_spectrum = self.kernel_.eigenvalues().reshape(ny, nx)
        else:
            freq_data = scipy.fft.rfft2(data, axes=(1, 2), workers=self.kernel_.workers)
            rfft_spectrum = self.kernel_.eigenvalues().reshape(ny, nx // 2 + 1)

        weights = np.sqrt(np.abs(rfft_spectrum))

        # Broadcast multiply
        weighted_freq = freq_data * weights[None, :, :]

        # 4. Flatten spatial dimensions for matrix multiplication
        # Result: (N_features, n_freq_bins)
        return weighted_freq.reshape(n_feats, -1)

    def compute_rstat(  # noqa: C901
        self,
        features_x: list[str] | None = None,
        features_y: list[str] | None = None,
        return_pval: bool = True,
        chunk_size: int | str = "auto",
        workers: int | str | None = "auto",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the bivariate spatial R-statistic across feature pairs.

        Requires :meth:`setup_data` to have been called.

        Parameters
        ----------
        features_x : list of str, optional
            Features for the X variable. If ``None`` and ``features_y`` is
            ``None``, uses all features (symmetric pairwise mode).
        features_y : list of str, optional
            Features for the Y variable. If ``None``, pairs are drawn from
            ``features_x`` (symmetric, upper-triangular). If provided, returns
            all X × Y pairs (bipartite).
        return_pval : bool, default True
            Whether to compute p-values + Benjamini–Hochberg–adjusted p-values.
        chunk_size : int or ``'auto'``, default ``'auto'``
            Y-features per batch (reuses the pre-computed ``K @ Y`` block).
            ``'auto'`` targets ~256 MB per embedding batch via
            :meth:`_auto_chunk_size`.
        workers : int, ``'auto'``, or None, default ``'auto'``
            Threads for scipy.fft inside the embedding pass. ``'auto'`` gives
            every FFT all CPU cores (the R-test loop is sequential over X/Y
            chunk pairs so there is no joblib contention).
        show_progress : bool, default True
            Show a tqdm progress bar over X chunks.

        Returns
        -------
        pandas.DataFrame
            Columns ``Feature_1``, ``Feature_2``, ``R``, ``Z_score`` and (if
            ``return_pval=True``) ``P_value``, ``P_adj``. Sorted by ``R`` desc.
        """
        import gc  # Garbage collector

        self._require_setup()
        raster_layer = self.sdata[self._img_key]
        table_name = self._table_name
        _, ny, nx = raster_layer.shape

        if isinstance(chunk_size, str):
            if chunk_size != "auto":
                raise ValueError(f"chunk_size must be 'auto' or int, got {chunk_size!r}.")
            chunk_size = self._auto_chunk_size()
        # compute_rstat is sequential across X-chunks, so give every FFT the
        # full CPU budget by default.
        if workers == "auto":
            import os

            workers = os.cpu_count() or 1
        self.kernel_.workers = workers

        # 2. Resolve Features
        all_features = raster_layer.coords["c"].values

        if features_x is None and features_y is None:
            features_x = all_features
            features_y = None
            mode = "symmetric"
        elif features_x is not None and features_y is None:
            mode = "symmetric"
        else:
            mode = "bipartite"

        features_x = self._filter_features(features_x, table_name)
        if mode == "bipartite":
            features_y = self._filter_features(features_y, table_name)

        # 3. Prepare Batches
        chunks_x = np.array_split(features_x, np.ceil(len(features_x) / chunk_size))
        if mode == "bipartite":
            chunks_y = np.array_split(features_y, np.ceil(len(features_y) / chunk_size))
        else:
            chunks_y = chunks_x

        logger.info(
            "Computing R-stats: %d x %d matrix.",
            len(features_x),
            len(features_y) if features_y is not None else len(features_x),
        )
        logger.info("Processing in %d chunks of size ~%d...", len(chunks_x), chunk_size)

        sigma = np.sqrt(self.kernel_.square_trace())
        results_list = []

        # 5. Block Iteration
        x_iter = tqdm(chunks_x, desc="Processing X chunks") if show_progress else chunks_x
        for i, batch_x_names in enumerate(x_iter):
            # Load Embeddings X (High Memory Usage)
            embeddings_x = self._compute_batch_spectral_embeddings(raster_layer, batch_x_names)

            start_j = i if mode == "symmetric" else 0

            for j in range(start_j, len(chunks_y)):
                batch_y_names = chunks_y[j]

                # Load Embeddings Y
                if mode == "symmetric" and i == j:
                    embeddings_y = embeddings_x  # Reference, no copy
                else:
                    embeddings_y = self._compute_batch_spectral_embeddings(
                        raster_layer, batch_y_names
                    )

                # --- CROSS-BATCH CORRELATION ---
                # R_block shape: (chunk_size, chunk_size) -> Very small
                # This step reduces millions of pixels down to a simple correlation number
                R_block = np.matmul(embeddings_x, embeddings_y.conj().T).real

                # Normalize by grid size (rfft2 is unnormalized)
                R_block /= nx * ny

                # --- Format Results ---
                # Create meshgrid of indices for this block
                n_x = len(batch_x_names)
                n_y = len(batch_y_names)

                # Create coordinate grids
                # If symmetric and diagonal block, we only want upper triangle
                if mode == "symmetric" and i == j:
                    # Get upper triangle indices
                    r_idx, c_idx = np.triu_indices(n_x)

                    # Extract values
                    r_vals = R_block[r_idx, c_idx]
                    feat_1 = batch_x_names[r_idx]
                    feat_2 = batch_y_names[c_idx]
                else:
                    # Full block
                    # Flatten the block
                    r_vals = R_block.ravel()
                    # Repeat X names for rows, Tile Y names for cols
                    feat_1 = np.repeat(batch_x_names, n_y)
                    feat_2 = np.tile(batch_y_names, n_x)

                # Store block results
                batch_df = pd.DataFrame({"Feature_1": feat_1, "Feature_2": feat_2, "R": r_vals})

                if return_pval:
                    if sigma > 1e-12:
                        z_scores = r_vals / sigma
                        p_vals = 2 * norm.sf(np.abs(z_scores))
                    else:
                        z_scores = np.zeros_like(r_vals)
                        p_vals = np.ones_like(r_vals)

                    batch_df["Z_score"] = z_scores
                    batch_df["P_value"] = p_vals

                results_list.append(batch_df)

                # Explicit cleanup for Y
                if not (mode == "symmetric" and i == j):
                    del embeddings_y

            # Explicit cleanup for X
            del embeddings_x
            gc.collect()  # Force memory release before next big load

        # 6. Finalize
        if not results_list:
            return pd.DataFrame(columns=["Feature_1", "Feature_2", "R", "P_value", "P_adj"])

        final_df = pd.concat(results_list, ignore_index=True)

        if return_pval and not final_df.empty:
            final_df["P_adj"] = apply_bh_correction(final_df["P_value"])

        return final_df.sort_values(by="R", key=abs, ascending=False)
