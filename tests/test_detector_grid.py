"""
Unit tests for DetectorGrid class.
Uses lightweight mocks for SpatialData and rasterization to avoid heavy dependencies.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quadsv.detectors.grid import DetectorGrid


class MockCoord:
    def __init__(self, values):
        self.values = values


class MockDataArray:
    """Minimal xarray-like container for tests."""

    def __init__(self, data, features):
        self.data = np.asarray(data)
        self._features = np.asarray(features)
        self.coords = {"c": MockCoord(self._features)}
        self.shape = self.data.shape

    def sel(self, c):
        # Select along feature axis (axis 0)
        idx = [np.where(self._features == f)[0][0] for f in c]
        return MockDataArray(self.data[idx], self._features[idx])

    @property
    def values(self):
        return self.data


class MockVarNames:
    """Mimics AnnData's var_names interface."""

    def __init__(self, names):
        self._names = list(names)

    def to_list(self):
        return self._names


class MockTable:
    """Minimal table with AnnData-like X and feature subset support."""

    def __init__(self, X, features):
        self.X = np.asarray(X)
        self.features = list(features)
        self.var_names = MockVarNames(features)

    def __getitem__(self, keys):
        # Handle 2D indexing: table[:, features] or table[features]
        if isinstance(keys, tuple):
            # 2D indexing: keys = (row_slice, feature_names)
            row_slice, feature_keys = keys
            idx = [self.features.index(k) for k in feature_keys]
            return MockTable(self.X[row_slice, :][:, idx], feature_keys)
        else:
            # 1D indexing: keys = feature_names
            idx = [self.features.index(k) for k in keys]
            return MockTable(self.X[:, idx], keys)


class MockSpatialData:
    """Container that mimics SpatialData for the detector."""

    def __init__(self, table_name, table):
        self.tables = {table_name: table}
        self._store = {}

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value


def _patch_rasterize(mock_da):
    """Patch rasterize_table (used internally by DetectorGrid) to return `mock_da`."""
    return patch("quadsv.detectors.grid.rasterize_table", return_value=mock_da, create=True)


class TestDetectorGrid(unittest.TestCase):
    def setUp(self):
        # Create small synthetic raster data: features x ny x nx
        self.features = np.array(["f1", "f2", "f3"])
        self.ny, self.nx = 2, 2
        self.raster_data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # f1
                [[0.0, 1.0], [0.0, 1.0]],  # f2
                [[2.0, 0.0], [0.0, 1.0]],  # f3
            ]
        )
        self.mock_da = MockDataArray(self.raster_data, self.features)

        # Table counts for min_count filtering (cells x features)
        self.table_X = np.array(
            [
                [10, 0, 10],
                [5, 1, 5],
                [0, 0, 0],
                [3, 0, 2],
            ]
        )
        self.table = MockTable(self.table_X, list(self.features))
        self.sdata = MockSpatialData("cells", self.table)

    def _setup(self, detector, **kwargs):
        """Helper to run setup_data with defaults for our mock sdata."""
        params = {"bins": "bins", "table_name": "cells", "col_key": "col", "row_key": "row"}
        params.update(kwargs)
        return detector.setup_data(self.sdata, **params)

    def test_compute_qstat_returns_dataframe(self):
        """DetectorGrid.compute_qstat produces a DataFrame with expected columns."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            self._setup(detector)
            df = detector.compute_qstat(features=None, n_jobs=1, return_pval=True, chunk_size=2)
        # Expect all features retained
        self.assertEqual(set(df.index), set(self.features))
        # Required columns
        self.assertTrue({"Q", "P_value", "Z_score"}.issubset(df.columns))
        # Adjusted p-value column added when return_pval=True
        self.assertIn("P_adj", df.columns)

    def test_min_count_filtering(self):
        """Features below min_count are filtered out."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            self._setup(detector, min_count=5)
            df = detector.compute_qstat(features=None, n_jobs=1, return_pval=False, chunk_size=2)
        # f2 should be removed due to low count
        self.assertNotIn("f2", df.index)
        self.assertIn("f1", df.index)
        self.assertIn("f3", df.index)

    def test_kernel_reuse_same_shape(self):
        """Kernel is kept across successive compute_qstat calls once setup is done."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            self._setup(detector)
            df1 = detector.compute_qstat(n_jobs=1, return_pval=False)
            kernel_ref = detector.kernel_
            df2 = detector.compute_qstat(n_jobs=1, return_pval=False)
            self.assertIs(detector.kernel_, kernel_ref)
            self.assertEqual(set(df1.index), set(df2.index))

    def test_docstring_example_init_and_workflow(self):
        """Test docstring example: detector initialization and lazy kernel."""
        detector = DetectorGrid(kernel_method="car", rho=0.8)
        self.assertEqual(detector.kernel_method_, "car")
        self.assertEqual(detector.kernel_params_["rho"], 0.8)
        self.assertIsNone(detector.kernel_)  # Lazy — only built in setup_data

    def test_docstring_example_compute_qstat_workflow(self):
        """Test docstring example: full compute_qstat workflow."""
        features = np.array(["f1", "f2", "f3"])
        ny, nx = 4, 4
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        with patch("quadsv._rasterize.rasterize_table", return_value=mock_da):
            detector = DetectorGrid(kernel_method="car", rho=0.8)
            detector.setup_data(
                self.sdata,
                bins="grid",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
            )
            q_results = detector.compute_qstat(
                features=list(features), n_jobs=1, return_pval=True, chunk_size=10
            )

            self.assertIsInstance(q_results, pd.DataFrame)
            self.assertEqual(set(q_results.index), set(features))
            required_cols = {"Q", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(q_results.columns))
            # Q values should be sorted descending
            q_vals = q_results["Q"].values
            self.assertTrue(np.all(q_vals[:-1] >= q_vals[1:]))

    def test_docstring_example_compute_rstat_symmetric(self):
        """Test docstring example: compute_rstat in symmetric (pairwise) mode."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="car", rho=0.8, workers=4)
            detector.setup_data(
                self.sdata,
                bins="grid_bins",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
            )
            r_results = detector.compute_rstat(
                features_x=None, features_y=None, return_pval=True, chunk_size=10
            )
            self.assertEqual(detector.kernel_params_["workers"], 4)

            self.assertIsInstance(r_results, pd.DataFrame)
            required_cols = {"Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(r_results.columns))

            # For 3 features symmetric: C(3,2) + 3 diagonals = 6 pairs
            self.assertEqual(len(r_results), 6)

            r_vals = np.abs(r_results["R"].values)
            self.assertTrue(np.all(r_vals[:-1] >= r_vals[1:]))

    def test_docstring_example_compute_rstat_bipartite(self):
        """Test docstring example: compute_rstat in bipartite (X vs Y) mode."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=2.0, workers=4)
            detector.setup_data(
                self.sdata,
                bins="grid_bins",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
            )
            features_x = ["f1", "f2"]
            features_y = ["f3"]

            r_results = detector.compute_rstat(
                features_x=features_x,
                features_y=features_y,
                return_pval=True,
                chunk_size=10,
            )
            self.assertEqual(detector.kernel_params_["workers"], 4)

            self.assertIsInstance(r_results, pd.DataFrame)
            required_cols = {"Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(r_results.columns))

            # Bipartite: 2 x 1 = 2 pairs
            self.assertEqual(len(r_results), 2)

            self.assertTrue(all(f1 in features_x for f1 in r_results["Feature_1"]))
            self.assertTrue(all(f2 in features_y for f2 in r_results["Feature_2"]))

    def test_compute_qstat_no_pval(self):
        """compute_qstat returns only Q and Z_score when return_pval=False."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            self._setup(detector)
            df = detector.compute_qstat(features=None, n_jobs=1, return_pval=False, chunk_size=2)
            self.assertIn("Q", df.columns)
            self.assertIn("Z_score", df.columns)
            self.assertNotIn("P_value", df.columns)
            self.assertNotIn("P_adj", df.columns)

    def test_compute_rstat_symmetric_upper_triangle(self):
        """compute_rstat symmetric mode returns only upper triangular pairs."""
        features = np.array(["f1", "f2", "f3", "f4"])
        ny, nx = 3, 3
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv._rasterize.rasterize_table", return_value=mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0, workers=1)
            detector.setup_data(
                sdata, bins="bins", table_name="cells", col_key="col", row_key="row"
            )
            r_results = detector.compute_rstat(
                features_x=None, features_y=None, return_pval=True, chunk_size=10
            )

            # For 4 features symmetric: C(4,2) + 4 diagonals = 10 pairs
            self.assertEqual(len(r_results), 10)

            for _, row in r_results.iterrows():
                # Lexicographic ordering ensures upper triangle
                self.assertTrue(row["Feature_1"] <= row["Feature_2"])

    def test_parallel_qstat_single_job(self):
        """compute_qstat with n_jobs=1 produces consistent results."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            self._setup(detector)
            df = detector.compute_qstat(features=None, n_jobs=1, return_pval=True, chunk_size=2)
            self.assertEqual(len(df), len(self.features))
            self.assertTrue(all(np.isfinite(df["Q"])))
            self.assertTrue(all(np.isfinite(df["Z_score"])))

    def test_parallel_qstat_multiple_jobs(self):
        """compute_qstat with n_jobs>1 produces identical results to n_jobs=1."""
        features = np.array([f"gene_{i}" for i in range(20)])
        ny, nx = 8, 8
        np.random.seed(42)
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv._rasterize.rasterize_table", return_value=mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
            detector.setup_data(
                sdata, bins="bins", table_name="cells", col_key="col", row_key="row"
            )

            df_serial = detector.compute_qstat(
                features=None, n_jobs=1, return_pval=True, chunk_size=5
            )

            # Parallel execution reuses the same setup (kernel stays built)
            df_parallel = detector.compute_qstat(
                features=None, n_jobs=2, return_pval=True, chunk_size=5
            )

            self.assertEqual(set(df_serial.index), set(df_parallel.index))

            df_serial = df_serial.sort_index()
            df_parallel = df_parallel.sort_index()

            np.testing.assert_allclose(df_serial["Q"].values, df_parallel["Q"].values, rtol=1e-10)
            np.testing.assert_allclose(
                df_serial["Z_score"].values, df_parallel["Z_score"].values, rtol=1e-10
            )
            np.testing.assert_allclose(
                df_serial["P_value"].values, df_parallel["P_value"].values, rtol=1e-10
            )

    def test_parallel_qstat_workers_parameter(self):
        """DetectorGrid passes `workers` to FFTKernel via kernel_params_."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0, workers=2)
            self._setup(detector)
            df = detector.compute_qstat(features=None, n_jobs=1, return_pval=True, chunk_size=2)

            self.assertIsNotNone(detector.kernel_)
            self.assertEqual(detector.kernel_params_["workers"], 2)
            self.assertEqual(len(df), len(self.features))
            self.assertTrue(all(np.isfinite(df["Q"])))

    def test_parallel_qstat_chunk_size_effect(self):
        """Different chunk_size values produce identical results."""
        features = np.array([f"gene_{i}" for i in range(12)])
        ny, nx = 6, 6
        np.random.seed(123)
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv._rasterize.rasterize_table", return_value=mock_da):
            detector = DetectorGrid(kernel_method="car", rho=0.8)
            detector.setup_data(
                sdata, bins="bins", table_name="cells", col_key="col", row_key="row"
            )

            df_small = detector.compute_qstat(
                features=None, n_jobs=1, return_pval=True, chunk_size=3
            )

            df_large = detector.compute_qstat(
                features=None, n_jobs=1, return_pval=True, chunk_size=10
            )

            df_small = df_small.sort_index()
            df_large = df_large.sort_index()

            np.testing.assert_allclose(df_small["Q"].values, df_large["Q"].values, rtol=1e-10)
            np.testing.assert_allclose(
                df_small["P_value"].values, df_large["P_value"].values, rtol=1e-10
            )

    def test_parallel_qstat_n_jobs_auto(self):
        """compute_qstat with n_jobs=-1 uses all available cores."""
        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            with patch("os.cpu_count", return_value=4):
                detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.0)
                self._setup(detector)
                df = detector.compute_qstat(
                    features=None, n_jobs=-1, return_pval=True, chunk_size=2
                )
                self.assertEqual(len(df), len(self.features))
                self.assertTrue(all(np.isfinite(df["Q"])))


class TestDetectorGridStatistic(unittest.TestCase):
    """``DetectorGrid.compute_qstat`` is ``spatial_q_test`` on the rasterized
    data with per-feature z-scoring across the grid. The identity must hold to
    float-precision (~1e-10) because both paths call the same FFTKernel on
    the same standardized inputs."""

    def setUp(self):
        features = np.array([f"g{i}" for i in range(4)])
        ny, nx = 8, 8
        rng = np.random.default_rng(0)
        self.features = features
        self.ny, self.nx = ny, nx
        # Structured first feature, noise for the rest — just so Q values
        # span a useful range for the equality comparison.
        raster = rng.standard_normal((len(features), ny, nx))
        yy = np.arange(ny)[:, None]
        raster[0] += np.sin(2 * np.pi * yy / 4.0)
        self.raster_data = raster
        self.mock_da = MockDataArray(raster, features)

        table_X = np.ones((16, len(features)), dtype=float)  # min_count passes
        self.table = MockTable(table_X, list(features))
        self.sdata = MockSpatialData("cells", self.table)

    def _setup(self, detector, **kwargs):
        params = {"bins": "bins", "table_name": "cells", "col_key": "col", "row_key": "row"}
        params.update(kwargs)
        return detector.setup_data(self.sdata, **params)

    def test_qstat_matches_raw_fftkernel_spatial_q_test(self):
        """``DetectorGrid.compute_qstat`` must equal a raw
        ``spatial_q_test(FFTKernel, z_scored_grid)`` — same FFTKernel, same
        per-feature grid z-score, so the gap is pure float-precision."""
        from quadsv.statistics import spatial_q_test

        with patch("quadsv._rasterize.rasterize_table", return_value=self.mock_da):
            detector = DetectorGrid(kernel_method="gaussian", bandwidth=1.5)
            self._setup(detector)
            df = detector.compute_qstat(n_jobs=1, return_pval=False, show_progress=False)

        # Reproduce the detector's internal prep:
        # - load (n_feats, ny, nx) raster, move axis to (ny, nx, n_feats),
        # - let spatial_q_test z-score per feature (is_standardized=False).
        raster_tyx = np.moveaxis(self.raster_data, 0, -1)  # (ny, nx, n_feats)
        Q_raw = np.asarray(spatial_q_test(raster_tyx, detector.kernel_, return_pval=False))
        # ``df`` is sorted by Q desc; align by feature name.
        Q_from_detector = df.loc[list(self.features), "Q"].to_numpy()
        np.testing.assert_allclose(Q_from_detector, Q_raw, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
