"""
Unit tests for DetectorIrregular class.
"""

import unittest
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from quadsv.detectors.irregular import DetectorIrregular


class TestDetectorIrregular(unittest.TestCase):
    """Test cases for DetectorIrregular class."""

    def setUp(self):
        """Set up test fixtures with AnnData."""
        np.random.seed(42)

        # Create AnnData object
        self.n_obs = 100
        self.n_vars = 50

        # Create spatial coordinates
        x = np.linspace(0, 4, int(np.sqrt(self.n_obs)))
        y = np.linspace(0, 4, int(np.sqrt(self.n_obs)))
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack((xx.ravel(), yy.ravel()))[: self.n_obs]

        # Create expression matrix
        X = np.random.randn(self.n_obs, self.n_vars)
        X[X < 0] = 0  # Make it non-negative like counts

        # Create obs dataframe with some categorical and numeric features
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(
                    ["A", "B", "C"] * (self.n_obs // 3) + ["A"] * (self.n_obs % 3)
                ),
                "numeric_feature": np.random.randn(self.n_obs),
                "batch": pd.Categorical(
                    ["batch1", "batch2"] * (self.n_obs // 2) + ["batch1"] * (self.n_obs % 2)
                ),
            },
            index=[f"Cell_{i}" for i in range(self.n_obs)],
        )

        # Create var dataframe
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(self.n_vars)])

        # Create obsp with connectivity matrix
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(coords)
        W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)

        # Create layers
        layers = {"normalized": X + np.random.randn(self.n_obs, self.n_vars) * 0.1}

        # Create real AnnData object
        self.adata = anndata.AnnData(
            X=sp.csr_matrix(X),
            obs=obs,
            var=var,
            obsm={"spatial": coords},
            obsp={"connectivities": W},
            layers=layers,
        )

        # Store coords for later use in tests
        self.coords = coords

    def test_docstring_example_workflow(self):
        """Test the workflow shown in class docstring."""
        # Example workflow from docstring
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=2.0)
        detector.setup_data(self.adata, min_cells=10)

        # Should be able to compute qstat
        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 2.0)

    def test_class_initialization(self):
        """Test DetectorIrregular initialization."""
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.0)

        # Before setup_data, n is None
        self.assertIsNone(detector.n)
        self.assertIsNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")

        detector.setup_data(self.adata, min_cells=5)
        self.assertEqual(detector.n, self.n_obs)
        self.assertIsNotNone(detector.kernel_)

    def test_available_kernels(self):
        """Test that available kernels are properly defined."""
        detector = DetectorIrregular(kernel_method="gaussian")

        expected_kernels = ("gaussian", "matern", "moran", "graph_laplacian", "car")
        self.assertEqual(detector._available_kernels, expected_kernels)

    def test_build_kernel_from_coordinates_docstring_example(self):
        """Test example from setup_data docstring with custom obsm coords."""
        coords = np.random.randn(self.n_obs, 2)
        adata2 = self.adata.copy()
        adata2.obsm["spatial"] = coords

        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.5)
        detector.setup_data(adata2, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 1.5)

    def test_build_kernel_from_coordinates_gaussian(self):
        """Test building Gaussian kernel from coordinates via setup_data."""
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.0)
        detector.setup_data(self.adata, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 1.0)

    def test_build_kernel_from_coordinates_matern(self):
        """Test building Matérn kernel from coordinates via setup_data."""
        detector = DetectorIrregular(kernel_method="matern", bandwidth=1.0, nu=1.5)
        detector.setup_data(self.adata, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "matern")
        self.assertEqual(detector.kernel_params_["nu"], 1.5)

    def test_build_kernel_from_coordinates_moran(self):
        """Test building Moran kernel from coordinates via setup_data."""
        detector = DetectorIrregular(kernel_method="moran", k_neighbors=8)
        detector.setup_data(self.adata, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "moran")
        self.assertEqual(detector.kernel_params_["k_neighbors"], 8)

    def test_build_kernel_from_coordinates_car(self):
        """Test building CAR kernel from coordinates via setup_data."""
        detector = DetectorIrregular(kernel_method="car", k_neighbors=8, rho=0.9)
        detector.setup_data(self.adata, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "car")
        self.assertEqual(detector.kernel_params_["rho"], 0.9)

    def test_build_kernel_invalid_method(self):
        """Test that invalid kernel method raises error at construction."""
        with self.assertRaises(ValueError) as context:
            DetectorIrregular(kernel_method="invalid_method")

        self.assertIn("kernel_method must be", str(context.exception))

    def test_build_kernel_coordinate_shape_mismatch(self):
        """Test that coordinate shape mismatch raises error in setup_data."""
        # Put bad coords in adata.obsm["spatial"] to trigger shape check.
        adata2 = self.adata.copy()
        adata2.obsm["spatial"] = np.random.randn(self.n_obs, 3)  # wrong ndim

        detector = DetectorIrregular(kernel_method="gaussian")
        with self.assertRaises(ValueError) as context:
            detector.setup_data(adata2, min_cells=5)

        self.assertIn("(n_obs, 2)", str(context.exception))

    # NOTE: test_build_kernel_from_obsp_precomputed was removed because the new
    # API rejects kernel_method='precomputed' at construction — the precomputed
    # obsp path is no longer user-reachable through DetectorIrregular.

    def test_build_kernel_from_obsp_moran(self):
        """Test building Moran kernel from connectivity matrix via setup_data."""
        detector = DetectorIrregular(kernel_method="moran")
        detector.setup_data(self.adata, obsp_key="connectivities", is_distance=False, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "moran")

    def test_build_kernel_from_obsp_graph_laplacian(self):
        """Test building Laplacian kernel from connectivity matrix via setup_data."""
        detector = DetectorIrregular(kernel_method="graph_laplacian")
        detector.setup_data(self.adata, obsp_key="connectivities", is_distance=False, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "graph_laplacian")

    def test_build_kernel_from_obsp_car(self):
        """Test building CAR kernel from connectivity matrix via setup_data."""
        detector = DetectorIrregular(kernel_method="car", rho=0.8)
        detector.setup_data(self.adata, obsp_key="connectivities", is_distance=False, min_cells=5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "car")

    def test_build_kernel_from_obsp_car_standardize(self):
        """Precomputed CAR with standardize → raw K has unit diagonal.

        We flip ``detector.kernel_.centering = False`` so ``realization()``
        returns raw ``K`` (not ``HKH``) for this diagonal check — the
        standardize contract is a property of ``K`` itself.
        """
        detector = DetectorIrregular(kernel_method="car", rho=0.9, standardize=True)
        detector.setup_data(self.adata, obsp_key="connectivities", is_distance=False, min_cells=5)
        detector.kernel_.centering = False
        K = detector.kernel_.realization()
        diag = np.diag(K)
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5, atol=1e-3)

    def test_build_kernel_from_obsp_missing_key(self):
        """Test that missing obsp key raises error in setup_data."""
        detector = DetectorIrregular(kernel_method="moran")

        with self.assertRaises(KeyError) as context:
            detector.setup_data(self.adata, obsp_key="nonexistent_key", min_cells=5)

        self.assertIn("not found", str(context.exception))

    def test_build_kernel_from_obsp_invalid_method(self):
        """Test that invalid method for obsp raises error.

        NOTE: invalid kernel_method is caught at construction now, so this test
        rewords the intent: construction itself raises.
        """
        with self.assertRaises(ValueError) as context:
            DetectorIrregular(kernel_method="invalid_method")

        self.assertIn("kernel_method must be", str(context.exception))

    def test_prepare_data_var_source(self):
        """Test data preparation from var (genes)."""
        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(self.adata, min_cells=5)

        # Prepare data for a subset of genes
        gene_subset = self.adata.var_names[:5].tolist()
        X_csc, names, means, stds = detector._prepare_data(
            source="var", keys=gene_subset, min_cells=1
        )

        # Check shapes
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertLessEqual(len(names), 5)

        # Check that means and stds are properly computed
        self.assertEqual(len(means), len(names))
        self.assertEqual(len(stds), len(names))

        # Check all stds are positive (constant features filtered)
        self.assertTrue(np.all(stds > 0))

    def test_prepare_data_obs_numeric(self):
        """Test data preparation from obs with numeric features."""
        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(self.adata, min_cells=5)

        X_csc, names, means, stds = detector._prepare_data(
            source="obs", keys=["numeric_feature"], min_cells=1
        )

        # Check shapes
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], "numeric_feature")

    def test_prepare_data_obs_categorical(self):
        """Test data preparation from obs with categorical features (one-hot encoding)."""
        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(self.adata, min_cells=5)

        X_csc, names, means, stds = detector._prepare_data(
            source="obs", keys=["cell_type"], min_cells=1
        )

        # Check that one-hot encoding occurred
        # cell_type has 3 categories, so we expect 3 columns
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertEqual(len(names), 3)  # One for each category

        # Check that names contain category values
        self.assertTrue(any("A" in name or "B" in name or "C" in name for name in names))

    def test_prepare_data_no_keys_obs_raises(self):
        """Test that not providing keys for obs source raises error."""
        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector._prepare_data(source="obs", keys=None, min_cells=1)

        self.assertIn("Keys must be provided", str(context.exception))

    def test_prepare_data_invalid_source(self):
        """Test that invalid source raises error."""
        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector._prepare_data(source="invalid", keys=None, min_cells=1)

        self.assertIn("Source must be", str(context.exception))

    def test_run_without_kernel_raises(self):
        """Test that running DetectorIrregular without setup_data raises error."""
        detector = DetectorIrregular(kernel_method="gaussian")

        with self.assertRaises((ValueError, RuntimeError)) as context:
            detector.compute_qstat(source="var", features=None)

        # The new error path raises from _require_setup OR from the "Kernel not initialized"
        # guard. Either message is acceptable.
        msg = str(context.exception)
        self.assertTrue(
            "setup_data" in msg or "Kernel not initialized" in msg,
            f"Unexpected error message: {msg!r}",
        )

    def test_compute_qstat_docstring_example(self):
        """Test compute_qstat example from docstring."""
        # Example from docstring: setup then compute qstat
        detector = DetectorIrregular(kernel_method="matern")
        detector.setup_data(self.adata, min_cells=5)

        # Select two genes as in docstring example
        gene_subset = self.adata.var_names[:2].tolist()
        results = detector.compute_qstat(
            source="var",
            features=gene_subset,
            n_jobs=1,  # Use 1 job for deterministic testing
        )

        # Check structure matches docstring
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("Q", results.columns)
        self.assertIn("Z_score", results.columns)
        self.assertIn("P_value", results.columns)
        self.assertIn("P_adj", results.columns)

        # Results should be sorted by Q descending
        self.assertTrue(results["Q"].is_monotonic_decreasing or len(results) <= 1)

        # Check we can get top genes
        top_genes = results.iloc[: min(10, len(results))]
        self.assertLessEqual(len(top_genes), 10)

    def test_run_var_source(self):
        """Test running DetectorIrregular on var (genes) source without mocks."""
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.0)
        detector.setup_data(self.adata, min_cells=5)

        # Run DetectorIrregular on a subset of genes
        # We select enough genes to likely trigger batch processing if batch size is small
        gene_subset = self.adata.var_names[:10].tolist()

        # Use n_jobs=1 to keep debugging simple, but this will exercise the full math stack
        results = detector.compute_qstat(source="var", features=gene_subset, n_jobs=1)

        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("Q" in results.columns)
        self.assertTrue("P_value" in results.columns)
        self.assertTrue("Z_score" in results.columns)

        # Check value ranges for real computation
        self.assertTrue(np.all(results["Q"] >= 0))
        self.assertTrue(np.all(results["P_value"] >= 0))
        self.assertTrue(np.all(results["P_value"] <= 1.0))

        # Check that results are sorted by Q descending
        self.assertTrue(results["Q"].is_monotonic_decreasing)

    def test_run_obs_source(self):
        """Test running DetectorIrregular on obs (cell metadata) source without mocks."""
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.0)
        detector.setup_data(self.adata, min_cells=5)

        # Run DetectorIrregular on obs features
        results = detector.compute_qstat(source="obs", features=["numeric_feature"], n_jobs=1)

        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue(len(results) > 0)
        self.assertTrue(np.all(results["P_value"] >= 0))
        self.assertTrue(np.all(results["P_value"] <= 1.0))

    def test_run_with_layer(self):
        """Test running DetectorIrregular with specific layer."""
        detector = DetectorIrregular(kernel_method="gaussian", bandwidth=1.0)
        detector.setup_data(self.adata, min_cells=5)

        # Mock the compute_null_params to avoid actual computation
        with patch("quadsv.statistics.compute_null_params") as mock_compute_null:
            with patch("quadsv.statistics.spatial_q_test") as mock_spatial_q_test:
                mock_compute_null.return_value = {"mean_Q": 1.0, "var_Q": 0.5}
                mock_spatial_q_test.return_value = (2.5, 0.05)

                gene_subset = self.adata.var_names[:3].tolist()
                results = detector.compute_qstat(
                    source="var", features=gene_subset, layer="normalized", n_jobs=1
                )

                self.assertIsInstance(results, pd.DataFrame)

    def test_compute_rstat_docstring_symmetric_example(self):
        """Test compute_rstat symmetric mode example from docstring."""
        # Example from docstring: all pairwise correlations within gene set
        detector = DetectorIrregular(kernel_method="matern")
        detector.setup_data(self.adata, min_cells=5)

        # Select 3 genes for symmetric pairwise
        gene_subset = self.adata.var_names[:3].tolist()
        results = detector.compute_rstat(features_x=gene_subset, n_jobs=1)

        # Check structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("Feature_1", results.columns)
        self.assertIn("Feature_2", results.columns)
        self.assertIn("R", results.columns)
        self.assertIn("Z_score", results.columns)
        self.assertIn("P_value", results.columns)
        self.assertIn("P_adj", results.columns)

        # Results should be sorted by |Z_score| descending
        abs_z = results["Z_score"].abs()
        self.assertTrue(abs_z.is_monotonic_decreasing or len(results) <= 1)

        # Symmetric mode now returns all pairs including (A,A), (A,B), (B,A), etc.
        # For 3 genes, we expect up to 9 pairs
        self.assertLessEqual(len(results), 9)

    def test_compute_rstat_docstring_bipartite_example(self):
        """Test compute_rstat bipartite mode example from docstring."""
        # Example from docstring: cross-correlation between two gene sets
        detector = DetectorIrregular(kernel_method="matern")
        detector.setup_data(self.adata, min_cells=5)

        # Select genes as in docstring: 2 x 2 = 4 pairs
        features_x = self.adata.var_names[:2].tolist()
        features_y = self.adata.var_names[2:4].tolist()

        results = detector.compute_rstat(features_x=features_x, features_y=features_y, n_jobs=1)

        # Check structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("R", results.columns)
        self.assertIn("Z_score", results.columns)

        # Bipartite mode: all X vs Y pairs
        # 2 x 2 = 4 pairs maximum
        self.assertLessEqual(len(results), 4)


class TestDetectorIrregularEdgeCases(unittest.TestCase):
    """Test edge cases for DetectorIrregular class."""

    def test_compute_with_constant_features(self):
        """Test that constant features are filtered out."""
        # Create mock adata with some constant features
        n_obs = 50
        n_vars = 10

        X = np.random.randn(n_obs, n_vars)
        X[:, 0] = 1.0  # Constant feature
        X[:, 1] = 0.0  # Zero feature
        coords = np.random.randn(n_obs, 2)

        adata = anndata.AnnData(
            X=sp.csr_matrix(X),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
            obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
            obsm={"spatial": coords},
        )

        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(adata, min_cells=1)
        X_csc, names, means, stds = detector._prepare_data(source="var", keys=None, min_cells=1)

        # Constant features should be filtered
        self.assertLess(len(names), n_vars)
        self.assertTrue(np.all(stds > 0))

    def test_compute_with_sparse_features(self):
        """Test DetectorIrregular with very sparse features."""
        n_obs = 100
        n_vars = 20

        # Create very sparse matrix with numeric dtype
        X = sp.random(n_obs, n_vars, density=0.05, format="csr", dtype=np.float64)
        coords = np.random.randn(n_obs, 2)

        adata = anndata.AnnData(
            X=X,
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
            obsm={"spatial": coords},
        )

        detector = DetectorIrregular(kernel_method="gaussian")
        detector.setup_data(adata, min_cells=5)

        # Should handle sparse data gracefully
        X_csc, names, means, stds = detector._prepare_data(
            source="var",
            keys=None,
            min_cells=10,  # Require at least 10 non-zero cells
        )

        self.assertIsInstance(X_csc, sp.csc_matrix)
        self.assertTrue(len(names) <= n_vars)


class TestDetectorIrregularBackendParity(unittest.TestCase):
    """``backend='matrix'`` and ``backend='nufft'`` target the same underlying
    Q-statistic but use different null-distribution approximations (Welch
    from the full matrix spectrum vs. analytic FFT-kernel moments rescaled by
    ``n/n'``). Exact Q values differ, but the *ranking* of structured vs.
    noise features should agree."""

    def setUp(self):
        import anndata

        rng = np.random.default_rng(0)
        n_spots = 300
        coords = rng.uniform(0, 20, size=(n_spots, 2))
        n_genes = 6
        X = rng.standard_normal((n_spots, n_genes))
        # g0 carries a clean sine pattern along the y-axis; the rest are iid.
        X[:, 0] = np.sin(2 * np.pi * coords[:, 0] / 5.0) + 0.3 * rng.standard_normal(n_spots)
        X = np.maximum(X + 3.0, 0.0)

        self.adata = anndata.AnnData(
            X=X,
            var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
            obsm={"spatial": coords},
        )

    def test_matrix_vs_nufft_rank_agreement_on_structured_gene(self):
        """Both backends rank ``g0`` first and agree on the broader Q
        ordering. Spearman ≥ 0.7: two distinct null approximations + matrix
        vs. grid operator means exact Q differs, but the *ordering* of a
        single structured gene against 5 iid noise genes should agree."""
        from scipy.stats import spearmanr

        det_m = DetectorIrregular(kernel_method="matern", backend="matrix", bandwidth=2.0, nu=1.5)
        det_m.setup_data(self.adata, min_cells=5)
        df_m = det_m.compute_qstat(n_jobs=1, show_progress=False)

        det_n = DetectorIrregular(kernel_method="matern", backend="nufft", bandwidth=2.0, nu=1.5)
        det_n.setup_data(self.adata, min_cells=5)
        df_n = det_n.compute_qstat(n_jobs=1, show_progress=False)

        # Normalize: the matrix path indexes by Feature; the NUFFT path keeps
        # it as a column. Re-index both so the test doesn't depend on that.
        def _by_feature(df):
            return df.set_index("Feature") if "Feature" in df.columns else df

        df_m = _by_feature(df_m)
        df_n = _by_feature(df_n)

        # Both are sorted Q descending → first row is the top-ranked gene.
        self.assertEqual(df_m.index[0], "g0")
        self.assertEqual(df_n.index[0], "g0")
        self.assertLess(df_m.loc["g0", "P_value"], 0.05)
        self.assertLess(df_n.loc["g0", "P_value"], 0.05)

        common = sorted(set(df_m.index) & set(df_n.index))
        q_m = df_m.loc[common, "Q"].to_numpy()
        q_n = df_n.loc[common, "Q"].to_numpy()
        rho, _ = spearmanr(q_m, q_n)
        self.assertGreaterEqual(rho, 0.7, msg=f"Spearman(Q_matrix, Q_nufft) = {rho:.2f}")


if __name__ == "__main__":
    unittest.main()
