"""
Tutorial and Integration Tests - Real-World Use Cases

This test module validates all tutorial examples from documentation and README.
Tests are organized by use case to demonstrate typical workflows.
"""

import importlib.util
import unittest

import numpy as np

try:
    import anndata as ad

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

HAS_SPATIALDATA = importlib.util.find_spec("spatialdata") is not None

from quadsv.kernels import MatrixKernel
from quadsv.statistics import compute_null_params, spatial_q_test, spatial_r_test


class TestTutorialBasicQTest(unittest.TestCase):
    """Tutorial: Basic Q-test for spatial variability detection.

    Use case: Test whether a single gene exhibits significant spatial clustering.
    """

    def setUp(self):
        """Set up synthetic spatial data."""
        np.random.seed(42)
        self.n_spots = 500
        # Simulate spot coordinates (e.g., Visium)
        self.coords = np.random.randn(self.n_spots, 2)
        # Simulate gene expression (with spatial structure)
        self.gene_expr = np.random.randn(self.n_spots)

    def test_q_test_basic_workflow(self):
        """Test: Q-test basic workflow from README.

        Step-by-step:
        1. Build CAR kernel from coordinates
        2. Compute Q-statistic and p-value
        3. Verify output format
        """
        # Build CAR kernel (recommended)
        kernel = MatrixKernel.from_coordinates(self.coords, method="car", k_neighbors=15, rho=0.9)

        # Compute Q-test
        Q, pval = spatial_q_test(self.gene_expr, kernel)

        # Verify output types
        self.assertIsInstance(Q, (float, np.floating))
        self.assertIsInstance(pval, (float, np.floating))

        # Verify ranges
        self.assertGreater(Q, 0)  # Q-statistic should be positive
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_q_test_different_null_approximations(self):
        """Test: Q-test with different null approximation methods."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="car", k_neighbors=15, rho=0.9)

        # Test with different null approximations
        methods = ["welch", "liu"]

        for method in methods:
            null_params = compute_null_params(kernel, method=method)
            Q, pval = spatial_q_test(
                self.gene_expr, kernel, null_params=null_params, return_pval=True
            )

            self.assertIsInstance(Q, (float, np.floating))
            self.assertIsInstance(pval, (float, np.floating))
            self.assertGreaterEqual(pval, 0)
            self.assertLessEqual(pval, 1)

    def test_q_test_multiple_genes(self):
        """Test: Q-test on multiple genes (matrix input)."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="car", k_neighbors=15, rho=0.9)

        # Stack multiple genes
        n_genes = 5
        genes_matrix = np.random.randn(self.n_spots, n_genes)

        # Compute Q-statistics for each gene
        Q_values, pvals = spatial_q_test(genes_matrix, kernel, return_pval=True)

        # Verify output
        self.assertEqual(Q_values.shape, (n_genes,))
        self.assertEqual(pvals.shape, (n_genes,))
        self.assertTrue(np.all(Q_values > 0))
        self.assertTrue(np.all(pvals >= 0))
        self.assertTrue(np.all(pvals <= 1))


class TestTutorialRTest(unittest.TestCase):
    """Tutorial: R-test for spatial co-expression detection."""

    def setUp(self):
        """Set up synthetic spatial data."""
        np.random.seed(42)
        self.n_spots = 500
        self.coords = np.random.randn(self.n_spots, 2)
        # Create two genes with spatial co-expression
        self.gene1 = np.random.randn(self.n_spots)
        self.gene2 = np.random.randn(self.n_spots)

    def test_r_test_basic_workflow(self):
        """Test: R-test basic workflow from README.

        Use case: Detect spatial co-expression between two genes.
        """
        kernel = MatrixKernel.from_coordinates(self.coords, method="car", k_neighbors=15, rho=0.9)

        # Compute R-test
        R, pval = spatial_r_test(self.gene1, self.gene2, kernel)

        # Verify output types and ranges
        self.assertIsInstance(R, (float, np.floating))
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_r_test_symmetry(self):
        """Test: R-test symmetry (order of genes shouldn't matter much)."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="car", k_neighbors=15, rho=0.9)

        R1, pval1 = spatial_r_test(self.gene1, self.gene2, kernel)
        R2, pval2 = spatial_r_test(self.gene2, self.gene1, kernel)

        # R should be identical (bilinear form is symmetric)
        self.assertAlmostEqual(R1, R2, places=10)
        # p-values should be identical for this symmetric test
        self.assertAlmostEqual(pval1, pval2, places=10)


@unittest.skipIf(not HAS_ANNDATA, "AnnData not installed")
class TestTutorialAnnDataWorkflow(unittest.TestCase):
    """Tutorial: Genome-wide analysis using AnnData.

    Use case: Detect spatially variable genes (SVGs) in a tissue sample.
    """

    def setUp(self):
        """Create a minimal AnnData object with spatial coordinates."""
        np.random.seed(42)
        n_obs = 500  # cells/spots
        n_vars = 100  # genes

        # Create synthetic count data
        X = np.random.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)

        # Create AnnData object
        self.adata = ad.AnnData(X)
        self.adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
        self.adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

        # Add synthetic spatial coordinates
        self.adata.obsm["spatial"] = np.random.randn(n_obs, 2)

    def test_anndata_integration(self):
        """Test: Basic AnnData integration and kernel building.

        Workflow:
        1. Load AnnData object
        2. Build kernel from obsm['spatial']
        3. Verify kernel properties
        """
        from quadsv.detectors.irregular import DetectorIrregular

        # Initialize detector with kernel config, then attach data
        detector = DetectorIrregular(kernel_method="car", k_neighbors=10, rho=0.9)
        detector.setup_data(self.adata, min_cells_frac=0.05)

        # Verify detector initialized
        self.assertIsNotNone(detector.adata)
        self.assertEqual(detector.adata.n_obs, 500)

        # Verify kernel was built
        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_.n, 500)

    def test_anndata_qstat_computation(self):
        """Test: Compute Q-statistics for subset of genes."""
        from quadsv.detectors.irregular import DetectorIrregular

        detector = DetectorIrregular(kernel_method="car", k_neighbors=10, rho=0.9)
        detector.setup_data(self.adata, min_cells_frac=0.05)

        # Compute Q-statistics for subset of genes
        features = ["Gene_0", "Gene_1", "Gene_2"]
        results = detector.compute_qstat(
            source="var",
            features=features,
            n_jobs=1,
            return_pval=True,  # Single job for testing
        )

        # Verify results structure
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertIn("Q", results.columns)
        self.assertIn("P_value", results.columns)

    def test_anndata_rstat_computation(self):
        """Test: Compute pairwise R-statistics."""
        from quadsv.detectors.irregular import DetectorIrregular

        detector = DetectorIrregular(kernel_method="car", k_neighbors=10, rho=0.9)
        detector.setup_data(self.adata, min_cells_frac=0.05)

        # Test on small subset for speed
        features_x = ["Gene_0", "Gene_1", "Gene_2"]

        results = detector.compute_rstat(
            source="var", features_x=features_x, features_y=features_x, n_jobs=1, return_pval=True
        )

        # Verify results structure
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertIn("R", results.columns)
        self.assertIn("P_value", results.columns)


class TestTutorialFFTKernel(unittest.TestCase):
    """Tutorial: FFT-accelerated kernels for regular grids.

    Use case: Analyze large Visium HD datasets efficiently.
    """

    def setUp(self):
        """Set up synthetic grid data."""
        np.random.seed(42)
        self.grid_shape = (100, 100)  # Smaller for testing
        self.n_spots = np.prod(self.grid_shape)

    def test_fft_kernel_basic(self):
        """Test: Basic FFT kernel construction and Q-test."""
        from quadsv.kernels.fft import FFTKernel

        # Create FFT kernel
        kernel_fft = FFTKernel(shape=self.grid_shape, method="car", rho=0.9, topology="square")

        # Verify properties
        self.assertEqual(kernel_fft.ny, self.grid_shape[0])
        self.assertEqual(kernel_fft.nx, self.grid_shape[1])

        # Generate grid data
        data_grid = np.random.randn(*self.grid_shape)

        # Compute Q-test
        Q = kernel_fft.xtKx(data_grid)
        self.assertGreater(Q, 0)

    def test_fft_kernel_vs_spatial_kernel(self):
        """Test: FFT kernel gives reasonable results vs. spatial kernel.

        For a small regular grid, compare FFT and spatial kernels.
        """
        from quadsv.kernels.fft import FFTKernel

        # Create spatial kernel from grid coordinates
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        xx, yy = np.meshgrid(x, y)
        grid_coords = np.column_stack((xx.ravel(), yy.ravel()))

        kernel_spatial = MatrixKernel.from_coordinates(
            grid_coords,
            method="car",
            k_neighbors=4,
            rho=0.9,  # 4-neighbor grid
        )

        # Create FFT kernel for same grid
        kernel_fft = FFTKernel(shape=(50, 50), method="car", rho=0.9, topology="square")

        # Generate test data
        np.random.seed(42)
        data_1d = np.random.randn(50 * 50)
        data_grid = data_1d.reshape(50, 50)

        # Compare Q-values
        Q_spatial = kernel_spatial.xtKx(data_1d)
        Q_fft = kernel_fft.xtKx(data_grid)

        # Should be in similar ballpark (not identical due to different construction)
        ratio = Q_fft / Q_spatial
        self.assertGreater(ratio, 0.5)  # FFT within 2x of spatial
        self.assertLess(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
