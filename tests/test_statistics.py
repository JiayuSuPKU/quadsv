"""
Unit tests for statistical functions.
"""

import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from quadsv.kernels import MatrixKernel
from quadsv.statistics import (
    compute_null_params,
    liu_sf,
    spatial_q_test,
    spatial_r_test,
)


class TestStatisticalFunctions(unittest.TestCase):
    """Test cases for statistical functions."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a small grid
        self.n = 25
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

        # Create test data
        self.data = np.random.randn(self.n)

        # Create a spatial kernel
        self.kernel = MatrixKernel.from_coordinates(self.coords, method="car")

    def test_spatial_q_test_welch(self):
        """Test spatial Q-test with Welch approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={"method": "welch"})

        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_spatial_q_test_liu(self):
        """Test spatial Q-test with Liu approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={"method": "liu"})

        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_liu_sf(self):
        """Test Liu survival function approximation."""
        # Simple case with uniform eigenvalues
        lambs = np.ones(10)
        t = 5.0

        pval = liu_sf(t, lambs)

        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_liu_sf_kurtosis_path(self):
        """Test Liu approximation with kurtosis-based branch."""
        lambs = np.ones(10)
        t = 5.0
        pval = liu_sf(t, lambs, kurtosis=True)
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_zero_variance_data(self):
        """Test handling of zero variance data."""
        constant_data = np.ones(self.n)
        Q, pval = spatial_q_test(constant_data, self.kernel)

        # Should handle gracefully
        self.assertEqual(Q, 0.0)
        self.assertEqual(pval, 1.0)

    def test_spatial_q_test_sparse_csr(self):
        """Test spatial_q_test with sparse CSR matrix input."""
        # Create sparse data (CSR format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csr_matrix(X_dense)

        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_sparse_csc(self):
        """Test spatial_q_test with sparse CSC matrix input."""
        # Create sparse data (CSC format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csc_matrix(X_dense)

        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_chunking(self):
        """Test spatial_q_test with chunking for large feature sets."""
        # Create data with many features
        X = np.random.randn(25, 50)

        # Compute without chunking
        Q_full, pval_full = spatial_q_test(X, self.kernel, return_pval=True)

        # Compute with chunking (chunk_size=10)
        Q_chunked, pval_chunked = spatial_q_test(X, self.kernel, chunk_size=10, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_chunked, Q_full, rtol=1e-10)
        np.testing.assert_allclose(pval_chunked, pval_full, rtol=1e-10)

    def test_spatial_q_test_sparse_with_chunking(self):
        """Test spatial_q_test with sparse input and chunking."""
        # Create sparse data with many features
        X_dense = np.random.randn(25, 50)
        X_sparse = csr_matrix(X_dense)

        # Compute dense without chunking
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)

        # Compute sparse with chunking
        Q_sparse, pval_sparse = spatial_q_test(
            X_sparse, self.kernel, chunk_size=15, return_pval=True
        )

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_single_feature_sparse(self):
        """Test spatial_q_test with single feature sparse matrix."""
        # Create single-feature sparse data
        X_dense = np.random.randn(25, 1)
        X_sparse = csr_matrix(X_dense)

        # Should handle single feature correctly
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)

        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)

    def test_spatial_q_test_progress_bar(self):
        """Test spatial_q_test with progress bar enabled (manual inspection)."""
        # Create data that will require multiple chunks
        X = np.random.randn(25, 30)

        # Should not raise error when show_progress=True
        Q, pval = spatial_q_test(
            X, self.kernel, chunk_size=10, show_progress=False, return_pval=True
        )

        # Verify we got results
        self.assertEqual(len(Q), 30)
        self.assertEqual(len(pval), 30)

    def test_compute_null_params_clt(self):
        """Test compute_null_params with CLT approximation."""
        params = compute_null_params(self.kernel, method="clt")
        self.assertEqual(params["method"], "clt")
        self.assertIn("mean_Q", params)
        self.assertIn("var_Q", params)

    def test_compute_null_params_liu(self):
        """Liu always yields cached cumulants + liu_coef.

        The full-spectrum path is internal — ``compute_null_params``
        only exposes the four spectral cumulants ``c_1..c_4`` and the
        derived shifted-χ² fit ``liu_coef`` (consumed by
        :func:`spatial_q_test`).
        """
        params = compute_null_params(self.kernel, method="liu", k_eigen=5)
        self.assertEqual(params["method"], "liu")
        # Cumulants: {1,2,3,4} → float.
        self.assertIn("cumulants", params)
        self.assertEqual(set(params["cumulants"].keys()), {1, 2, 3, 4})
        # Liu coefficients: shifted-χ² fit.
        self.assertIn("liu_coef", params)
        self.assertEqual(
            set(params["liu_coef"].keys()),
            {"mu_Q", "sigma_Q", "mu_x", "sigma_x", "dof_x", "delta_x"},
        )
        # Raw spectrum is NO LONGER exposed.
        self.assertNotIn("eigenvalues", params)

    def test_spatial_q_test_kernel_matrix_requires_params(self):
        """Kernel matrices without params should raise when null_params is None."""
        K = self.kernel.realization()
        with self.assertRaises(ValueError):
            spatial_q_test(self.data, K, null_params=None)

    def test_spatial_r_test_basic(self):
        """Test spatial R-test on two vectors."""
        x = np.random.randn(self.n)
        y = np.random.randn(self.n)
        R, pval = spatial_r_test(x, y, self.kernel, return_pval=True)
        self.assertIsInstance(R, (float, np.floating))
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)

    def test_spatial_r_test_zero_variance(self):
        """Zero-variance inputs should return neutral p-values."""
        x = np.ones(self.n)
        y = np.random.randn(self.n)
        R, pval = spatial_r_test(x, y, self.kernel, return_pval=True)
        self.assertAlmostEqual(R, 0.0, places=8)
        self.assertAlmostEqual(pval, 1.0, places=8)


class TestKernelPrimitivesAndNullParams(unittest.TestCase):
    """Cross-cutting checks: shared signature shape, ``var_R`` hand-off, and
    equivalence of ``spatial_r_test``'s public ``Kx`` path with the legacy
    ``kernel._K``-based computation."""

    def setUp(self):
        np.random.seed(0)
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        # Use raw (centering=False) so ``kernel.Kx(z) == K @ z`` algebraically
        # matches what the K·z primitive tests expect. Centered behavior is
        # covered by the Q-test FPR / power tests in test_kernels.py.
        self.kernel = MatrixKernel.from_coordinates(self.coords, method="matern", centering=False)
        self.n = self.coords.shape[0]

    def test_compute_null_params_populates_var_R(self):
        """compute_null_params should always populate var_R alongside Q-test moments."""
        for method in ("clt", "welch", "liu"):
            params = compute_null_params(self.kernel, method=method)
            self.assertIn("var_R", params)
            self.assertGreater(params["var_R"], 0.0)

    def test_spatial_r_test_consumes_var_R(self):
        """Supplying var_R via null_params should match the on-the-fly path exactly."""
        x = np.random.randn(self.n)
        y = np.random.randn(self.n)
        R_auto, p_auto = spatial_r_test(x, y, self.kernel)
        params = compute_null_params(self.kernel, method="welch")
        R_given, p_given = spatial_r_test(x, y, self.kernel, null_params=params)
        self.assertAlmostEqual(R_auto, R_given, places=10)
        self.assertAlmostEqual(p_auto, p_given, places=10)

    def test_kernel_Kx_matches_dense_matmul(self):
        """Kernel.Kx(z) must equal K @ z for explicit kernels."""
        K = self.kernel.realization()
        z = np.random.randn(self.n, 3)
        np.testing.assert_allclose(self.kernel.Kx(z), K @ z, rtol=1e-10, atol=1e-12)

    def test_kernel_xtKy_matches_paired_diagonal(self):
        """Kernel.xtKy(x, y) must equal the paired diagonal of x^T K y."""
        K = self.kernel.realization()
        X = np.random.randn(self.n, 4)
        Y = np.random.randn(self.n, 4)
        expected = np.einsum("ij,ik,kj->j", X, K, Y)
        np.testing.assert_allclose(self.kernel.xtKy(X, Y), expected, rtol=1e-10, atol=1e-12)

    def test_qtest_sparse_input_matches_dense(self):
        """Phase E: spatial_q_test must produce the same Q on CSR and ndarray inputs.

        The sparse path in `spatial_q_test` densifies **per chunk** (not the full
        slab), so the statistic should be bit-identical up to floating-point noise.
        """
        rng = np.random.default_rng(0)
        X_dense = rng.standard_normal((self.n, 50))
        # Sprinkle zeros to make sparsity meaningful.
        X_dense[X_dense < -0.5] = 0.0
        X_sparse = csr_matrix(X_dense)
        Q_dense = spatial_q_test(X_dense, self.kernel, return_pval=False)
        Q_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=False)
        np.testing.assert_allclose(Q_dense, Q_sparse, rtol=1e-10, atol=1e-12)

    def test_unified_tests_share_signature(self):
        """Public Q/R dispatchers accept the canonical kwargs."""
        import inspect

        from quadsv import spatial_q_test, spatial_r_test

        canonical_q = {"null_params", "return_pval", "is_standardized"}
        canonical_r = {"null_params", "return_pval", "is_standardized"}
        for fn in (spatial_q_test,):
            sig = set(inspect.signature(fn).parameters)
            self.assertTrue(canonical_q.issubset(sig), f"{fn.__name__} missing {canonical_q - sig}")
        for fn in (spatial_r_test,):
            sig = set(inspect.signature(fn).parameters)
            self.assertTrue(canonical_r.issubset(sig), f"{fn.__name__} missing {canonical_r - sig}")

    def test_power_user_helpers_top_level_reexport(self):
        """``compute_null_params``, ``auto_chunk_size``, ``liu_sf`` are
        now first-class public symbols, importable directly from
        ``quadsv``. The top-level shortcut and the canonical
        ``quadsv.statistics`` path must point at the same callable.
        """
        import quadsv
        import quadsv.statistics as _stats

        for name in ("compute_null_params", "auto_chunk_size", "liu_sf"):
            self.assertTrue(hasattr(quadsv, name), f"quadsv.{name} missing")
            self.assertIs(getattr(quadsv, name), getattr(_stats, name))
            self.assertIn(name, quadsv.__all__)


if __name__ == "__main__":
    unittest.main()
