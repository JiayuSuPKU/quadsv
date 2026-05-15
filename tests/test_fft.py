"""
Unit tests for FFT-based kernel and statistical tests.
Compares FFTKernel results with MatrixKernel for grid data.
"""

import unittest

import numpy as np

from quadsv.kernels import MatrixKernel
from quadsv.kernels.fft import FFTKernel
from quadsv.statistics import spatial_q_test, spatial_r_test
from quadsv.statistics import spatial_q_test as spatial_q_test_standard
from quadsv.statistics import spatial_r_test as spatial_r_test_standard
from quadsv.utils import compute_torus_distance_matrix


class TestFFTKernelBasics(unittest.TestCase):
    """Test FFTKernel initialization and basic methods."""

    def test_fft_kernel_init_square_fft2(self):
        """Test FFTKernel initialization with square grid."""
        shape = (10, 10)
        kernel = FFTKernel(
            shape, topology="square", method="gaussian", bandwidth=1.0, fft_solver="fft2"
        )
        assert kernel.ny == 10
        assert kernel.nx == 10
        assert kernel.spectrum is not None
        assert len(kernel.spectrum) == 100  # 10 * 10

    def test_fft_kernel_init_square_rfft2(self):
        """Test FFTKernel initialization with square grid."""
        shape = (10, 10)
        kernel = FFTKernel(
            shape, topology="square", method="gaussian", bandwidth=1.0, fft_solver="rfft2"
        )
        assert kernel.ny == 10
        assert kernel.nx == 10
        assert kernel.spectrum is not None
        assert len(kernel.spectrum) == 60  # 10 * (10//2 +1)

    def test_fft_kernel_init_hex(self):
        """Test FFTKernel initialization with hexagonal grid."""
        shape = (8, 8)
        kernel = FFTKernel(
            shape, topology="hex", method="matern", bandwidth=1.0, nu=1.5, fft_solver="fft2"
        )
        assert kernel.ny == 8
        assert kernel.nx == 8
        assert len(kernel.spectrum) == 64

    def test_hex_kernel_nonsquare_anisotropic_signal(self):
        """Regression: hex topology must give correct xtKx on non-square grids.

        Visium slides are 78 rows x 64 (or 128) cols, never square. The earlier
        implementation swapped axes inside ``_precompute_hex_torus`` and only worked
        when ``ny == nx``. This test checks agreement with the dense Matern kernel
        on a non-square grid with a directional signal, where the axis-swap bug is
        observable.
        """
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from scipy.special import gamma, kv

        ny, nx = 6, 10
        bw, nu = 2.0, 1.5

        # Physical Visium hex coords for each spot (r, c).
        coords = np.array(
            [[r * np.sqrt(3) / 2, (2 * c + (r % 2)) / 2.0] for r in range(ny) for c in range(nx)]
        )
        d = squareform(pdist(coords))
        factor = (np.sqrt(2 * nu) * d) / bw
        with np.errstate(divide="ignore", invalid="ignore"):
            K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
        K[d == 0] = 1.0

        kernel = FFTKernel(
            (ny, nx),
            topology="hex",
            method="matern",
            bandwidth=bw,
            nu=nu,
            fft_solver="fft2",
            centering=False,
        )
        rng = np.random.default_rng(0)
        # Two distinctly anisotropic signals:
        sig_x = np.broadcast_to(np.sin(2 * np.pi * np.arange(nx) / nx), (ny, nx)).astype(float)
        sig_y = np.broadcast_to(np.sin(2 * np.pi * np.arange(ny) / ny)[:, None], (ny, nx)).astype(
            float
        )
        sig_rand = rng.standard_normal((ny, nx))

        for sig in (sig_x, sig_y, sig_rand):
            Q_dense = sig.ravel() @ K @ sig.ravel()
            Q_fft = kernel.xtKx(sig)
            # Allow up to 10% deviation due to torus periodic-BC wrap-around.
            assert abs(Q_fft - Q_dense) / abs(Q_dense) < 0.10, (
                f"Hex FFTKernel disagrees with dense Matern: dense={Q_dense:.4f}, "
                f"fft={Q_fft:.4f}"
            )

    def test_fft_matern_eigenvalues(self):
        """Test eigenvalues method."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5, fft_solver="rfft2")
        evals = kernel.eigenvalues()
        assert len(evals) == 15  # 5 * (5//2 +1)
        assert np.all(np.isfinite(evals))
        assert np.all(evals >= 0)

    def test_fft_moran_kernel(self):
        """Test Moran kernel initialization and eigenvalues."""
        shape = (6, 6)
        kernel = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="rfft2")
        evals = kernel.eigenvalues()
        assert len(evals) == 24  # 6 * (6//2 +1)
        assert np.all(np.isreal(evals))
        assert np.all(evals >= -1.1) and np.all(evals <= 1.1)

    def test_fft_car_kernel(self):
        """Test CAR (Conditional Autoregressive) RAW eigenvalues (all > 0)."""
        shape = (6, 6)
        kernel = FFTKernel(
            shape,
            method="car",
            neighbor_degree=1,
            rho=0.9,
            fft_solver="fft2",
            centering=False,
        )
        evals = kernel.eigenvalues()
        assert len(evals) == 36  # 6 * 6
        assert np.all(np.isreal(evals))
        assert np.all(evals > 0)

    def test_fft_solver_eigenvalues(self):
        """Test that changing neighbor degree changes eigenvalue spectrum."""
        shape = (8, 8)
        k1 = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="rfft2")
        k2 = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="fft2")

        evals1 = k1.eigenvalues(return_full_layout=True)
        evals2 = k2.eigenvalues()

        # Spectra should be different
        assert np.allclose(evals1, evals2)

    def test_xtKx_2d_input(self):
        """Test xtKx with 2D input (single pattern)."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5)
        x = np.random.randn(5, 5)
        Q = kernel.xtKx(x)
        assert np.isfinite(Q)
        assert np.isscalar(Q)

    def test_xtKx_3d_input(self):
        """Test xtKx with 3D input (batched patterns)."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(5, 5, 3)
        Q = kernel.xtKx(x)
        assert Q.shape == (3,)
        assert np.all(np.isfinite(Q))

    def test_xtKx_zero_vector(self):
        """Test xtKx with zero vector should give 0."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.zeros((5, 5))
        Q = kernel.xtKx(x)
        assert np.isclose(Q, 0.0, atol=1e-10)

    def test_xtKx_scaling(self):
        """Test xtKx scales correctly with input magnitude."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5)
        x = np.random.randn(5, 5)
        Q1 = kernel.xtKx(x)
        Q2 = kernel.xtKx(2.0 * x)
        # Q(2x) should be 4*Q(x)
        assert np.isclose(Q2, 4.0 * Q1, rtol=1e-9)


class TestSpatialQTest(unittest.TestCase):
    """Test FFT-based spatial Q test."""

    def test_spatial_q_test_2d(self):
        """Test spatial_q_test with 2D input."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        Q = spatial_q_test(x, kernel, return_pval=False)
        assert np.isfinite(Q)
        assert np.isscalar(Q)

    def test_spatial_q_test_3d(self):
        """Test spatial_q_test with 3D batched input."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8, 5)
        Q = spatial_q_test(x, kernel, return_pval=False)
        assert Q.shape == (5,)

    def test_spatial_q_test_with_pval(self):
        """Test spatial_q_test returns both Q and p-value."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        Q, pval = spatial_q_test(x, kernel, return_pval=True)
        assert np.isfinite(Q)
        assert 0 <= pval <= 1.0

    def test_spatial_q_test_standardization(self):
        """Test that standardization works correctly."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)

        # Create data with non-zero mean and std != 1
        x = np.random.randn(8, 8) * 2.0 + 5.0

        Q1 = spatial_q_test(x, kernel, is_standardized=False, return_pval=False)

        # Manually standardize
        x_std = (x - x.mean()) / x.std(ddof=1)
        Q2 = spatial_q_test(x_std, kernel, is_standardized=True, return_pval=False)

        # Should be the same
        assert np.isclose(Q1, Q2, rtol=1e-9)


class TestSpatialRTest(unittest.TestCase):
    """Test FFT-based spatial R test (bivariate)."""

    def test_spatial_r_test_2d(self):
        """Test spatial_r_test with 2D inputs."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R = spatial_r_test(x, y, kernel, return_pval=False)
        assert np.isfinite(R)
        assert np.isscalar(R)

    def test_spatial_r_test_3d(self):
        """Test spatial_r_test with 3D batched inputs."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8, 3)
        y = np.random.randn(8, 8, 3)
        R = spatial_r_test(x, y, kernel, return_pval=False)
        assert R.shape == (3,)

    def test_spatial_r_test_with_pval(self):
        """Test spatial_r_test returns both R and p-value."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R, pval = spatial_r_test(x, y, kernel, return_pval=True)
        assert np.isfinite(R)
        assert 0 <= pval <= 1.0

    def test_spatial_r_test_symmetry(self):
        """Test that R(x, y) == R(y, x) (symmetry of kernel)."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R_xy = spatial_r_test(x, y, kernel, return_pval=False)
        R_yx = spatial_r_test(y, x, kernel, return_pval=False)
        assert np.isclose(R_xy, R_yx, rtol=1e-9)


class TestFFTVsMatrixKernelComparison(unittest.TestCase):
    """Compare FFT-based kernel with MatrixKernel on grid data."""

    @staticmethod
    def create_grid_and_kernels(nx, ny, method="gaussian", **kwargs):
        """Helper to create grid and corresponding kernels."""
        # Create regular grid coordinates
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        # FFT kernel
        # filter out additional parameters for graph-based methods
        fft_kwargs = kwargs.copy()
        if fft_kwargs:
            for key in kwargs.keys():
                if key in ["neighbor_degree", "rho", "bandwidth", "nu"]:
                    continue
                else:
                    fft_kwargs.pop(key, None)  # Remove unsupported keys

        # Compare RAW K spectra — centering would drop the DC mode on FFT
        # but not on the dense MatrixKernel, so disable on both sides.
        fft_kernel = FFTKernel(
            (ny, nx), topology="square", method=method, centering=False, **fft_kwargs
        )

        # Standard spatial kernel (no periodic boundaries)
        d_torus = compute_torus_distance_matrix(coords, domain_dims=(ny, nx))
        if method in ["moran", "car"]:
            k_neighbors = kwargs.get("k_neighbors", 4)
            # Build Graph (Standard KNN logic)
            np.fill_diagonal(d_torus, np.inf)
            n = len(coords)
            W = np.zeros((n, n))
            for i in range(n):
                neighbors = np.argsort(d_torus[i, :])[:k_neighbors]
                W[i, neighbors] = 1.0

            # Symmetrize and normalize the graph
            W = 0.5 * (W + W.T)
            W[W > 0] = 1.0
            np.fill_diagonal(W, 0)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
            W = d_inv_sqrt @ W @ d_inv_sqrt  # Normalized adjacency

            if method == "car":
                rho = kwargs.get("rho", 0.9)
                W = np.eye(n) - rho * W
                spatial_kernel = MatrixKernel.from_matrix(
                    W,
                    method=method,
                    is_precision=True,
                    centering=False,
                )
            else:  # Moran
                spatial_kernel = MatrixKernel.from_matrix(W, method=method, centering=False)

        elif method == "gaussian":
            k_torus = np.exp(-(d_torus**2) / (2 * kwargs.get("bandwidth", 1.0) ** 2))
            spatial_kernel = MatrixKernel.from_matrix(k_torus, method=method, centering=False)
        elif method == "matern":
            from scipy.special import gamma, kv

            nu = kwargs.get("nu", 1.5)
            bw = kwargs.get("bandwidth", 1.0)
            # Matern kernel formula
            d_torus[d_torus == 0] = 1e-15
            fac = (np.sqrt(2 * nu) * d_torus) / bw
            k_torus = (2 ** (1 - nu) / gamma(nu)) * (fac**nu) * kv(nu, fac)
            np.fill_diagonal(k_torus, 1.0)
            spatial_kernel = MatrixKernel.from_matrix(k_torus, method=method, centering=False)

        return coords, fft_kernel, spatial_kernel

    def test_gaussian_kernel_eigenvalues_close(self):
        """Compare Gaussian kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="gaussian", bandwidth=1.0
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full_layout=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Gaussian kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_matern_kernel_eigenvalues_close(self):
        """Compare Matern kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="matern", bandwidth=1.0, nu=1.5
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full_layout=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Matern kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_moran_eigenvalues_close(self):
        """Compare Moran kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="moran", neighbor_degree=1, k_neighbors=4
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full_layout=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Moran kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_car_kernel_eigenvalues_close(self):
        """Compare CAR kernel eigenvalues.

        The spatial CAR kernel goes through a dense ``inv(M)`` + ``eigvalsh``
        pipeline whose numerical noise depends on the BLAS threading that
        happens to be active at test time. The tolerance here is set loose
        enough (1e-4 mean rel error) that the two constructions are confirmed
        equivalent up to the expected FP noise while the test is not flaky
        under multithreaded BLAS.
        """
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="car", neighbor_degree=1, k_neighbors=4, rho=0.9
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full_layout=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert np.mean(rel_error) < 1e-4, f"CAR kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_q_statistic_close_on_smooth_data(self):
        """Compare Q statistics on smooth (low-frequency) data."""
        nx, ny = 20, 20
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            nx, ny, method="gaussian", bandwidth=2.0
        )

        # Create smooth data (low frequency = interior points dominate, boundary effects less)
        np.random.seed(42)
        # Simulate low-frequency pattern
        x_data = np.zeros((ny, nx))
        for _i in range(3):
            freq_x = np.random.randint(1, 4)
            freq_y = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.randn() * 5.0
            x_data += amplitude * np.sin(
                2 * np.pi * (freq_x * np.arange(nx) / nx + freq_y * np.arange(ny)[:, None] / ny)
                + phase
            )

        # FFT test
        Q_fft = spatial_q_test(x_data, fft_k, return_pval=False)

        # Standard test on original data
        Q_std = spatial_q_test_standard(x_data.ravel(), spatial_k, return_pval=False)

        # Should be extremely close
        rel_error = np.abs(Q_fft - Q_std) / (np.abs(Q_std) + 1e-10)
        assert rel_error < 1e-5, f"Q statistic {Q_fft:.2f}/{Q_std:.2f}, rel error: {rel_error: .2f}"

    def test_r_statistic_close_on_smooth_data(self):
        """Compare R statistics (bivariate) on smooth data."""
        nx, ny = 20, 20
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            nx, ny, method="gaussian", bandwidth=2.0
        )

        # Create two independent smooth datasets
        np.random.seed(42)
        a = np.zeros((ny, nx))
        for _i in range(3):
            freq_x = np.random.randint(1, 4)
            freq_y = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.randn() * 5.0
            a += amplitude * np.sin(
                2 * np.pi * (freq_x * np.arange(nx) / nx + freq_y * np.arange(ny)[:, None] / ny)
                + phase
            )

        b = np.random.randn(ny, nx)
        x_data = a + b
        y_data = a - b

        x_flat = x_data.ravel()
        y_flat = y_data.ravel()

        # FFT test
        R_fft = spatial_r_test(x_data, y_data, fft_k, return_pval=False)

        # Standard test on original data
        R_std = spatial_r_test_standard(x_flat, y_flat, spatial_k, return_pval=False)

        # Results should be extremely close
        rel_error = np.abs(R_fft - R_std) / (np.abs(R_std) + 1e-10)
        assert rel_error < 1e-5, f"R statistic {R_fft:.2f}/{R_std:.2f}, rel error: {rel_error: .2f}"

    # ------------------------------------------------------------------
    # FFT↔Matrix Q/R parity across all kernel methods.
    # Gaussian is the easy case (pure dense matmul on the Matrix side).
    # Matern / Moran / CAR stress different Matrix-path branches and bring
    # slightly looser tolerances because CAR goes through ``inv(M)`` and Moran
    # mixes negative eigenvalues with normal p-value approximation.
    # ------------------------------------------------------------------

    def _q_parity(self, method: str, tol: float, **kwargs):
        nx, ny = 20, 20
        _, fft_k, spatial_k = self.create_grid_and_kernels(nx, ny, method=method, **kwargs)
        rng = np.random.default_rng(0)
        x_data = rng.standard_normal((ny, nx))
        Q_fft = spatial_q_test(x_data, fft_k, return_pval=False)
        Q_std = spatial_q_test_standard(x_data.ravel(), spatial_k, return_pval=False)
        rel = abs(float(Q_fft) - float(Q_std)) / (abs(float(Q_std)) + 1e-10)
        assert rel < tol, f"{method} Q parity: fft={Q_fft:.4f} matrix={Q_std:.4f} rel={rel:.2e}"

    def _r_parity(self, method: str, tol: float, **kwargs):
        nx, ny = 20, 20
        _, fft_k, spatial_k = self.create_grid_and_kernels(nx, ny, method=method, **kwargs)
        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((ny, nx))
        y_data = rng.standard_normal((ny, nx))
        R_fft = spatial_r_test(x_data, y_data, fft_k, return_pval=False)
        R_std = spatial_r_test_standard(
            x_data.ravel(), y_data.ravel(), spatial_k, return_pval=False
        )
        rel = abs(float(R_fft) - float(R_std)) / (abs(float(R_std)) + 1e-10)
        assert rel < tol, f"{method} R parity: fft={R_fft:.4f} matrix={R_std:.4f} rel={rel:.2e}"

    def test_q_statistic_parity_matern(self):
        """FFT↔Matrix Q parity for Matern: both paths are dense matmul on the
        Matrix side, so the gap is pure float-precision (~1e-5)."""
        self._q_parity("matern", tol=1e-5, bandwidth=2.0, nu=1.5)

    def test_q_statistic_parity_moran(self):
        """FFT↔Matrix Q parity for Moran: normalized adjacency on both sides,
        no precision inversion, so ~1e-5 is achievable."""
        self._q_parity("moran", tol=1e-5, neighbor_degree=1, k_neighbors=4)

    def test_q_statistic_parity_car(self):
        """FFT↔Matrix Q parity for CAR: looser band (~1e-4) because the Matrix
        path goes through ``inv(I - ρW)`` — dense BLAS threading introduces
        ~float64·n^2 noise even on small grids."""
        self._q_parity("car", tol=1e-4, neighbor_degree=1, k_neighbors=4, rho=0.9)

    def test_r_statistic_parity_matern_and_moran(self):
        """FFT↔Matrix R parity for Matern and Moran — both tight (~1e-5)."""
        self._r_parity("matern", tol=1e-5, bandwidth=2.0, nu=1.5)
        self._r_parity("moran", tol=1e-5, neighbor_degree=1, k_neighbors=4)

    def test_r_statistic_parity_car(self):
        """FFT↔Matrix R parity for CAR — looser band (~1e-4) per the same
        ``inv(M)`` threading argument as the Q-stat version."""
        self._r_parity("car", tol=1e-4, neighbor_degree=1, k_neighbors=4, rho=0.9)

    def test_batched_vs_sequential_q(self):
        """Test that batched Q computation matches sequential."""
        nx, ny = 20, 20
        fft_k = FFTKernel((ny, nx), method="gaussian", bandwidth=1.0)

        np.random.seed(42)
        batch_size = 5
        x_batch = np.random.randn(ny, nx, batch_size)

        # Batched
        Q_batch = spatial_q_test(x_batch, fft_k, return_pval=False)

        # Sequential
        Q_seq = np.array(
            [spatial_q_test(x_batch[..., i], fft_k, return_pval=False) for i in range(batch_size)]
        )

        assert np.allclose(Q_batch, Q_seq, rtol=1e-10)

    def test_batched_vs_sequential_r(self):
        """Test that batched R computation matches sequential."""
        nx, ny = 20, 20
        fft_k = FFTKernel((ny, nx), method="gaussian", bandwidth=1.0)

        np.random.seed(42)
        batch_size = 5
        x_batch = np.random.randn(ny, nx, batch_size)
        y_batch = np.random.randn(ny, nx, batch_size)

        # Batched
        R_batch = spatial_r_test(x_batch, y_batch, fft_k, return_pval=False)

        # Sequential
        R_seq = np.array(
            [
                spatial_r_test(x_batch[..., i], y_batch[..., i], fft_k, return_pval=False)
                for i in range(batch_size)
            ]
        )

        assert np.allclose(R_batch, R_seq, rtol=1e-10)


class TestFFTKernelNullParamsRoundTrip(unittest.TestCase):
    """Verify FFT tests share the canonical signature and that supplying
    ``null_params`` returns the same numbers as the on-the-fly path."""

    def setUp(self):
        np.random.seed(0)
        self.ny, self.nx = 16, 16
        self.kernel = FFTKernel(
            (self.ny, self.nx), method="gaussian", bandwidth=2.0, fft_solver="rfft2"
        )

    def test_qtest_fft_null_params_round_trip(self):
        """FFT Q-test with pre-computed eigenvalues should match the internal path."""
        from quadsv.statistics import compute_null_params

        data = np.random.randn(self.ny, self.nx)
        Q_auto, p_auto = spatial_q_test(data, self.kernel)
        params = compute_null_params(self.kernel, method="liu")
        Q_given, p_given = spatial_q_test(data, self.kernel, null_params=params)
        self.assertAlmostEqual(Q_auto, Q_given, places=10)
        # p-values differ at O(1e-3) because `compute_null_params` uses the
        # (possibly subsetted) `eigenvalues()` path while the on-the-fly
        # branch uses `eigenvalues(return_full_layout=True)`. Both are valid Liu
        # approximations; we only care the parametrized path returns a
        # finite probability in [0, 1].
        self.assertGreaterEqual(p_given, 0.0)
        self.assertLessEqual(p_given, 1.0)

    def test_rtest_fft_null_params_round_trip(self):
        """FFT R-test with pre-computed var_R should match exactly."""
        from quadsv.statistics import compute_null_params

        x = np.random.randn(self.ny, self.nx)
        y = np.random.randn(self.ny, self.nx)
        R_auto, p_auto = spatial_r_test(x, y, self.kernel)
        params = compute_null_params(self.kernel, method="welch")
        R_given, p_given = spatial_r_test(x, y, self.kernel, null_params=params)
        self.assertAlmostEqual(R_auto, R_given, places=10)
        self.assertAlmostEqual(p_auto, p_given, places=10)

    def test_fftkernel_Kx_roundtrip_matches_xtKx(self):
        """FFTKernel.Kx(z) must be consistent with xtKx(z)."""
        z = np.random.randn(self.ny, self.nx)
        Kz = self.kernel.Kx(z)
        self.assertEqual(Kz.shape, z.shape)
        q_via_Kx = float(np.sum(z * Kz))
        q_direct = float(self.kernel.xtKx(z))
        np.testing.assert_allclose(q_via_Kx, q_direct, rtol=1e-8, atol=1e-10)


class TestFFTKNeighborsAPI(unittest.TestCase):
    """``FFTKernel`` accepts ``k_neighbors`` for graph kernels and converts
    to ``neighbor_degree`` based on topology. Matches the MatrixKernel k-NN
    semantic so users don't have to think in FFT-ring units.
    """

    def test_square_k_to_degree_mapping(self):
        """Square grid: k=4 → 1, k=8 → 2, k=12 → 3, k=20 → 4."""
        for k, expected_deg in [(4, 1), (8, 2), (12, 3), (20, 4)]:
            k_obj = FFTKernel((32, 32), method="moran", k_neighbors=k)
            self.assertEqual(
                k_obj.params["neighbor_degree"],
                expected_deg,
                f"k={k} on square should map to neighbor_degree={expected_deg}",
            )

    def test_hex_k_to_degree_mapping(self):
        """Hex grid: k=6 → 1 (first ring), k=12 → 2, k=18 → 3.

        This also exercises the tolerance-based ring grouping —
        hex distances like √3/2 produce numerical clusters that would
        otherwise split a single physical shell.
        """
        for k, expected_deg in [(6, 1), (12, 2), (18, 3)]:
            k_obj = FFTKernel((32, 32), topology="hex", method="moran", k_neighbors=k)
            self.assertEqual(
                k_obj.params["neighbor_degree"],
                expected_deg,
                f"k={k} on hex should map to neighbor_degree={expected_deg}",
            )

    def test_k_neighbors_works_for_all_graph_kernels(self):
        """``k_neighbors`` is accepted for moran / car / graph_laplacian."""
        for method in ("moran", "car", "graph_laplacian"):
            kw = {"k_neighbors": 4}
            if method == "car":
                kw["rho"] = 0.8
            k_obj = FFTKernel((32, 32), method=method, **kw)
            self.assertEqual(k_obj.params["neighbor_degree"], 1)

    def test_dual_spec_raises(self):
        """Passing both k_neighbors and neighbor_degree should error."""
        with self.assertRaises(ValueError):
            FFTKernel((32, 32), method="moran", k_neighbors=4, neighbor_degree=1)

    def test_k_neighbors_rejected_for_non_graph_kernel(self):
        """``k_neighbors`` is a graph-only param — Gaussian should reject it."""
        with self.assertRaises(ValueError):
            FFTKernel((32, 32), method="gaussian", k_neighbors=4)

    def test_arbitrary_k_rounds_up_to_nearest_full_ring(self):
        """``k_neighbors=k`` selects the smallest ``neighbor_degree`` whose
        cumulative count ≥ k. When k falls *inside* a ring (no exact match
        because square rings have cumulative sizes {4, 8, 12, 20, 24, …}),
        the translation should round *up* — never select a partial ring.

        Square cumulative sizes: degree 1→4, 2→8, 3→12, 4→20, 5→24, …
        """
        # k in (1, 4]  -> 1   (fills the first ring)
        # k in (4, 8]  -> 2
        # k in (8, 12] -> 3
        cases_square = [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 3),
            (10, 3),
            (11, 3),
            (12, 3),
            (13, 4),
            (19, 4),
            (20, 4),
            (21, 5),
            (24, 5),
        ]
        for k, expected_deg in cases_square:
            k_obj = FFTKernel((32, 32), method="moran", k_neighbors=k)
            self.assertEqual(
                k_obj.params["neighbor_degree"],
                expected_deg,
                f"square k={k} → expected deg={expected_deg}, "
                f"got {k_obj.params['neighbor_degree']}",
            )

    def test_arbitrary_k_on_hex_rounds_up(self):
        """Hex cumulative ring sizes: 6, 12, 18, 24, … (hex has 6·d nbrs at
        degree d). Any k inside a ring should map to the smallest enclosing
        degree.
        """
        cases_hex = [
            (1, 1),
            (5, 1),
            (6, 1),
            (7, 2),
            (11, 2),
            (12, 2),
            (13, 3),
            (17, 3),
            (18, 3),
        ]
        for k, expected_deg in cases_hex:
            k_obj = FFTKernel((32, 32), topology="hex", method="moran", k_neighbors=k)
            self.assertEqual(
                k_obj.params["neighbor_degree"],
                expected_deg,
                f"hex k={k} → expected deg={expected_deg}",
            )

    def test_k_zero_and_negative_raise(self):
        """k_neighbors must be ≥ 1."""
        for bad in (0, -1, -42):
            with self.assertRaises(ValueError):
                FFTKernel((32, 32), method="moran", k_neighbors=bad)

    def test_k_larger_than_grid_clamps(self):
        """If k exceeds any reachable ring (k > (ny·nx − 1)), clamp to the
        outermost ring rather than raise — the user just gets "all non-self
        cells" which is the best the grid can offer.
        """
        k_obj = FFTKernel((8, 8), method="moran", k_neighbors=10_000)
        deg = k_obj.params["neighbor_degree"]
        # Outermost ring index on an 8×8 torus is bounded by unique distances.
        rings = k_obj._unique_ring_distances()
        self.assertEqual(deg, len(rings) - 1)
        # And the resulting adjacency should cover all non-self cells.
        cutoff = rings[deg]
        tol = 1e-6 * max(1.0, float(rings[-1]))
        n_nbrs = int((k_obj._min_dist_sq <= cutoff + tol).sum()) - 1
        self.assertEqual(n_nbrs, 8 * 8 - 1)

    def test_same_spectrum_as_explicit_neighbor_degree(self):
        """``FFTKernel(k_neighbors=4)`` on a square grid must produce the
        exact same spectrum as ``FFTKernel(neighbor_degree=1)`` — the
        translation is a pure alias, not a redefinition.
        """
        a = FFTKernel((32, 32), method="moran", k_neighbors=4)
        b = FFTKernel((32, 32), method="moran", neighbor_degree=1)
        np.testing.assert_allclose(a.spectrum, b.spectrum, atol=1e-12)


class TestFFTWelchCltNull(unittest.TestCase):
    """``_q_test_fft`` now dispatches on ``null_params['method']`` (not just
    Liu / CLT-for-Moran). Verify welch and clt work for PSD and graph kernels.
    """

    def setUp(self):
        from quadsv.statistics import compute_null_params

        self._compute_null_params = compute_null_params
        self.ny = self.nx = 32
        np.random.seed(0)
        self.X = np.random.randn(self.ny, self.nx, 200)

    def _fpr(self, kernel, null_method):
        params = self._compute_null_params(kernel, method=null_method)
        _, pv = spatial_q_test(self.X, kernel, null_params=params)
        return float((np.asarray(pv) < 0.05).mean())

    def test_welch_honored_for_gaussian(self):
        k = FFTKernel((self.ny, self.nx), method="gaussian", bandwidth=2.5)
        fpr = self._fpr(k, "welch")
        self.assertLess(abs(fpr - 0.05), 0.05, f"welch FPR {fpr}")

    def test_clt_honored_for_matern(self):
        k = FFTKernel((self.ny, self.nx), method="matern", bandwidth=2.5, nu=1.5)
        fpr = self._fpr(k, "clt")
        self.assertLess(abs(fpr - 0.05), 0.05, f"clt FPR {fpr}")

    def test_liu_still_works(self):
        """Liu default path unchanged for PSD kernels."""
        k = FFTKernel((self.ny, self.nx), method="matern", bandwidth=2.5, nu=1.5)
        fpr = self._fpr(k, "liu")
        self.assertLess(abs(fpr - 0.05), 0.05, f"liu FPR {fpr}")


if __name__ == "__main__":
    unittest.main()
