"""Tests for comparator spectral feature helpers."""

from __future__ import annotations

import numpy as np
import pytest

from quadsv.comparators.features import (
    align_spectra_by_rotation,
    apply_rotations_to_spectra,
    compute_sample_spectrum,
    effective_rank,
    estimate_rotations_from_landmarks,
    gene_pattern_diversity,
    power_spectrum_anisotropy,
    radial_bin_spectrum,
    stream_geomean_landmark,
    stream_polar_features,
    stream_radial_features,
)
from quadsv.kernels.fft import power_spectrum_2d


def _axis_angle_error(observed: float, expected: float) -> float:
    """Smallest axial-angle error in degrees."""
    return abs((observed - expected + 90.0) % 180.0 - 90.0)


class TestPowerSpectrumAnisotropy:
    """Second-angular-moment diagnostics for raw 2D spectra."""

    def test_axis_aligned_full_fft_spectra(self):
        power = np.zeros((3, 8, 8), dtype=float)
        power[0, 0, 1] = power[0, 0, -1] = 2.0
        power[1, 1, 0] = power[1, -1, 0] = 3.0
        power[2, 0, 1] = power[2, 0, -1] = 2.0
        power[2, 1, 0] = power[2, -1, 0] = 2.0

        df = power_spectrum_anisotropy(power, grid_shape=(8, 8), fft_solver="fft2")

        np.testing.assert_allclose(df["anisotropy"].iloc[:2], [1.0, 1.0], atol=1e-12)
        assert _axis_angle_error(df["dominant_angle_deg"].iloc[0], 0.0) < 1e-12
        assert _axis_angle_error(df["dominant_angle_deg"].iloc[1], 90.0) < 1e-12
        assert df["anisotropy"].iloc[2] < 1e-12
        np.testing.assert_allclose(df["total_power"], [4.0, 6.0, 8.0], atol=1e-12)

    def test_rfft2_matches_full_fft2_layout(self):
        ny, nx = 32, 40
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        images = np.stack(
            [
                np.sin(2 * np.pi * (5.0 * yy / ny + 2.0 * xx / nx)),
                np.sin(2 * np.pi * (2.0 * yy / ny + 6.0 * xx / nx)),
            ],
            axis=0,
        )
        full = compute_sample_spectrum(images, fft_solver="fft2")
        half = compute_sample_spectrum(images, fft_solver="rfft2")

        df_full = power_spectrum_anisotropy(full, (ny, nx), fft_solver="fft2")
        df_half = power_spectrum_anisotropy(half, (ny, nx), fft_solver="rfft2")

        np.testing.assert_allclose(df_half["anisotropy"], df_full["anisotropy"], rtol=1e-12)
        np.testing.assert_allclose(
            df_half["dominant_angle_deg"],
            df_full["dominant_angle_deg"],
            atol=1e-12,
        )
        np.testing.assert_allclose(df_half["total_power"], df_full["total_power"], rtol=1e-12)

    @pytest.mark.parametrize("grid_shape", [(7, 9), (8, 9), (7, 10), (8, 10)])
    def test_rfft2_matches_full_fft2_on_even_and_odd_grids(self, grid_shape):
        ny, nx = grid_shape
        rng = np.random.default_rng(3)
        images = rng.standard_normal((4, ny, nx))
        full = compute_sample_spectrum(images, fft_solver="fft2")
        half = compute_sample_spectrum(images, fft_solver="rfft2")

        df_full = power_spectrum_anisotropy(full, grid_shape, fft_solver="fft2")
        df_half = power_spectrum_anisotropy(half, grid_shape, fft_solver="rfft2")

        np.testing.assert_allclose(df_half["anisotropy"], df_full["anisotropy"], rtol=1e-12)
        np.testing.assert_allclose(df_half["total_power"], df_full["total_power"], rtol=1e-12)
        angle_err = [
            _axis_angle_error(observed, expected)
            for observed, expected in zip(
                df_half["dominant_angle_deg"],
                df_full["dominant_angle_deg"],
                strict=True,
            )
        ]
        assert max(angle_err) < 1e-10

    def test_spacing_changes_physical_frequency_angle(self):
        power = np.zeros((1, 16, 16), dtype=float)
        power[0, 1, 1] = power[0, -1, -1] = 1.0

        df_square = power_spectrum_anisotropy(power, (16, 16), spacing=(1.0, 1.0))
        df_rect = power_spectrum_anisotropy(power, (16, 16), spacing=(2.0, 1.0))

        assert _axis_angle_error(df_square["dominant_angle_deg"].iloc[0], 45.0) < 1e-12
        assert (
            _axis_angle_error(
                df_rect["dominant_angle_deg"].iloc[0],
                np.degrees(np.arctan(0.5)),
            )
            < 1e-12
        )

    def test_zero_power_reports_nan_orientation(self):
        df = power_spectrum_anisotropy(np.zeros((2, 8, 5)), (8, 8), fft_solver="rfft2")
        assert df["anisotropy"].isna().all()
        assert df["dominant_angle_deg"].isna().all()
        np.testing.assert_array_equal(df["total_power"].to_numpy(), np.zeros(2))

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="incompatible with grid width"):
            power_spectrum_anisotropy(np.zeros((2, 8, 7)), (8, 10))


class TestPatternDiversity:
    """Effective-rank and pattern-diversity feature diagnostics."""

    def test_effective_rank_identity_equals_K(self):
        for K in [3, 10, 30]:
            assert effective_rank(np.eye(K)) == pytest.approx(K, abs=1e-12)

    def test_effective_rank_rank_one_equals_one(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(15)
        cov = np.outer(v, v)

        assert effective_rank(cov) == pytest.approx(1.0, abs=1e-10)

    def test_effective_rank_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            K = rng.integers(2, 30)
            X = rng.standard_normal((50, K))
            cov = X.T @ X / 50

            ke = effective_rank(cov)

            assert ke >= 1.0 - 1e-10
            assert ke <= K + 1e-10

    def test_effective_rank_with_weights_changes_value(self):
        K = 20
        cov = np.eye(K)
        ke_uniform = effective_rank(cov, weights=np.ones(K) / K)
        w_concentrated = np.zeros(K)
        w_concentrated[0] = 1.0
        ke_concentrated = effective_rank(cov, weights=w_concentrated)

        assert ke_uniform == pytest.approx(K, abs=1e-10)
        assert ke_concentrated == pytest.approx(1.0, abs=1e-10)

    def test_effective_rank_invalid_inputs(self):
        with pytest.raises(ValueError, match="square 2D matrix"):
            effective_rank(np.zeros((5, 4)))
        with pytest.raises(ValueError, match="non-negative"):
            effective_rank(np.eye(5), weights=np.array([1, 1, 1, 1, -1]))

    def test_gene_pattern_diversity_low_vs_high_heterogeneity(self):
        """Rank-1 spectra should have lower diversity than iid spectra."""
        rng = np.random.default_rng(0)
        K = 10
        shape = np.linspace(1, 5, K)
        gene_scales = rng.uniform(0.5, 2.0, size=200)
        spectra_rank1 = np.exp(np.log(shape)[None, :] + np.log(gene_scales)[:, None])
        spectra_iid = np.exp(rng.standard_normal((200, K)))

        assert gene_pattern_diversity(spectra_rank1) < 1.5
        assert gene_pattern_diversity(spectra_iid) > K * 0.5


class TestRadialBinning:
    """Radial power binning across FFT solver layouts."""

    def test_isotropic_gaussian_bump_decreases_radially(self):
        ny, nx = 32, 32
        y, x = np.meshgrid(np.arange(ny) - ny / 2, np.arange(nx) - nx / 2, indexing="ij")
        r2 = y**2 + x**2
        bump = np.exp(-r2 / (2 * 4.0**2))
        P = power_spectrum_2d(bump, fft_solver="fft2")
        rb = radial_bin_spectrum(P, grid_shape=(ny, nx), n_bins=20, fft_solver="fft2")
        assert rb[0] > rb[-1]
        # Allow small mid-range reshuffles from discretization.
        assert (np.diff(rb) <= 1e-6).sum() >= len(rb) - 4

    def test_radial_consistent_across_solvers(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((16, 24))
        P_full = power_spectrum_2d(img, fft_solver="fft2")
        P_half = power_spectrum_2d(img, fft_solver="rfft2")
        rb_full = radial_bin_spectrum(P_full, grid_shape=(16, 24), n_bins=8, fft_solver="fft2")
        rb_half = radial_bin_spectrum(P_half, grid_shape=(16, 24), n_bins=8, fft_solver="rfft2")
        np.testing.assert_allclose(rb_full, rb_half, rtol=1e-9, atol=1e-9)

    def test_exclude_dc_keeps_first_radial_interval(self):
        spectrum = np.zeros((4, 4))
        spectrum[0, 0] = 100.0  # DC should be removed from the first interval.
        spectrum[0, 1] = 4.0  # The first nonzero kx frequency should remain.

        rb = radial_bin_spectrum(
            spectrum,
            grid_shape=(4, 4),
            fft_solver="fft2",
            edges=np.array([0.0, 0.3, 1.0]),
        )

        assert rb.shape == (2,)
        assert rb[0] > 0.0

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="last two dims"):
            radial_bin_spectrum(np.zeros((10, 10)), grid_shape=(8, 8), fft_solver="fft2")


class TestRotationAlignment:
    """Rotation alignment APIs on small synthetic spectra."""

    @staticmethod
    def _stripes(ny: int, nx: int, period: float = 8.0) -> np.ndarray:
        y = np.arange(ny)[:, None]
        return np.broadcast_to(np.sin(2 * np.pi * y / period).astype(float), (ny, nx)).copy()

    @staticmethod
    def _stripes_rotated(ny: int, nx: int, angle: float, period: float = 8.0) -> np.ndarray:
        import scipy.ndimage

        base = TestRotationAlignment._stripes(ny, nx, period=period)
        return scipy.ndimage.rotate(base, angle=angle, reshape=False, order=1, mode="reflect")

    def test_single_landmark_recovers_known_rotation(self):
        ny = nx = 48
        true_angle = 25.0
        ref = self._stripes(ny, nx)
        rot = self._stripes_rotated(ny, nx, true_angle)

        sp_ref = compute_sample_spectrum(ref[None, :, :], fft_solver="fft2")
        sp_rot = compute_sample_spectrum(rot[None, :, :], fft_solver="fft2")
        _, angles = align_spectra_by_rotation(
            [sp_ref, sp_rot],
            grid_shapes=[(ny, nx), (ny, nx)],
            target_spectra=[sp_ref, sp_rot],
            fft_solver="fft2",
            reference_index=0,
            n_theta=360,
        )
        recovered = angles[1] % 180.0
        true_mod = true_angle % 180.0
        diff = min(abs(recovered - true_mod), 180.0 - abs(recovered - true_mod))
        assert diff < 5.0, f"recovered={recovered}, true={true_mod}, diff={diff}"

    def test_per_landmark_alignment_handles_perpendicular_landmarks(self):
        """Per-landmark alignment survives landmarks on perpendicular axes."""
        ny = nx = 96
        true_angle = 30.0

        lm_h = TestRotationSimulation._sine_at_angle(ny, nx, 12.0, 0.0, 0.0)
        lm_v = TestRotationSimulation._sine_at_angle(ny, nx, 0.0, 12.0, 0.0)
        ref_stack = np.stack([lm_h, lm_v], axis=0)
        rot_stack = np.stack(
            [
                TestRotationSimulation._sine_at_angle(ny, nx, 12.0, 0.0, true_angle),
                TestRotationSimulation._sine_at_angle(ny, nx, 0.0, 12.0, true_angle),
            ],
            axis=0,
        )
        sp_ref = compute_sample_spectrum(ref_stack, fft_solver="fft2")
        sp_rot = compute_sample_spectrum(rot_stack, fft_solver="fft2")

        angles = estimate_rotations_from_landmarks(
            [sp_ref, sp_rot],
            grid_shapes=[(ny, nx), (ny, nx)],
            fft_solver="fft2",
            n_theta=720,
        )
        diff = TestRotationSimulation._canon_err(angles[1], true_angle)
        assert diff < 3.0, f"per-landmark recovered={angles[1]}, expected {true_angle}"

    def test_shape_validation(self):
        sp = compute_sample_spectrum(
            np.random.default_rng(0).standard_normal((3, 16, 16)), fft_solver="fft2"
        )
        sp_bad = compute_sample_spectrum(
            np.random.default_rng(0).standard_normal((2, 16, 16)), fft_solver="fft2"
        )
        with pytest.raises(ValueError, match="must match across samples"):
            align_spectra_by_rotation(
                [sp, sp_bad], grid_shapes=[(16, 16), (16, 16)], fft_solver="fft2"
            )
        with pytest.raises(ValueError, match="reference_index"):
            align_spectra_by_rotation(
                [sp, sp],
                grid_shapes=[(16, 16), (16, 16)],
                reference_index=9,
                fft_solver="fft2",
            )

    def test_apply_rotations_to_different_target(self):
        import scipy.ndimage

        ny = nx = 48
        true_angle = 18.0
        ref_landmark = self._stripes(ny, nx)
        cur_landmark = self._stripes_rotated(ny, nx, true_angle)

        rng = np.random.default_rng(0)
        ref_target = rng.standard_normal((3, ny, nx))
        cur_target = np.stack(
            [
                scipy.ndimage.rotate(
                    ref_target[j], true_angle, reshape=False, order=1, mode="reflect"
                )
                for j in range(3)
            ],
            axis=0,
        )
        sp_ref_lm = compute_sample_spectrum(ref_landmark[None, :, :], fft_solver="fft2")
        sp_cur_lm = compute_sample_spectrum(cur_landmark[None, :, :], fft_solver="fft2")
        sp_ref_tgt = compute_sample_spectrum(ref_target, fft_solver="fft2")
        sp_cur_tgt = compute_sample_spectrum(cur_target, fft_solver="fft2")

        angles = estimate_rotations_from_landmarks(
            [sp_ref_lm, sp_cur_lm],
            grid_shapes=[(ny, nx), (ny, nx)],
            fft_solver="fft2",
            n_theta=360,
        )
        rotated = apply_rotations_to_spectra(
            [sp_ref_tgt, sp_cur_tgt],
            grid_shapes=[(ny, nx), (ny, nx)],
            angles_deg=angles,
            fft_solver="fft2",
        )
        # After applying the recovered rotation to the target spectra, the
        # L2 distance between sample 1 and the reference should drop.
        # Tolerance is loose because scipy.ndimage.rotate is itself lossy
        # on discrete grids (interpolation ringing around high-amplitude
        # FFT peaks). Even a perfectly recovered angle leaves substantial
        # residual mismatch; we only require a clear improvement.
        unrot_dist = np.linalg.norm(sp_cur_tgt - sp_ref_tgt)
        rot_dist = np.linalg.norm(rotated[1] - sp_ref_tgt)
        assert rot_dist < 0.95 * unrot_dist, (
            f"rotation failed to reduce distance: unrot={unrot_dist:.2g} " f"rot={rot_dist:.2g}"
        )


class TestRotationSimulation:
    """Rotation simulations that avoid pixel-interpolation bias."""

    @staticmethod
    def _sine_at_angle(ny, nx, ky0, kx0, phi_deg):
        """Generate a sinusoid after rotating its wave vector analytically."""
        phi = np.deg2rad(phi_deg)
        c, s = np.cos(phi), np.sin(phi)
        ky = ky0 * c - kx0 * s
        kx = ky0 * s + kx0 * c
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return np.sin(2 * np.pi * (ky * yy / ny + kx * xx / nx))

    @staticmethod
    def _canon_err(a, b):
        d = np.abs(np.mod(a, 180.0) - np.mod(b, 180.0))
        return np.minimum(d, 180.0 - d)

    def test_multi_sample_recovery_with_multiple_landmarks(self):
        ny = nx = 96
        rng = np.random.default_rng(42)
        n_samples = 8
        k0s = [10.0, 16.0, 22.0, 30.0]

        ref_landmarks = np.stack([self._sine_at_angle(ny, nx, k0, 0.0, 0.0) for k0 in k0s], axis=0)
        true_angles = np.concatenate([[0.0], rng.uniform(-80.0, 80.0, size=n_samples - 1)])
        samples = [ref_landmarks]
        for ang in true_angles[1:]:
            samples.append(
                np.stack(
                    [self._sine_at_angle(ny, nx, k0, 0.0, ang) for k0 in k0s],
                    axis=0,
                )
            )
        spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
        recovered = estimate_rotations_from_landmarks(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            fft_solver="fft2",
            n_theta=720,
        )
        errs = self._canon_err(recovered, true_angles)
        assert errs[0] == 0, "reference angle must be exactly 0"
        # FFT-bin aliasing sets the practical accuracy limit for these landmarks.
        assert errs[1:].max() < 4.0, (
            f"per-sample errors (deg): {errs[1:].round(2).tolist()}; "
            f"true angles: {true_angles[1:].round(2).tolist()}; "
            f"recovered: {recovered[1:].round(2).tolist()}"
        )
        assert errs[1:].mean() < 2.0

    def test_more_landmarks_reduces_bias(self):
        """More landmark frequencies should reduce mean recovery error."""
        ny = nx = 96
        rng = np.random.default_rng(0)
        true_angles = np.concatenate([[0.0], rng.uniform(-70.0, 70.0, size=6)])

        def errs_for_k0s(k0s):
            samples = [
                np.stack([self._sine_at_angle(ny, nx, k, 0.0, a) for k in k0s], axis=0)
                for a in true_angles
            ]
            spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
            rec = estimate_rotations_from_landmarks(
                spectra,
                grid_shapes=[(ny, nx)] * len(samples),
                fft_solver="fft2",
                n_theta=720,
            )
            return self._canon_err(rec, true_angles)[1:]

        one = errs_for_k0s([12.0]).mean()
        many = errs_for_k0s([10.0, 14.0, 20.0, 26.0, 34.0]).mean()
        assert many <= one + 0.1, (
            f"multi-landmark bias ({many:.2f} deg) did not improve over "
            f"single-landmark ({one:.2f} deg)"
        )

    def test_apply_rotations_matches_raw_reference(self):
        ny = nx = 96
        rng = np.random.default_rng(7)
        n_samples = 5
        k0s = [10.0, 16.0, 24.0]

        ref_landmarks = np.stack([self._sine_at_angle(ny, nx, k0, 0.0, 0.0) for k0 in k0s], axis=0)
        true_angles = np.concatenate([[0.0], rng.uniform(-60.0, 60.0, size=n_samples - 1)])
        samples = [ref_landmarks]
        for ang in true_angles[1:]:
            samples.append(
                np.stack(
                    [self._sine_at_angle(ny, nx, k0, 0.0, ang) for k0 in k0s],
                    axis=0,
                )
            )
        spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
        angles = estimate_rotations_from_landmarks(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            fft_solver="fft2",
            n_theta=720,
        )
        corrected = apply_rotations_to_spectra(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            angles_deg=angles,
            fft_solver="fft2",
        )
        for i in range(1, n_samples):
            d_before = np.linalg.norm(spectra[i] - spectra[0])
            d_after = np.linalg.norm(corrected[i] - spectra[0])
            # Tolerance is loose because (a) scipy.ndimage.rotate on a
            # discrete spectrum is lossy: every applied angle ringing-blurs
            # high-amplitude FFT peaks, and (b) delta-like spectra of
            # sinusoids are the worst-case for bilinear rotation, so a
            # "perfect" recovery still leaves ~20-40% residual L2. The
            # recovery accuracy of ``angles`` itself is tested elsewhere;
            # here we only require a visible improvement.
            assert d_after < 0.95 * d_before, (
                f"sample {i}: rotation-correction did not tighten distance "
                f"(before={d_before:.2g}, after={d_after:.2g})"
            )


class TestPhysicalFrequencyBinning:
    """Physical-frequency binning with spacing and shared bin edges."""

    def test_physical_spacing_changes_bin_scale(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((40, 40))

        P = power_spectrum_2d(img, fft_solver="rfft2")
        rb1 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(1.0, 1.0)
        )
        rb2 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(10.0, 10.0)
        )
        np.testing.assert_allclose(rb1, rb2, rtol=1e-10)

    def test_common_edges_across_heterogeneous_grids(self):
        rng = np.random.default_rng(0)

        img_a = rng.standard_normal((40, 40))
        img_b = rng.standard_normal((50, 60))
        Pa = power_spectrum_2d(img_a, fft_solver="rfft2")
        Pb = power_spectrum_2d(img_b, fft_solver="rfft2")
        edges = np.linspace(0, 0.3, 11)
        rb_a = radial_bin_spectrum(
            Pa, grid_shape=(40, 40), fft_solver="rfft2", spacing=(1.0, 1.0), edges=edges
        )
        rb_b = radial_bin_spectrum(
            Pb, grid_shape=(50, 60), fft_solver="rfft2", spacing=(1.0, 1.0), edges=edges
        )
        assert rb_a.shape == rb_b.shape == (10,)

    def test_explicit_edges_non_monotonic_raises(self):
        P = np.zeros((8, 5))
        with pytest.raises(ValueError, match="monotonically"):
            radial_bin_spectrum(
                P, grid_shape=(8, 8), fft_solver="rfft2", edges=np.array([0.0, 0.5, 0.2])
            )


class TestStreamingFeatureHelpers:
    """Streaming helpers should match full-stack feature construction."""

    def test_stream_radial_features_matches_full_stack_binning(self):
        rng = np.random.default_rng(0)
        spec = rng.uniform(0.1, 4.0, size=(5, 10, 6))
        edges = np.linspace(0.0, 0.5, 7)
        calls = []

        def chunk(start, stop):
            calls.append((start, stop))
            return spec[start:stop]

        streamed = stream_radial_features(
            chunk,
            n_genes=spec.shape[0],
            grid_shape=(10, 10),
            chunk_size=2,
            fft_solver="rfft2",
            spacing=(1.0, 1.0),
            edges=edges,
        )
        full = radial_bin_spectrum(
            spec,
            grid_shape=(10, 10),
            n_bins=6,
            fft_solver="rfft2",
            spacing=(1.0, 1.0),
            edges=edges,
        )
        assert calls == [(0, 2), (2, 4), (4, 5)]
        np.testing.assert_allclose(streamed, full, rtol=1e-12, atol=1e-12)

    def test_stream_geomean_landmark_matches_full_stack_and_subset(self):
        rng = np.random.default_rng(1)
        spec = rng.uniform(0.1, 3.0, size=(6, 8, 8))

        def chunk(start, stop):
            return spec[start:stop]

        all_genes = stream_geomean_landmark(
            chunk,
            n_genes=spec.shape[0],
            grid_shape=(8, 8),
            chunk_size=2,
            eps=1e-12,
        )
        expected_all = np.exp(np.log(spec + 1e-12).mean(axis=0))[None, ...]
        np.testing.assert_allclose(all_genes, expected_all, rtol=1e-12, atol=1e-12)

        subset = np.array([1, 4, 5])
        subset_landmark = stream_geomean_landmark(
            chunk,
            n_genes=spec.shape[0],
            grid_shape=(8, 8),
            chunk_size=2,
            gene_subset=subset,
            eps=1e-12,
        )
        expected_subset = np.exp(np.log(spec[subset] + 1e-12).mean(axis=0))[None, ...]
        np.testing.assert_allclose(subset_landmark, expected_subset, rtol=1e-12, atol=1e-12)

        with pytest.raises(ValueError, match=">= 1 gene"):
            stream_geomean_landmark(
                chunk,
                n_genes=spec.shape[0],
                grid_shape=(8, 8),
                chunk_size=2,
                gene_subset=np.array([], dtype=int),
            )

    def test_stream_polar_features_is_chunk_size_invariant_for_rfft2(self):
        rng = np.random.default_rng(2)
        images = rng.standard_normal((4, 16, 16))
        spec = compute_sample_spectrum(images, fft_solver="rfft2")
        edges = np.linspace(0.0, 0.5, 5)

        def chunk(start, stop):
            return spec[start:stop]

        one_chunk = stream_polar_features(
            chunk,
            n_genes=spec.shape[0],
            grid_shape=(16, 16),
            angle_deg=17.0,
            chunk_size=spec.shape[0],
            freq_edges=edges,
            spacing=(1.0, 1.0),
            n_theta=12,
            fft_solver="rfft2",
        )
        many_chunks = stream_polar_features(
            chunk,
            n_genes=spec.shape[0],
            grid_shape=(16, 16),
            angle_deg=17.0,
            chunk_size=1,
            freq_edges=edges,
            spacing=(1.0, 1.0),
            n_theta=12,
            fft_solver="rfft2",
        )
        assert one_chunk.shape == (4, 4 * 12)
        np.testing.assert_allclose(many_chunks, one_chunk, rtol=1e-12, atol=1e-12)
