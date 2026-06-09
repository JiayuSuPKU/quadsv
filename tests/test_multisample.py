"""Tests for quadsv.multisample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from quadsv.comparators.multisample import (
    align_spectra_by_rotation,
    apply_rotations_to_spectra,
    compare_glm,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
    compute_sample_spectrum,
    estimate_rotations_from_landmarks,
    normalize_background,
    normalize_covariates,
    normalize_shape,
    radial_bin_spectrum,
    stream_geomean_landmark,
    stream_polar_features,
    stream_radial_features,
)
from quadsv.kernels.fft import power_spectrum_2d


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
        # FFT peaks) — even a perfectly recovered angle leaves substantial
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
            f"multi-landmark bias ({many:.2f}°) did not improve over "
            f"single-landmark ({one:.2f}°)"
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
            # discrete spectrum is lossy — every applied angle ringing-blurs
            # high-amplitude FFT peaks — and (b) delta-like spectra of
            # sinusoids are the worst-case for bilinear rotation, so a
            # "perfect" recovery still leaves ~20-40% residual L2. The
            # recovery accuracy of ``angles`` itself is tested elsewhere;
            # here we only require a visible improvement.
            assert d_after < 0.95 * d_before, (
                f"sample {i}: rotation-correction did not tighten distance "
                f"(before={d_before:.2g}, after={d_after:.2g})"
            )


class TestNormalizationPrimitives:
    """Primitive ``normalize_*`` helper behavior."""

    def test_background_identical_genes_become_unit(self):
        spec = np.tile(np.arange(1.0, 6.0), (10, 1))
        out = normalize_background(spec)
        np.testing.assert_allclose(out, np.ones_like(out), atol=1e-9)

    def test_covariates_perfect_predictor_residual_is_unity(self):
        rng = np.random.default_rng(0)
        cov = rng.uniform(0.1, 5.0, size=(2, 8))
        gene = np.exp(2.0 + 1.5 * np.log(cov[0]) - 0.7 * np.log(cov[1]))
        gene = np.tile(gene, (5, 1))
        out = normalize_covariates(gene, cov, fit_intercept=True)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_covariates_validate_last_axis(self):
        with pytest.raises(ValueError, match="Last axis"):
            normalize_covariates(np.zeros((3, 5)), np.zeros((2, 4)))

    def test_shape_normalization_sums_to_one_along_axis(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.1, 10.0, size=(4, 7, 12))
        out = normalize_shape(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-12)

    def test_shape_normalization_cancels_scalar_rescale(self):
        rng = np.random.default_rng(1)
        row = rng.uniform(0.5, 3.0, size=10)
        scales = np.array([[0.3], [1.0], [50.0]])
        stack = scales * row[None, :]
        out = normalize_shape(stack, axis=-1)
        np.testing.assert_allclose(out[0], out[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[1], out[2], rtol=1e-10, atol=1e-12)

    def test_normalizers_share_eps_default(self):
        import inspect

        for fn in (normalize_background, normalize_covariates, normalize_shape):
            sig = inspect.signature(fn)
            assert "eps" in sig.parameters, f"{fn.__name__} missing eps kwarg"
            assert (
                sig.parameters["eps"].default == 1e-12
            ), f"{fn.__name__} eps default differs from 1e-12"

    def test_covariates_output_strictly_positive(self):
        rng = np.random.default_rng(0)
        gene = rng.lognormal(size=(50, 10))
        cov = rng.lognormal(size=(3, 10))
        out = normalize_covariates(gene, cov)
        assert (out > 0).all(), "log-space normalize_covariates must return > 0"

    def test_covariates_commutes_with_background_in_log_space(self):
        """Background and covariate projections commute in log space."""
        rng = np.random.default_rng(0)
        spec = rng.lognormal(size=(40, 12))
        cov = rng.lognormal(size=(2, 12))
        order_a = normalize_covariates(normalize_background(spec, axis=-2), cov)
        order_b = normalize_background(normalize_covariates(spec, cov), axis=-2)
        np.testing.assert_allclose(order_a, order_b, rtol=1e-10, atol=1e-12)

    def test_covariates_without_covariates_centers_by_geomean(self):
        rng = np.random.default_rng(0)
        spec = rng.lognormal(size=(15, 9))
        out = normalize_covariates(spec, np.zeros((0, spec.shape[-1])))
        geo_mean_per_gene = np.exp(np.mean(np.log(spec + 1e-12), axis=-1, keepdims=True))
        np.testing.assert_allclose(out, spec / geo_mean_per_gene, rtol=1e-10)

    def test_covariates_accepts_spectra_keyword(self):
        rng = np.random.default_rng(0)
        gene = rng.uniform(size=(20, 8))
        cov = rng.uniform(size=(2, 8))
        out_kw = normalize_covariates(spectra=gene, covariate_spectra=cov)
        out_pos = normalize_covariates(gene, cov)
        np.testing.assert_array_equal(out_kw, out_pos)


class TestTwoGroupNullCalibration:
    """Permutation-null calibration checks."""

    def test_log_l2_pvalues_are_uniform_under_h0(self):
        rng = np.random.default_rng(42)
        n_samples, n_genes, K = 8, 200, 12
        spectra = rng.uniform(0.5, 5.0, size=(n_samples, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Force the sampling path (n_perm_max=0) so the KS test can check
        # continuous uniformity. Exact enumeration with only C(8,4)=70
        # distinct relabellings produces a discrete distribution on 71
        # values, which KS rejects by construction even under perfect
        # calibration; its own calibration is covered by
        # ``test_exact_permutation_on_small_samples`` below.
        df = compare_two_groups(
            spectra,
            groups,
            statistic="log_l2",
            n_perm=300,
            random_state=0,
            n_perm_max=0,
        )
        ks_stat, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"p-values not uniform under H0: KS p={ks_p:.4f}"

    def test_exact_permutation_on_small_samples(self):
        """Exact enumeration is deterministic and calibrated despite discrete p-values."""
        rng = np.random.default_rng(0)
        n_genes, K = 50, 10
        spectra = rng.uniform(0.5, 5.0, size=(8, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df_a = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=5000, random_state=0)
        df_b = compare_two_groups(
            spectra, groups, statistic="log_l2", n_perm=5000, random_state=999
        )
        np.testing.assert_allclose(
            df_a.set_index("Feature").loc[df_b["Feature"], "P_value"].to_numpy(),
            df_b["P_value"].to_numpy(),
        )
        unique_pvals = set(df_a["P_value"].round(8).tolist())
        assert len(unique_pvals) <= 71
        assert 0.4 < df_a["P_value"].mean() < 0.6


class TestLogL2WaldNull:
    """Analytic Wald-type null for ``log_l2``."""

    def test_synthetic_h0_pvalues_are_uniform(self):
        rng = np.random.default_rng(0)
        n_a, n_b, K = 4, 4, 30
        spectra = np.exp(rng.standard_normal((n_a + n_b, 5000, K)))
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        ks_stat, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"Wald p-values not uniform under H0: KS p={ks_p:.4f}"
        assert 0.45 < df["P_value"].mean() < 0.55

    def test_wald_breaks_permutation_floor_on_implanted_signal(self):
        """Wald can beat the exact-permutation p-value floor on strong signals."""
        rng = np.random.default_rng(7)
        n_per = 4
        n_genes, K = 50, 30
        a = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        b = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        b[:, :5, :3] *= 5.0
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df_perm = compare_two_groups(
            spectra, groups, statistic="log_l2", null="permutation", random_state=0
        )
        df_wald = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        for g in ("0", "1", "2", "3", "4"):
            p_perm = df_perm.loc[df_perm["Feature"] == g, "P_value"].iloc[0]
            p_wald = df_wald.loc[df_wald["Feature"] == g, "P_value"].iloc[0]
            assert p_wald < p_perm, f"gene {g}: Wald={p_wald:.3g} not < perm={p_perm:.3g}"
            assert p_wald < 1.0 / 71.0, f"gene {g}: Wald={p_wald:.3g} above perm floor"

    def test_wald_argument_is_ignored_for_welch_t_cauchy(self):
        """``welch_t_cauchy`` ignores ``null`` and still returns its schema."""
        rng = np.random.default_rng(3)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic="welch_t_cauchy", null="wald")
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()

    def test_unknown_null_raises(self):
        rng = np.random.default_rng(4)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="Unknown null"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="bootstrap")

    def test_small_df_emits_user_warning(self):
        """Wald warns when residual df is too small for stable variance estimates."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((3, 50, 20)))
        groups = np.array([0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='wald' at residual df=1"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        spectra = np.exp(rng.standard_normal((4, 50, 20)))
        groups = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='wald' at residual df=2"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")

    def test_full_sigma_calibrates_under_correlated_bins(self):
        """Full-covariance Wald should stay calibrated with correlated bins."""
        rng = np.random.default_rng(42)
        n_a, n_b, G, K = 4, 4, 2000, 30
        # Shared noise creates near rank-1 covariance across bins.
        shared = rng.standard_normal((n_a + n_b, G, 1)) * 1.0
        per_bin = rng.standard_normal((n_a + n_b, G, K)) * 0.2
        log_y = shared + per_bin
        spectra = np.exp(log_y)
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        ks_p = kstest(df["P_value"].to_numpy(), "uniform")[1]
        fpr05 = (df["P_value"] < 0.05).mean()
        assert fpr05 < 0.10, f"Full-Σ Wald should be near-calibrated; got Pr(p<.05)={fpr05:.3f}"
        assert (
            ks_p > 1e-3 or fpr05 < 0.07
        ), f"Full-Σ Wald should be roughly uniform; KS_p={ks_p:.3g} fpr05={fpr05:.3f}"

    def test_masked_wald_parity_with_unmasked_when_complete(self):
        """All-present masks should match unmasked Wald."""
        rng = np.random.default_rng(5)
        n_samples, n_genes, K = 6, 100, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df_un = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_m = (
            compare_two_groups_masked(spectra, groups, presence, statistic="log_l2", null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            df_un["Statistic"].to_numpy(), df_m["Statistic"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            df_un["P_value"].to_numpy(), df_m["P_value"].to_numpy(), atol=1e-12
        )

    def test_masked_wald_calibration_under_synthetic_missingness(self):
        """Random missingness should not strongly distort H0 false positives."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 8, 800, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0] * 4 + [1] * 4)

        df_full = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        fpr_full = (df_full["P_value"] < 0.05).mean()

        presence = rng.uniform(size=(n_samples, n_genes)) >= 0.25
        df_m = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="wald",
            min_samples_per_group=2,
        )
        valid = df_m["P_value"].notna()
        assert valid.sum() > 100, f"only {valid.sum()} testable genes after masking"
        fpr_m = (df_m.loc[valid, "P_value"] < 0.05).mean()

        assert 0.0 < fpr_m < 0.20, f"masked FPR={fpr_m:.3f} far from nominal"
        assert (
            0.5 * fpr_full < fpr_m < 2.0 * fpr_full + 0.02
        ), f"masked vs unmasked FPR mismatch: masked={fpr_m:.3f} unmasked={fpr_full:.3f}"

    def test_masked_wald_skips_genes_with_insufficient_presence(self):
        """Genes below per-group presence thresholds should report NaN p-values."""
        rng = np.random.default_rng(7)
        n_samples, n_genes, K = 6, 20, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        presence[1:3, 0] = False
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="wald",
            min_samples_per_group=2,
        )
        row = df[df["Feature"] == "0"].iloc[0]
        assert np.isnan(row["P_value"]), f"expected NaN, got {row['P_value']}"
        assert row["n_obs_A"] == 1
        valid_genes = df[df["Feature"] != "0"]
        assert valid_genes["P_value"].notna().all()

    def test_masked_wald_raises_when_no_eligible_genes(self):
        """A fully ineligible mask should raise instead of returning all-NaN results."""
        rng = np.random.default_rng(8)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        presence[0:3, :] = False
        with pytest.raises(ValueError, match="no genes meet"):
            compare_two_groups_masked(
                spectra,
                groups,
                presence,
                statistic="log_l2",
                null="wald",
                min_samples_per_group=2,
            )

    def test_masked_wald_argument_is_ignored_for_welch_t_cauchy(self):
        """Masked ``welch_t_cauchy`` ignores ``null`` like the unmasked path."""
        rng = np.random.default_rng(9)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="welch_t_cauchy",
            null="wald",
        )
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()


class TestCompareDesignsAndGLMWald:
    """GLM Wald comparisons for DataFrame, dict, and ndarray designs."""

    def test_binary_design_matches_groups_path_byte_close(self):
        """A binary GLM design should match the two-group Wald path."""
        import pandas as pd

        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((8, 200, 25)))
        groups = np.array(["WT"] * 4 + ["TG"] * 4)

        df_g = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        design = pd.DataFrame({"genotype": groups})
        df_d = (
            compare_glm(spectra, design, contrast="genotype", null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            df_g["P_value"].to_numpy(), df_d["P_value"].to_numpy(), atol=1e-10
        )
        np.testing.assert_allclose(
            df_g["Statistic"].to_numpy(), df_d["Statistic"].to_numpy(), atol=1e-10
        )

    def test_continuous_contrast_recovers_planted_signal(self):
        import pandas as pd

        rng = np.random.default_rng(7)
        n, n_genes, K = 8, 100, 25
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros((n_genes, K))
        beta[:5, :3] = 5.0
        log_y = beta[None, :, :] * x[:, None, None] + 0.3 * rng.standard_normal((n, n_genes, K))
        spectra = np.exp(log_y)
        design = pd.DataFrame({"time": x})
        df = compare_glm(spectra, design, contrast="time", null="wald")
        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10, f"missing planted: {top10}"

    def test_dict_contrast_normalization(self):
        """A dict contrast should match the equivalent column-name contrast."""
        import pandas as pd

        rng = np.random.default_rng(2)
        spectra = np.exp(rng.standard_normal((8, 30, 20)))
        design = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": np.arange(8)})
        df_dict = (
            compare_glm(spectra, design, contrast={"a": 1.0}, null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_str = (
            compare_glm(spectra, design, contrast="a", null="wald")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(df_dict["P_value"].to_numpy(), df_str["P_value"].to_numpy())

    def test_ndarray_design_with_intercept_only(self):
        rng = np.random.default_rng(3)
        n, n_genes, K = 6, 20, 18
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X = np.column_stack([np.ones(n), [0, 0, 0, 1, 1, 1]])
        df = compare_glm(spectra, X, contrast=np.array([0.0, 1.0]), null="wald")
        assert df.shape[0] == n_genes
        assert df["P_value"].between(0, 1).all()

    def test_compare_glm_rejects_permutation_null(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((6, 10, 12)))
        design = pd.DataFrame({"g": [0, 0, 0, 1, 1, 1]})
        with pytest.raises(NotImplementedError, match="Permutation null"):
            compare_glm(spectra, design, contrast="g", null="permutation")

    def test_compare_glm_rejects_non_log_l2(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((6, 10, 12)))
        design = pd.DataFrame({"g": [0, 0, 0, 1, 1, 1]})
        with pytest.raises(ValueError, match="only supports statistic='log_l2'"):
            compare_glm(spectra, design, contrast="g", statistic="welch_t_cauchy", null="wald")


class TestTwoGroupPower:
    """Power checks for implanted spectral differences."""

    def test_implanted_difference_is_recovered(self):
        rng = np.random.default_rng(7)
        n_per = 6
        n_genes, K = 50, 10
        a = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b[:, :5, :3] += 0.8
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=400, random_state=0)
        top10 = set(df.head(10)["Feature"].astype(str))
        implanted = {"0", "1", "2", "3", "4"}
        recovered = len(top10 & implanted)
        assert recovered >= 4, f"only recovered {recovered}/5 top-10: {top10}"


class TestStatisticAliases:
    """Statistic selection and frequency-weight behavior."""

    @pytest.mark.parametrize("stat", ["log_l2", "welch_t_cauchy"])
    def test_each_statistic_runs(self, stat):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 8, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic=stat, n_perm=50, random_state=0)
        assert df.shape[0] == 8
        assert {"Feature", "Statistic", "P_value", "P_adj"} <= set(df.columns)
        assert df["P_value"].between(0, 1).all()
        if stat == "welch_t_cauchy":
            assert "P_value_per_bin" in df.columns
            per_bin = np.stack(df["P_value_per_bin"].to_numpy())
            assert per_bin.shape == (8, 6)
            assert ((per_bin >= 0) & (per_bin <= 1)).all()

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            compare_two_groups(
                np.zeros((4, 3, 5)),
                np.array([0, 0, 1, 1]),
                statistic="bogus",
            )

    def test_log_l2_freq_weights(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 6, 4, 8
        base = rng.uniform(0.5, 1.5, size=(n_samples, n_genes, K))
        base[3:, 0, :2] *= 3.0
        base[3:, 1, -2:] *= 3.0
        groups = np.array([0, 0, 0, 1, 1, 1])
        df_equal = compare_two_groups(base, groups, statistic="log_l2", n_perm=200, random_state=0)
        low_pass = np.concatenate([np.ones(2), np.zeros(K - 2)])
        df_low = compare_two_groups(
            base,
            groups,
            statistic="log_l2",
            n_perm=200,
            random_state=0,
            freq_weights=low_pass,
        )
        assert df_low["Feature"].iloc[0] == "0"
        assert df_equal["Feature"].iloc[0] != df_low["Feature"].iloc[-1]

    def test_log_l2_freq_weights_validation(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(4, 3, 6))
        groups = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="length K="):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.ones(5),
            )
        with pytest.raises(ValueError, match="non-negative"):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            )


class TestScalarTestCalibration:
    """Scalar differential-expression calibration and power."""

    def test_welch_permutation_is_uniform_under_h0(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 300
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(
            values, groups, null="permutation", n_perm=300, random_state=0
        )
        ks_stat, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"DE-test p-values not uniform under H0, KS p={ks_p:.4f}"

    def test_welch_analytic_is_uniform_under_h0(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 1000
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(values, groups, null="wald")
        ks_stat, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic-Welch p-values not uniform under H0, KS p={ks_p:.4f}"

    def test_welch_analytic_matches_scipy(self):
        from scipy.stats import ttest_ind

        rng = np.random.default_rng(2)
        n_per, n_genes = 5, 50
        a = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b = rng.normal(0.3, 1.5, size=(n_per, n_genes))
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups, null="wald")
        df = df.set_index("Feature").loc[[str(i) for i in range(n_genes)]]
        scipy_p = ttest_ind(a, b, equal_var=False, axis=0).pvalue
        np.testing.assert_allclose(df["P_value"].to_numpy(), scipy_p, rtol=1e-9, atol=1e-300)

    def test_welch_analytic_breaks_perm_raw_p_floor(self):
        """Analytic Welch can beat the small-cohort permutation p-value floor."""
        rng = np.random.default_rng(3)
        n_per, n_genes = 4, 50
        a = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b[:, 0] += 8.0
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)

        df_perm = compare_two_groups_scalar(
            values, groups, null="permutation", n_perm=1000, random_state=0
        )
        df_welch = compare_two_groups_scalar(values, groups, null="wald")
        assert df_perm.iloc[0]["Feature"] == "0"
        assert df_welch.iloc[0]["Feature"] == "0"
        top_perm_raw = float(df_perm.iloc[0]["P_value"])
        top_welch_raw = float(df_welch.iloc[0]["P_value"])
        assert top_perm_raw >= 0.01, f"perm raw-p unexpectedly tight: {top_perm_raw}"
        assert top_welch_raw < 1e-3, f"analytic raw p too large: {top_welch_raw}"
        assert top_perm_raw / max(top_welch_raw, 1e-300) > 10.0

    def test_invalid_null_raises(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((6, 3))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="null must be"):
            compare_two_groups_scalar(values, groups, null="bogus")

    def test_implanted_mean_shift_recovered(self):
        rng = np.random.default_rng(7)
        n_per, n_genes = 6, 40
        a = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b[:, :5] += 2.0
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups, null="wald")
        top5 = set(df.head(5).Feature.astype(str).tolist())
        assert top5 == {"0", "1", "2", "3", "4"}


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
        assert rb_a.shape == rb_b.shape == (9,)

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
            n_bins=6,
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


class TestIncompleteData:
    """Masked two-group comparison behavior for missing features."""

    def test_masked_matches_unmasked_when_full(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 4, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((6, 4), dtype=bool)
        df_full = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=50, random_state=0)
        df_mask = compare_two_groups_masked(
            spectra, groups, presence, statistic="log_l2", n_perm=50, random_state=0
        )
        np.testing.assert_allclose(
            df_full.set_index("Feature").loc[df_mask["Feature"], "Statistic"].to_numpy(),
            df_mask["Statistic"].to_numpy(),
            rtol=1e-10,
        )
        assert (df_mask["n_obs_A"] == 3).all()
        assert (df_mask["n_obs_B"] == 3).all()

    def test_gene_skipped_when_below_min_samples_per_group(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 3, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, True, True],
                [True, True, True],
            ]
        )
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            n_perm=20,
            random_state=0,
            min_samples_per_group=2,
        )
        skipped_row = df[df.Feature == "0"].iloc[0]
        assert np.isnan(skipped_row["P_value"])
        assert np.isnan(skipped_row["P_adj"])
        assert skipped_row["n_obs_A"] == 3
        assert skipped_row["n_obs_B"] == 1


def _stub_spectra_groups(seed: int = 0, n_a: int = 4, n_b: int = 4, n_genes: int = 30, K: int = 12):
    """Build reusable spectra and labels for normalize_shape option tests."""

    rng = np.random.default_rng(seed)
    spec = rng.uniform(0.5, 2.0, size=(n_a + n_b, n_genes, K))
    groups = np.array([0] * n_a + [1] * n_b)
    return spec, groups


class TestNormalizeShapeComparisonOptions:
    """``normalize_shape=True`` delegates to manual shape normalization."""

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_two_groups_option_matches_manual_normalization(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_kw = compare_two_groups(spec, groups, **kw, normalize_shape=True)
        df_manual = compare_two_groups(normalize_shape(spec), groups, **kw)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            df_kw["Statistic"].to_numpy(),
            df_manual["Statistic"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_masked_two_groups_option_matches_manual_normalization(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        presence = np.ones((spec.shape[0], spec.shape[1]), dtype=bool)
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_kw = compare_two_groups_masked(spec, groups, presence, **kw, normalize_shape=True)
        df_manual = compare_two_groups_masked(normalize_shape(spec), groups, presence, **kw)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_glm_option_matches_manual_normalization(self):
        spec, _ = _stub_spectra_groups()
        design = pd.DataFrame({"x": np.arange(spec.shape[0], dtype=float)})
        df_kw = compare_glm(
            spec, design, "x", statistic="log_l2", null="wald", normalize_shape=True
        )
        df_manual = compare_glm(normalize_shape(spec), design, "x", statistic="log_l2", null="wald")
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )
