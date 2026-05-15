"""Tests for quadsv.multisample."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from quadsv.comparators import ComparatorIrregular
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
)
from quadsv.kernels.fft import power_spectrum_2d

# ---------------------------------------------------------------------------
# Test helpers for the AnnData-based ComparatorIrregular API (Phase D)
# ---------------------------------------------------------------------------


def _grid_to_adata(sample: np.ndarray, gene_names, spacing=(1.0, 1.0)):
    """Wrap a pre-rasterized ``(n_genes, ny, nx)`` array into an AnnData with a
    regular-grid ``obsm['spatial']``. The NUFFT path will evaluate the spectrum
    on (approximately) the same grid, and radial binning with explicit
    ``spacing`` makes feature vectors cross-sample comparable."""
    n_genes, ny, nx = sample.shape
    X = sample.reshape(n_genes, ny * nx).T  # (ny*nx, n_genes)
    yy, xx = np.meshgrid(
        np.arange(ny) * spacing[0],
        np.arange(nx) * spacing[1],
        indexing="ij",
    )
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
    a = ad.AnnData(X=X.astype(np.float64))
    a.var_names = list(gene_names)
    a.obsm["spatial"] = coords
    return a


def _samples_to_adata_list(samples, gene_names, spacings=None):
    spacings = spacings if spacings is not None else [(1.0, 1.0)] * len(samples)
    return [_grid_to_adata(s, gene_names, spacing=spacings[i]) for i, s in enumerate(samples)]


# ---------------------------------------------------------------------------
# Sanity: power spectrum
# ---------------------------------------------------------------------------


class TestPowerSpectrumSanity:
    def test_constant_image_has_only_dc(self):
        img = 3.0 * np.ones((16, 16))
        P = power_spectrum_2d(img, fft_solver="fft2")
        # All power concentrated at DC (k=0).
        assert P[0, 0] == pytest.approx((3.0 * 16 * 16) ** 2)
        P[0, 0] = 0.0
        np.testing.assert_allclose(P, 0.0, atol=1e-20)

    def test_translation_invariance(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((24, 32))
        shifted = np.roll(img, shift=(5, -7), axis=(0, 1))
        P1 = power_spectrum_2d(img, fft_solver="fft2")
        P2 = power_spectrum_2d(shifted, fft_solver="fft2")
        np.testing.assert_allclose(P1, P2, atol=1e-9)

    def test_rfft2_shape(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((10, 16))
        P = power_spectrum_2d(img, fft_solver="rfft2")
        assert P.shape == (10, 9)  # 16 // 2 + 1


# ---------------------------------------------------------------------------
# Radial binning
# ---------------------------------------------------------------------------


class TestRadialBinning:
    def test_isotropic_gaussian_bump_decreases_radially(self):
        """An isotropic Gaussian bump in space has a Gaussian PSD; radial spectrum decreases."""
        ny, nx = 32, 32
        y, x = np.meshgrid(np.arange(ny) - ny / 2, np.arange(nx) - nx / 2, indexing="ij")
        r2 = y**2 + x**2
        bump = np.exp(-r2 / (2 * 4.0**2))
        P = power_spectrum_2d(bump, fft_solver="fft2")
        rb = radial_bin_spectrum(P, grid_shape=(ny, nx), n_bins=20, fft_solver="fft2")
        # First (DC excluded) bin should have the most energy; last bin the least.
        assert rb[0] > rb[-1]
        # Roughly monotonically decreasing (allow small reshuffles in mid-range).
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


# ---------------------------------------------------------------------------
# Rotation alignment
# ---------------------------------------------------------------------------


class TestRotationAlignment:
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
        """One striped landmark per sample → recovered angle ≈ true angle."""
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

    def test_per_landmark_beats_mean_template(self):
        """With landmarks on perpendicular anisotropy axes, a mean-template
        would be near-isotropic and alignment would break down. Per-landmark
        alignment still locks onto the shared rotation because each landmark
        cross-correlates against its own same-index counterpart.
        """
        ny = nx = 96
        true_angle = 30.0

        # Two analytic landmarks with wave vectors on perpendicular axes.
        lm_h = TestRotationSimulation._sine_at_angle(ny, nx, 12.0, 0.0, 0.0)  # along +y
        lm_v = TestRotationSimulation._sine_at_angle(ny, nx, 0.0, 12.0, 0.0)  # along +x
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
        """Estimate rotation from one landmark, apply to an independent panel."""
        import scipy.ndimage

        ny = nx = 48
        true_angle = 18.0
        ref_landmark = self._stripes(ny, nx)
        cur_landmark = self._stripes_rotated(ny, nx, true_angle)

        # Target panel: three arbitrary genes per sample (distinct from the landmark).
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
    """Simulation-based validation with **analytic** rotations (we rotate the
    wave vector of a sinusoidal landmark directly, so no pixel-interpolation
    bias is introduced by the simulator).

    The estimator's accuracy is fundamentally limited by the FFT grid: a
    sinusoid whose rotated wave vector lands between integer bins has a
    spectrum peak that is off the nearest bin by up to ~one angular bin
    (~180/n_theta degrees at the lowest landmark radius). Larger grids and
    multiple landmarks at different radii push this down.
    """

    @staticmethod
    def _sine_at_angle(ny, nx, ky0, kx0, phi_deg):
        """Generate ``sin(2π (ky·y + kx·x) / N)`` with ``(ky, kx)`` rotated
        by ``phi_deg``. No interpolation."""
        phi = np.deg2rad(phi_deg)
        c, s = np.cos(phi), np.sin(phi)
        ky = ky0 * c - kx0 * s
        kx = ky0 * s + kx0 * c
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return np.sin(2 * np.pi * (ky * yy / ny + kx * xx / nx))

    @staticmethod
    def _canon_err(a, b):
        """Angular error modulo 180°."""
        d = np.abs(np.mod(a, 180.0) - np.mod(b, 180.0))
        return np.minimum(d, 180.0 - d)

    def test_multi_sample_recovery_with_multiple_landmarks(self):
        """Per-sample rotations drawn from U(-80, 80). Using 4 sinusoidal
        landmarks at different frequencies — the per-landmark alignment
        (sum of per-gene cross-correlations) averages out the per-landmark
        aliasing, so every sample's angle is recovered to within a few
        angular bins."""
        ny = nx = 96
        rng = np.random.default_rng(42)
        n_samples = 8
        # Distinct k0 — higher radius → finer angular resolution.
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
        # With 4 landmarks on a 96×96 grid the recovered angle is within ~3°
        # of truth in every sample (aliasing-limited).
        assert errs[1:].max() < 4.0, (
            f"per-sample errors (deg): {errs[1:].round(2).tolist()}; "
            f"true angles: {true_angles[1:].round(2).tolist()}; "
            f"recovered: {recovered[1:].round(2).tolist()}"
        )
        # Mean error should be well under 2° — most of the budget is max-bin.
        assert errs[1:].mean() < 2.0

    def test_more_landmarks_reduces_bias(self):
        """Adding more landmarks (at distinct frequencies) should reduce the
        mean rotation-recovery error. This is a key property of the
        landmarks-first API — the previous mean-template design averaged the
        landmarks first and then cross-correlated, which did not have this
        property."""
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
        """End-to-end: after rotation-correction, every sample's spectrum
        lands much closer to the reference than the un-corrected version.
        """
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


# ---------------------------------------------------------------------------
# Background normalization & residualization
# ---------------------------------------------------------------------------


class TestBackgroundNormalization:
    def test_identical_genes_become_unit_after_normalization(self):
        # Every gene has the same spectrum -> geo mean = same -> ratio = 1.
        spec = np.tile(np.arange(1.0, 6.0), (10, 1))  # (10 genes, K=5), all rows equal
        out = normalize_background(spec)
        np.testing.assert_allclose(out, np.ones_like(out), atol=1e-9)

    def test_preserves_shape(self):
        rng = np.random.default_rng(0)
        spec = rng.uniform(0.1, 10.0, size=(7, 12))
        out = normalize_background(spec)
        assert out.shape == spec.shape


class TestResidualization:
    def test_log_space_perfect_predictor_residual_is_unity(self):
        """If ``log gene = β_0 + Σ β_c log cov_c`` exactly, the (log-space)
        residual is zero and the exponentiated output equals 1.0."""
        rng = np.random.default_rng(0)
        cov = rng.uniform(0.1, 5.0, size=(2, 8))  # 2 covariates, K=8
        # Log-space linear combo: gene = exp(β_0) · cov_0^{β_1} · cov_1^{β_2}
        gene = np.exp(2.0 + 1.5 * np.log(cov[0]) - 0.7 * np.log(cov[1]))
        gene = np.tile(gene, (5, 1))
        out = normalize_covariates(gene, cov, fit_intercept=True)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="Last axis"):
            normalize_covariates(np.zeros((3, 5)), np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# Two-group test: calibration & power
# ---------------------------------------------------------------------------


class TestTwoGroupNullCalibration:
    def test_log_l2_pvalues_are_uniform_under_h0(self):
        # All samples drawn from same distribution -> p-values should be ~Uniform(0,1).
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
        """With ``n_perm_max`` above ``C(n, n_a)`` the test enumerates every
        distinct relabelling. The resulting p-values are **discrete** with
        values in ``{1/(M+1), ..., (M+1)/(M+1)}`` where ``M = C(n, n_a)``,
        but remain calibrated: repeated runs are deterministic (no RNG
        sampling), and under H0 the rank of the observed statistic is
        Uniform on ``{1, ..., M+1}``.
        """
        rng = np.random.default_rng(0)
        n_genes, K = 50, 10
        # 4 vs 4: only C(8, 4) = 70 distinct relabellings → exact path.
        spectra = rng.uniform(0.5, 5.0, size=(8, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df_a = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=5000, random_state=0)
        # Re-run with a different seed → same p-values (exact → RNG-free).
        df_b = compare_two_groups(
            spectra, groups, statistic="log_l2", n_perm=5000, random_state=999
        )
        np.testing.assert_allclose(
            df_a.set_index("Feature").loc[df_b["Feature"], "P_value"].to_numpy(),
            df_b["P_value"].to_numpy(),
        )
        # The p-value alphabet is 71 = C(8,4) + 1 values.
        unique_pvals = set(df_a["P_value"].round(8).tolist())
        assert len(unique_pvals) <= 71
        # Mean p-value under H0 should still be ≈ 0.5 (up to finite-gene noise).
        assert 0.4 < df_a["P_value"].mean() < 0.6


class TestLogL2WaldNull:
    """Analytic Wald-type null for ``log_l2`` (mixture-χ² tail via Liu)."""

    def test_synthetic_h0_pvalues_are_uniform(self):
        """Under H0 with iid genes, Wald p-values should pass KS-uniform."""
        rng = np.random.default_rng(0)
        n_a, n_b, K = 4, 4, 30
        # log-spectra ~ N(0, 1) → spectra ~ lognormal → realistic data scale.
        spectra = np.exp(rng.standard_normal((n_a + n_b, 5000, K)))
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        ks_stat, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"Wald p-values not uniform under H0: KS p={ks_p:.4f}"
        # Mean should also be near 0.5.
        assert 0.45 < df["P_value"].mean() < 0.55

    def test_wald_breaks_permutation_floor_on_implanted_signal(self):
        """An implanted strong shift gets p ≪ 1/(M+1) under Wald, vs flooring at
        the smallest exact-permutation value under permutation.
        """
        rng = np.random.default_rng(7)
        n_per = 4  # → C(8, 4) = 70 perms; floor = 1/71 ≈ 0.0141.
        n_genes, K = 50, 30
        a = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        b = np.exp(rng.normal(loc=0.0, scale=0.2, size=(n_per, n_genes, K)))
        # Strong shift on the first 5 genes' low-frequency bins.
        b[:, :5, :3] *= 5.0
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df_perm = compare_two_groups(
            spectra, groups, statistic="log_l2", null="permutation", random_state=0
        )
        df_wald = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        # All five implanted genes should pass through both filters.
        # Under permutation they all hit the floor 1/71 ≈ 0.0141.
        # Under Wald they should be far below the floor.
        for g in ("0", "1", "2", "3", "4"):
            p_perm = df_perm.loc[df_perm["Feature"] == g, "P_value"].iloc[0]
            p_wald = df_wald.loc[df_wald["Feature"] == g, "P_value"].iloc[0]
            assert p_wald < p_perm, f"gene {g}: Wald={p_wald:.3g} not < perm={p_perm:.3g}"
            assert p_wald < 1.0 / 71.0, f"gene {g}: Wald={p_wald:.3g} above perm floor"

    def test_liu_alias_retired(self):
        """The retired ``null='liu'`` alias must now raise. Rename guard for
        the alias-cleanup that left ``null='wald'`` as the single canonical
        token."""
        rng = np.random.default_rng(1)
        spectra = np.exp(rng.standard_normal((6, 100, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="Unknown null='liu'"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="liu")

    def test_wald_is_deterministic(self):
        rng = np.random.default_rng(2)
        spectra = np.exp(rng.standard_normal((6, 100, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Two calls with different seeds (Wald is RNG-free).
        df_a = compare_two_groups(spectra, groups, statistic="log_l2", null="wald", random_state=0)
        df_b = compare_two_groups(
            spectra, groups, statistic="log_l2", null="wald", random_state=999
        )
        np.testing.assert_array_equal(
            df_a.sort_values("Feature")["P_value"].to_numpy(),
            df_b.sort_values("Feature")["P_value"].to_numpy(),
        )

    def test_wald_argument_is_ignored_for_welch_t_cauchy(self):
        """``welch_t_cauchy`` has its own analytic null; the ``null`` argument
        is documented as ignored. With the package default
        ``null='wald'``, the call must succeed (no spurious rejection on
        a moot kwarg) and produce the cauchy-welch output schema."""
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
        """At residual df < 3 (n_a + n_b ≤ 4), the Wald path should warn the
        user that σ̂² is noisy and recommend welch_t_cauchy.
        """
        rng = np.random.default_rng(0)
        # 1v2 split → df = 1 + 2 - 2 = 1; should warn.
        spectra = np.exp(rng.standard_normal((3, 50, 20)))
        groups = np.array([0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='wald' at residual df=1"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        # 2v2 split → df = 2; still below the floor, should warn.
        spectra = np.exp(rng.standard_normal((4, 50, 20)))
        groups = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='wald' at residual df=2"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        # 3v3 split → df = 4; should NOT warn.
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        # warnings.simplefilter('error') to fail if any warning leaks.
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            compare_two_groups(spectra, groups, statistic="log_l2", null="wald")

    def test_full_sigma_calibrates_under_correlated_bins(self):
        """When bins are highly correlated within each gene, the diagonal-Σ
        Wald is anti-conservative; the full-Σ implementation must remain
        approximately calibrated.

        We synthesise H0 spectra whose log-form has a near rank-1 covariance
        across bins (one shared multiplicative noise term + small per-bin
        noise). Diagonal-Σ would predict a tail much lighter than truth and
        produce a heavy left-spike in the p-value histogram; full-Σ should
        track Uniform(0,1).
        """
        rng = np.random.default_rng(42)
        n_a, n_b, G, K = 4, 4, 2000, 30
        # Per-sample shared noise across all bins (creates the rank-1 component).
        shared = rng.standard_normal((n_a + n_b, G, 1)) * 1.0
        # Per-bin independent noise (small).
        per_bin = rng.standard_normal((n_a + n_b, G, K)) * 0.2
        log_y = shared + per_bin  # H0: same distribution in both groups
        spectra = np.exp(log_y)
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        ks_p = kstest(df["P_value"].to_numpy(), "uniform")[1]
        fpr05 = (df["P_value"] < 0.05).mean()
        # With diagonal Σ this would yield Pr(p<.05) ≈ 0.5+ (we measured 0.21
        # to 0.71 on real data with strongly-correlated bins). With full Σ
        # we should be within a few % of the nominal 0.05.
        assert fpr05 < 0.10, f"Full-Σ Wald should be near-calibrated; got Pr(p<.05)={fpr05:.3f}"
        # KS p > 0.001: histogram should look uniform-ish (with some drift
        # because the H0 model isn't perfectly captured by our pooling
        # — this is a real-data-like scenario, not a textbook iid sim).
        assert (
            ks_p > 1e-3 or fpr05 < 0.07
        ), f"Full-Σ Wald should be roughly uniform; KS_p={ks_p:.3g} fpr05={fpr05:.3f}"

    def test_masked_wald_parity_with_unmasked_when_complete(self):
        """When presence is all True, masked Wald should match unmasked Wald."""
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
        # Statistics must match modulo float64 round-off.
        np.testing.assert_allclose(
            df_un["Statistic"].to_numpy(), df_m["Statistic"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            df_un["P_value"].to_numpy(), df_m["P_value"].to_numpy(), atol=1e-12
        )

    def test_masked_wald_calibration_under_synthetic_missingness(self):
        """Under H0 with random missingness, masked Wald p-values should
        give Pr(p<.05) within a few percent of nominal α.

        Constructs a true-H0 cohort (synthetic iid Gaussian spectra, random
        labels) and compares the FPR with and without 25% per-cell
        missingness. The masked path should retain calibration; the
        ratio of FPRs (masked / unmasked) should be in [0.5, 2.0].
        """
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 8, 800, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0] * 4 + [1] * 4)

        # Unmasked baseline
        df_full = compare_two_groups(spectra, groups, statistic="log_l2", null="wald")
        fpr_full = (df_full["P_value"] < 0.05).mean()

        # 25% missingness, masked path
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
        # We need a meaningful test set
        assert valid.sum() > 100, f"only {valid.sum()} testable genes after masking"
        fpr_m = (df_m.loc[valid, "P_value"] < 0.05).mean()

        # Both should be near nominal — and similar to each other
        assert 0.0 < fpr_m < 0.20, f"masked FPR={fpr_m:.3f} far from nominal"
        assert (
            0.5 * fpr_full < fpr_m < 2.0 * fpr_full + 0.02
        ), f"masked vs unmasked FPR mismatch: masked={fpr_m:.3f} unmasked={fpr_full:.3f}"

    def test_masked_wald_skips_genes_with_insufficient_presence(self):
        """Genes whose present count in either arm < min_samples_per_group
        should report NaN p_value (consistent with permutation path)."""
        rng = np.random.default_rng(7)
        n_samples, n_genes, K = 6, 20, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        # Make gene 0: only 1 sample present in group A → should be skipped.
        presence[1:3, 0] = False  # group A has only sample 0 present
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="wald",
            min_samples_per_group=2,
        )
        # Find the row for gene "0"
        row = df[df["Feature"] == "0"].iloc[0]
        assert np.isnan(row["P_value"]), f"expected NaN, got {row['P_value']}"
        assert row["n_obs_A"] == 1
        # Other genes with full presence should have valid p-values
        valid_genes = df[df["Feature"] != "0"]
        assert valid_genes["P_value"].notna().all()

    def test_masked_wald_raises_when_no_eligible_genes(self):
        """If no gene meets min_samples_per_group, raise ValueError with
        a helpful message instead of silently returning NaNs."""
        rng = np.random.default_rng(8)
        n_samples, n_genes, K = 6, 10, 12
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Wipe out group A in every gene
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
        """Same as the unmasked counterpart: ``welch_t_cauchy`` ignores
        ``null=``, so passing the package default ``null='wald'`` must
        succeed (no spurious rejection on a moot kwarg)."""
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
    """`compare_glm` GLM Wald path + `Comparator.test_diff_freq(design=DataFrame)`."""

    def test_binary_design_matches_groups_path_byte_close(self):
        """Two-group Wald via 1-D labels and via DataFrame design must agree to ~1e-10.

        This is the central calibration anchor: the GLM Wald path with a
        single binary indicator literally recovers the binary Wald math
        from `compare_two_groups`, modulo float64 round-off in the OLS
        solve.
        """
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
        """A linear time-trend planted on the first few genes should be
        recovered by `contrast='time'`.
        """
        import pandas as pd

        rng = np.random.default_rng(7)
        n, n_genes, K = 8, 100, 25
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros((n_genes, K))
        beta[:5, :3] = 5.0  # planted on low-frequency bins of first 5 genes
        log_y = beta[None, :, :] * x[:, None, None] + 0.3 * rng.standard_normal((n, n_genes, K))
        spectra = np.exp(log_y)
        design = pd.DataFrame({"time": x})
        df = compare_glm(spectra, design, contrast="time", null="wald")
        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10, f"missing planted: {top10}"

    def test_dict_contrast_normalization(self):
        """A dict contrast spec must produce the same result as the
        equivalent ndarray.
        """
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
        """Numpy design matrix (caller-built) flows through compare_glm."""
        rng = np.random.default_rng(3)
        n, n_genes, K = 6, 20, 18
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        # Design = [intercept, group_indicator]
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

    def test_invalid_1d_design_not_binary(self):
        rng = np.random.default_rng(0)
        samples = [
            _grid_to_adata(
                rng.uniform(size=(5, 8, 8)),
                gene_names=[f"g{i}" for i in range(5)],
            )
            for _ in range(4)
        ]
        # 1-D array with 3 distinct values: rejected since the 1-D path is
        # binary-only. Wrap in a DataFrame to use the continuous path.
        # Validation is deferred to the test method (design is no longer a
        # constructor argument).
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(ValueError, match="exactly two distinct labels"):
            cmp.test_diff_freq(np.array([0, 1, 2, 2]))

    def test_invalid_design_missing(self):
        """`design` is a required positional on the test method."""
        rng = np.random.default_rng(0)
        samples = [
            _grid_to_adata(
                rng.uniform(size=(5, 8, 8)),
                gene_names=[f"g{i}" for i in range(5)],
            )
            for _ in range(4)
        ]
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(TypeError, match="missing.*required.*positional"):
            cmp.test_diff_freq()


class TestEffectiveRank:
    """K_eff (effective rank / participation ratio) primitives + accessors."""

    def test_effective_rank_identity_equals_K(self):
        """K_eff(I_K) = K — uniformly spread eigenvalues."""
        from quadsv import effective_rank

        for K in [3, 10, 30]:
            assert abs(effective_rank(np.eye(K)) - K) < 1e-12

    def test_effective_rank_rank_one_equals_one(self):
        """K_eff of an outer product vv^T is 1."""
        from quadsv import effective_rank

        rng = np.random.default_rng(0)
        v = rng.standard_normal(15)
        cov = np.outer(v, v)
        assert abs(effective_rank(cov) - 1.0) < 1e-10

    def test_effective_rank_bounds(self):
        """1 ≤ K_eff ≤ K for any PSD covariance."""
        from quadsv import effective_rank

        rng = np.random.default_rng(1)
        for _ in range(20):
            K = rng.integers(2, 30)
            X = rng.standard_normal((50, K))
            cov = X.T @ X / 50
            ke = effective_rank(cov)
            assert 1.0 - 1e-10 <= ke <= K + 1e-10, f"K_eff={ke} not in [1, {K}]"

    def test_effective_rank_with_weights_changes_value(self):
        """Non-uniform weights skew the effective rank."""
        from quadsv import effective_rank

        K = 20
        cov = np.eye(K)
        # Uniform weights → still K
        ke_uniform = effective_rank(cov, weights=np.ones(K) / K)
        assert abs(ke_uniform - K) < 1e-10
        # Concentrate all weight on one bin → K_eff drops to 1
        w_concentrated = np.zeros(K)
        w_concentrated[0] = 1.0
        ke_concentrated = effective_rank(cov, weights=w_concentrated)
        assert abs(ke_concentrated - 1.0) < 1e-10

    def test_effective_rank_invalid_inputs(self):
        from quadsv import effective_rank

        with pytest.raises(ValueError, match="square 2D matrix"):
            effective_rank(np.zeros((5, 4)))
        with pytest.raises(ValueError, match="non-negative"):
            effective_rank(np.eye(5), weights=np.array([1, 1, 1, 1, -1]))

    def test_gene_pattern_diversity_low_vs_high_heterogeneity(self):
        """Constant spectra + rank-1 perturbation → K_eff close to 1.
        Iid noise across genes → K_eff close to K."""
        from quadsv import gene_pattern_diversity

        rng = np.random.default_rng(0)
        K = 10
        # Rank-1 heterogeneity: each gene gets the same shape, scaled by a
        # gene-specific factor. After centering log → all residuals along
        # one direction → K_eff ≈ 1.
        shape = np.linspace(1, 5, K)
        gene_scales = rng.uniform(0.5, 2.0, size=200)
        spectra_rank1 = np.exp(np.log(shape)[None, :] + np.log(gene_scales)[:, None])
        ke_rank1 = gene_pattern_diversity(spectra_rank1)
        assert ke_rank1 < 1.5, f"expected K_eff close to 1, got {ke_rank1:.2f}"
        # Iid: each (gene, bin) is independent log-normal → K_eff close to K
        spectra_iid = np.exp(rng.standard_normal((200, K)))
        ke_iid = gene_pattern_diversity(spectra_iid)
        assert ke_iid > K * 0.5, f"expected K_eff > {K/2}, got {ke_iid:.2f}"

    def test_gene_pattern_diversity_random_spectra_near_K(self):
        from quadsv import gene_pattern_diversity

        rng = np.random.default_rng(2)
        spectra = np.exp(rng.standard_normal((500, 20)))
        ke = gene_pattern_diversity(spectra)
        # iid log-normal across genes — covariance ≈ I/G after centring →
        # K_eff close to K=20
        assert ke > 15.0, f"expected K_eff close to 20, got {ke:.2f}"

    def test_within_group_pattern_diversity_real_data_like(self):
        """Synthetic cohort with rank-1 within-group structure: K_eff ≈ 1."""
        from quadsv import within_group_pattern_diversity

        rng = np.random.default_rng(3)
        n_a, n_b, G, K = 4, 4, 800, 20
        # Per-sample shared scalar noise across all bins → near rank-1 Σ
        scalar = rng.standard_normal((n_a + n_b, G, 1))
        spectra = np.exp(scalar)  # All bins identical → singular Σ
        groups = np.array([0] * n_a + [1] * n_b)
        ke = within_group_pattern_diversity(spectra, groups)
        assert ke < 2.0, f"expected K_eff close to 1, got {ke:.2f}"

        # Independent noise per bin → near rank-K Σ
        spectra_iid = np.exp(rng.standard_normal((n_a + n_b, G, K)))
        ke_iid = within_group_pattern_diversity(spectra_iid, groups)
        assert ke_iid > K * 0.5, f"expected K_eff > {K/2}, got {ke_iid:.2f}"

    def test_comparator_effective_rank_within_group(self):
        from quadsv.comparators import ComparatorIrregular

        rng = np.random.default_rng(4)
        n_per = 4
        # Wrap synthetic spectra into AnnData via the test helper
        adatas = []
        for _ in range(2 * n_per):
            adatas.append(
                _grid_to_adata(
                    rng.uniform(size=(200, 8, 8)), gene_names=[f"g{j}" for j in range(200)]
                )
            )
        groups = np.array([0] * n_per + [1] * n_per)
        cmp = ComparatorIrregular(
            adatas,
            gene_names=[f"g{j}" for j in range(200)],
            feature_mode="radial",
            n_radial_bins=15,
            presence_threshold=0.0,
        )
        cmp.compute_spectra(n_jobs=1, progress=False)
        ke = cmp.effective_rank(level="within_group", design=groups)
        assert isinstance(ke, float)
        assert 1.0 - 1e-9 <= ke <= 15.0 + 1e-9

    def test_comparator_effective_rank_per_sample(self):
        from quadsv.comparators import ComparatorIrregular

        rng = np.random.default_rng(5)
        n_per = 3
        adatas = []
        n_total = 2 * n_per
        for _ in range(n_total):
            adatas.append(
                _grid_to_adata(
                    rng.uniform(size=(150, 8, 8)), gene_names=[f"g{j}" for j in range(150)]
                )
            )
        cmp = ComparatorIrregular(
            adatas,
            gene_names=[f"g{j}" for j in range(150)],
            feature_mode="radial",
            n_radial_bins=12,
            presence_threshold=0.0,
        )
        cmp.compute_spectra(n_jobs=1, progress=False)
        ke_arr = cmp.effective_rank(level="per_sample")
        assert ke_arr.shape == (n_total,)
        assert np.all(ke_arr >= 1.0 - 1e-9)
        assert np.all(ke_arr <= 12.0 + 1e-9)


class TestTwoGroupPower:
    def test_implanted_difference_is_recovered(self):
        rng = np.random.default_rng(7)
        n_per = 6
        n_genes, K = 50, 10
        # Group A: spectra ~ N(1, 0.1)
        a = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        # Implant a strong shift on the first 5 genes' low-frequency bins for group B.
        b[:, :5, :3] += 0.8
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=400, random_state=0)
        # The 5 implanted genes (named "0".."4") should rank in the top 10.
        top10 = set(df.head(10)["Feature"].astype(str))
        implanted = {"0", "1", "2", "3", "4"}
        recovered = len(top10 & implanted)
        assert recovered >= 4, f"only recovered {recovered}/5 top-10: {top10}"


class TestStatisticAliases:
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
            # Each entry is an (K,) array of per-bin p-values in [0, 1].
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
        """Non-uniform weights should shift gene ranking compared to uniform."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 6, 4, 8
        # Gene 0: low-frequency difference only.
        # Gene 1: high-frequency difference only.
        base = rng.uniform(0.5, 1.5, size=(n_samples, n_genes, K))
        base[3:, 0, :2] *= 3.0  # low-freq bump in group B for gene 0
        base[3:, 1, -2:] *= 3.0  # high-freq bump in group B for gene 1
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Uniform weights: both genes score similarly.
        df_equal = compare_two_groups(base, groups, statistic="log_l2", n_perm=200, random_state=0)
        # Low-pass weights: gene 0 should come out on top.
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
        # Sanity: equal-weights result is different from the low-pass ranking.
        assert df_equal["Feature"].iloc[0] != df_low["Feature"].iloc[-1]

    def test_log_l2_freq_weights_validation(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(4, 3, 6))
        groups = np.array([0, 0, 1, 1])
        # Wrong length:
        with pytest.raises(ValueError, match="length K="):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.ones(5),
            )
        # Negative weight:
        with pytest.raises(ValueError, match="non-negative"):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            )


# ---------------------------------------------------------------------------
# End-to-end: ComparatorIrregular
# ---------------------------------------------------------------------------


class TestComparatorIrregularEndToEnd:
    def test_pipeline_radial_runs_and_finds_implanted_gene(self):
        rng = np.random.default_rng(3)
        n_per = 4
        ny = nx = 32
        n_genes = 10
        gene_names = [f"g{i}" for i in range(n_genes)]

        def make_sample(group: int) -> np.ndarray:
            x = rng.standard_normal((n_genes, ny, nx)) * 0.1
            if group == 1:
                yy = np.arange(ny)[:, None]
                stripes = np.broadcast_to(np.sin(2 * np.pi * yy / 16.0), (ny, nx))
                x[0] += stripes * 1.5
            return x

        samples = [make_sample(0) for _ in range(n_per)] + [make_sample(1) for _ in range(n_per)]
        groups = np.array([0] * n_per + [1] * n_per)

        cmp = (
            ComparatorIrregular(_samples_to_adata_list(samples, gene_names), gene_names)
            .compute_spectra()
            .normalize_background()
        )
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=300, random_state=0)
        assert df["Feature"].iloc[0] == "g0", f"expected g0 first, got {df.head().to_dict()}"

    def test_pipeline_residualize_runs(self):
        rng = np.random.default_rng(0)
        n_per = 3
        ny = nx = 16
        gene_names = [f"g{i}" for i in range(4)]
        samples = [rng.standard_normal((4, ny, nx)) for _ in range(2 * n_per)]
        covariates = [rng.standard_normal((1, ny, nx)) for _ in range(2 * n_per)]
        groups = np.array([0] * n_per + [1] * n_per)
        cmp = ComparatorIrregular(_samples_to_adata_list(samples, gene_names), gene_names)
        cmp.compute_spectra().normalize_covariates(covariates)
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=50, random_state=0)
        assert df.shape[0] == 4

    def test_invalid_groups_raises(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4))] * 3, gene_names)
        cmp = ComparatorIrregular(adatas, gene_names).compute_spectra()
        with pytest.raises(ValueError, match="exactly two distinct"):
            cmp.test_diff_freq(np.array([0, 1, 2]))

    def test_must_compute_spectra_before_test(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        with pytest.raises(RuntimeError, match=r"\.compute_spectra\(\)"):
            cmp.test_diff_freq(np.array([0, 1]))


class TestNormalizeCovariatesObsKeys:
    """``normalize_covariates`` polymorphic input mode: shared ``obs`` column
    names per AnnData sample. Mirrors the per-spot covariate workflow users
    typically have on cell-type proportion maps."""

    @staticmethod
    def _build_samples(n_samples=4, n_spots=200, n_genes=4, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_samples):
            X = rng.standard_normal((n_spots, n_genes))
            a = ad.AnnData(X=X)
            a.var_names = [f"g{i}" for i in range(n_genes)]
            a.obsm["spatial"] = rng.uniform(0, 50, size=(n_spots, 2))
            a.obs["cov_a"] = rng.uniform(0.0, 1.0, size=n_spots)
            a.obs["cov_b"] = rng.uniform(0.0, 1.0, size=n_spots)
            a.obs["batch"] = pd.Categorical(["A", "B"] * (n_spots // 2))
            out.append(a)
        return out

    def test_obs_key_path_runs_and_mutates_spectra(self):
        """Calling with a ``Sequence[str]`` is interpreted as obs-column
        names; spectra_ should change in place."""
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra().normalize_background()
        before = cmp.spectra_.copy()
        ret = cmp.normalize_covariates(["cov_a", "cov_b"])
        assert ret is cmp, "should be chainable"
        assert not np.array_equal(
            cmp.spectra_, before
        ), "spectra_ must change after residualisation"
        assert cmp.spectra_.shape == before.shape

    def test_key_missing_from_obs_and_var_raises_keyerror(self):
        """Resolution checks both obs.columns and var_names; if neither
        contains the key, raise with both lists in the message."""
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(KeyError, match="in neither obs.columns nor"):
            cmp.normalize_covariates(["does_not_exist"])

    def test_var_names_key_path(self):
        """A key that matches a gene in ``var_names`` is treated as a
        per-spot expression covariate (housekeeping-gene workflow)."""
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra().normalize_background()
        before = cmp.spectra_.copy()
        # g0 exists in var_names; use it as a covariate.
        cmp.normalize_covariates(["g0"])
        assert not np.array_equal(cmp.spectra_, before)
        assert cmp.spectra_.shape == before.shape

    def test_obs_takes_precedence_over_var_on_collision(self):
        """When the same name appears in both obs and var_names, the obs
        column wins (matches the user's mental model that they're naming
        a metadata column)."""
        samples = self._build_samples()
        # Inject a synthetic obs column whose name collides with a gene.
        for a in samples:
            a.obs["g0"] = np.linspace(0.0, 1.0, a.n_obs)
        cmp = ComparatorIrregular(samples).compute_spectra().normalize_background()
        before = cmp.spectra_.copy()
        cmp.normalize_covariates(["g0"])
        # Sanity: spectra moved, and the obs path runs (no float-cast error
        # on the linspace column).
        assert not np.array_equal(cmp.spectra_, before)

    def test_mixed_obs_and_var_keys_in_one_call(self):
        """obs and var_names keys can be mixed in a single call — they're
        resolved per-key, then the per-spot vectors are stacked into one
        block before NUFFT."""
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra().normalize_background()
        before = cmp.spectra_.copy()
        cmp.normalize_covariates(["cov_a", "g0"])  # obs + var_names
        assert not np.array_equal(cmp.spectra_, before)

    def test_obs_key_categorical_raises_value_error(self):
        """Categorical obs columns can't be cast to float — surface a clear
        error pointing at encoding."""
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(ValueError, match="cannot be cast to float"):
            cmp.normalize_covariates(["batch"])

    def test_array_path_still_works(self):
        """The legacy per-sample (n_cov, ny, nx) array input must keep
        working alongside the new key-list mode."""
        rng = np.random.default_rng(1)
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra().normalize_background()
        before = cmp.spectra_.copy()
        arrays = [rng.standard_normal((1, 8, 8)) for _ in samples]
        cmp.normalize_covariates(arrays)
        assert not np.array_equal(cmp.spectra_, before)

    def test_empty_sequence_rejected(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(ValueError, match="non-empty"):
            cmp.normalize_covariates([])

    def test_mixed_str_and_array_rejected(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra()
        with pytest.raises(TypeError, match="Mixed str and non-str"):
            cmp.normalize_covariates(["cov_a", np.zeros((1, 8, 8))])


# ---------------------------------------------------------------------------
# DC/AC decomposition & DE-vs-pattern orthogonality
# ---------------------------------------------------------------------------


class TestMeanCenteringMakesDcZero:
    """The FFT pipeline always mean-centres each gene's grid so the
    spectral DC bin is exactly zero — this orthogonalises the AC pattern
    test against the DC :func:`compare_two_groups_scalar` DE test."""

    def test_mean_center_yields_dc_zero(self):
        """The k=0 bin of the spectrum is numerically zero by construction."""
        rng = np.random.default_rng(0)
        sample = rng.standard_normal((3, 12, 14)) + 5.0  # non-zero mean
        spec = compute_sample_spectrum(sample, fft_solver="fft2")
        # The DC bin is at index (0, 0) for both fft2 and rfft2 layouts.
        np.testing.assert_allclose(spec[:, 0, 0], 0.0, atol=1e-18)

    def test_return_dc_reports_per_gene_grid_means(self):
        rng = np.random.default_rng(1)
        sample = rng.standard_normal((4, 8, 10)) + np.arange(4)[:, None, None]
        spec, dc = compute_sample_spectrum(sample, return_dc=True)
        np.testing.assert_allclose(dc, sample.mean(axis=(1, 2)), rtol=1e-12)
        # Spectrum shape preserved.
        assert spec.shape[0] == 4


class TestScalarTestCalibration:
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
        """null='wald' (analytic) p-values should be uniform under iid Gaussian H0."""
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 1000
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(values, groups, null="wald")
        ks_stat, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic-Welch p-values not uniform under H0, KS p={ks_p:.4f}"

    def test_welch_analytic_matches_scipy(self):
        """null='wald' p-values should agree with scipy.stats.ttest_ind(equal_var=False)."""
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
        """At small n the permutation raw-p has a floor at 1/(perms+1); analytic does not.

        4 vs 4 has C(8, 4) = 70 unique permutations, so the permutation null
        cannot give a raw p-value smaller than ~1/70 ≈ 0.014 even for an
        arbitrarily strong signal. The analytic Welch path can give raw p
        many orders of magnitude smaller. This is the floor that translates
        downstream into the BH-FDR power problem on small cohorts.
        """
        rng = np.random.default_rng(3)
        n_per, n_genes = 4, 50
        a = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        # Implant a very strong shift so the analytic test gives a tiny p.
        b[:, 0] += 8.0
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)

        df_perm = compare_two_groups_scalar(
            values, groups, null="permutation", n_perm=1000, random_state=0
        )
        df_welch = compare_two_groups_scalar(values, groups, null="wald")
        # Most-significant gene should be the same in both rankings (gene 0).
        assert df_perm.iloc[0]["Feature"] == "0"
        assert df_welch.iloc[0]["Feature"] == "0"
        top_perm_raw = float(df_perm.iloc[0]["P_value"])
        top_welch_raw = float(df_welch.iloc[0]["P_value"])
        # Permutation raw-p floor at n=4v4: ~1 / C(8, 4) ≈ 0.014.
        assert top_perm_raw >= 0.01, f"perm raw-p unexpectedly tight: {top_perm_raw}"
        # Analytic should be < 1e-3 for a 8σ implant on a non-degenerate Welch t.
        assert top_welch_raw < 1e-3, f"analytic raw p too large: {top_welch_raw}"
        # The analytic-vs-perm raw-p ratio at the top gene should be at least 10×.
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
        b[:, :5] += 2.0  # large mean shift on genes 0..4
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups, null="wald")
        top5 = set(df.head(5).Feature.astype(str).tolist())
        assert top5 == {"0", "1", "2", "3", "4"}


class TestDeAndPatternOrthogonality:
    def test_pure_dc_shift_does_not_light_up_pattern_test(self):
        """A gene with only a mean-shift in one group (identical pattern otherwise)
        should be highly significant for DE but NOT for the pattern test."""
        rng = np.random.default_rng(0)
        n_per = 5
        ny = nx = 24
        n_genes = 6
        # Shared spatial "pattern" per gene (shared across all samples).
        pattern = rng.standard_normal((n_genes, ny, nx))

        samples = []
        for _ in range(2 * n_per):
            samples.append(pattern + 0.05 * rng.standard_normal((n_genes, ny, nx)))
        # Add a big mean shift to gene 0 only for group 1.
        for i in range(n_per, 2 * n_per):
            samples[i][0] += 10.0

        groups = np.array([0] * n_per + [1] * n_per)
        gene_names = [f"g{i}" for i in range(n_genes)]
        cmp = ComparatorIrregular(
            _samples_to_adata_list(samples, gene_names),
            gene_names=gene_names,
            n_radial_bins=8,
        ).compute_spectra()

        de = cmp.test_diff_expr(groups, n_perm=400, random_state=0)
        pattern_df = cmp.test_diff_freq(groups, n_perm=400, random_state=0)

        de_g0 = de.set_index("Feature").loc["g0"]
        pat_g0 = pattern_df.set_index("Feature").loc["g0"]

        # g0 should be the top DE hit.
        assert de.Feature.iloc[0] == "g0"
        assert de_g0.P_value < 0.05
        # The pattern test should NOT find g0 uniquely significant — its pattern
        # is identical between groups by construction.
        assert pat_g0.P_value > 0.05

    def test_dc_bin_always_zero_under_mean_centering(self):
        """The pipeline always mean-centres before FFT, so the DC bin is
        exactly zero in the returned spectrum — no DE signal can leak into
        the pattern test."""
        rng = np.random.default_rng(2)
        sample = rng.standard_normal((3, 12, 12)) + 4.0  # non-zero mean
        spec = compute_sample_spectrum(sample, fft_solver="fft2")
        np.testing.assert_allclose(spec[:, 0, 0], 0.0, atol=1e-18)


class TestComparatorIrregularDcAccess:
    def test_fit_populates_dc(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((3, 8, 10)) + s for s in range(4)]
        gene_names = ["a", "b", "c"]
        cmp = ComparatorIrregular(
            _samples_to_adata_list(samples, gene_names), gene_names
        ).compute_spectra()
        assert cmp.dc_ is not None
        assert cmp.dc_.shape == (4, 3)
        # DC equals per-sample grid mean of the raw signal.
        expected = np.array([samples[i].mean(axis=(1, 2)) for i in range(4)])
        np.testing.assert_allclose(cmp.dc_, expected, rtol=1e-12)

    def test_test_diff_expr_requires_compute_spectra(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        with pytest.raises(RuntimeError, match=r"\.compute_spectra\(\)"):
            cmp.test_diff_expr(np.array([0, 1]))


# ---------------------------------------------------------------------------
# normalize_shape: magnitude-invariant spectrum shapes
# ---------------------------------------------------------------------------


class TestShapeNormalize:
    def test_sum_to_one_along_axis(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.1, 10.0, size=(4, 7, 12))
        out = normalize_shape(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-12)

    def test_cancels_scalar_rescale(self):
        """Two rows that differ only by a positive scalar get the same shape."""
        rng = np.random.default_rng(1)
        row = rng.uniform(0.5, 3.0, size=10)
        scales = np.array([[0.3], [1.0], [50.0]])
        stack = scales * row[None, :]
        out = normalize_shape(stack, axis=-1)
        # All three rows become identical probability vectors after L1
        # normalization (the shared shape of the row).
        np.testing.assert_allclose(out[0], out[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[1], out[2], rtol=1e-10, atol=1e-12)

    def test_preserves_shape(self):
        x = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 8, 6))
        assert normalize_shape(x).shape == x.shape

    def test_diff_freq_normalize_shape_kwarg_matches_standalone(self):
        """``cmp.test_diff_freq(..., normalize_shape=True)`` produces the same
        result as calling the standalone ``compare_two_groups`` with
        ``normalize_shape=True`` on ``cmp.spectra_``."""
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((4, 12, 14)) for _ in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2", "g3"]
        cmp = (
            ComparatorIrregular(
                _samples_to_adata_list(samples, gene_names), gene_names, n_radial_bins=8
            )
            .compute_spectra()
            .normalize_background()
        )
        df_kw = cmp.test_diff_freq(groups, statistic="log_l2", null="wald", normalize_shape=True)
        df_manual = compare_two_groups(
            cmp.spectra_,
            groups,
            gene_names=cmp.gene_names,
            statistic="log_l2",
            null="wald",
            normalize_shape=True,
        )
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_diff_freq_normalize_shape_kwarg_is_non_destructive(self):
        """``cmp.spectra_`` must be byte-identical before vs after a
        ``test_diff_freq(..., normalize_shape=True)`` call."""
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((4, 12, 14)) for _ in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2", "g3"]
        cmp = (
            ComparatorIrregular(
                _samples_to_adata_list(samples, gene_names), gene_names, n_radial_bins=8
            )
            .compute_spectra()
            .normalize_background()
        )
        before = cmp.spectra_.copy()
        cmp.test_diff_freq(groups, statistic="log_l2", null="wald", normalize_shape=True)
        np.testing.assert_array_equal(cmp.spectra_, before)


# ---------------------------------------------------------------------------
# Physical-frequency binning (radial_bin_spectrum with spacing/edges)
# ---------------------------------------------------------------------------


class TestPhysicalFrequencyBinning:
    def test_physical_spacing_changes_bin_scale(self):
        """With the same spectrum, larger spacing -> lower Nyquist -> smaller max bin edge."""
        rng = np.random.default_rng(0)
        img = rng.standard_normal((40, 40))

        P = power_spectrum_2d(img, fft_solver="rfft2")
        rb1 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(1.0, 1.0)
        )
        rb2 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(10.0, 10.0)
        )
        # Per-bin values identical when edges auto-span [0, Nyquist] — only the
        # physical labelling of the axis changes between the two calls.
        np.testing.assert_allclose(rb1, rb2, rtol=1e-10)

    def test_common_edges_across_heterogeneous_grids(self):
        """Explicit edges let different-shape samples map onto the same bin grid."""
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
        # exclude_dc=True by default drops the first bin, so output length = 10 - 1 = 9.
        assert rb_a.shape == rb_b.shape == (9,)

    def test_explicit_edges_non_monotonic_raises(self):
        P = np.zeros((8, 5))
        with pytest.raises(ValueError, match="monotonically"):
            radial_bin_spectrum(
                P, grid_shape=(8, 8), fft_solver="rfft2", edges=np.array([0.0, 0.5, 0.2])
            )


class TestComparatorIrregularWithSpacings:
    def test_physical_spacings_produce_comparable_bins(self):
        """ComparatorIrregular with per-sample auto-grids handles heterogeneous shapes."""
        rng = np.random.default_rng(0)
        shapes = [(32, 40), (30, 42), (34, 38), (33, 41)]
        samples = [rng.standard_normal((3, ny, nx)) for (ny, nx) in shapes]
        gene_names = ["g0", "g1", "g2"]
        cmp = ComparatorIrregular(
            _samples_to_adata_list(samples, gene_names),
            gene_names=gene_names,
            n_radial_bins=8,
        ).compute_spectra()
        # 8 edges -> 7 bins after DC-drop.
        assert cmp.spectra_.shape == (4, 3, 7)
        assert cmp.freq_edges is not None
        assert cmp.freq_edges.shape == (9,)


class TestIncompleteData:
    def test_masked_matches_unmasked_when_full(self):
        """With all-True presence mask, masked == unmasked (same rng)."""
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 4, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((6, 4), dtype=bool)
        df_full = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=50, random_state=0)
        df_mask = compare_two_groups_masked(
            spectra, groups, presence, statistic="log_l2", n_perm=50, random_state=0
        )
        # Features line up after sort; the statistic column agrees exactly.
        np.testing.assert_allclose(
            df_full.set_index("Feature").loc[df_mask["Feature"], "Statistic"].to_numpy(),
            df_mask["Statistic"].to_numpy(),
            rtol=1e-10,
        )
        # n_obs columns should be fully populated at n_samples each.
        assert (df_mask["n_obs_A"] == 3).all()
        assert (df_mask["n_obs_B"] == 3).all()

    def test_gene_skipped_when_below_min_samples_per_group(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 3, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Gene 0 only observed in one sample of group B → must be skipped.
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

    def test_presence_threshold_propagates_to_comparator(self):
        import anndata as ad

        rng = np.random.default_rng(0)
        ny = nx = 12
        gene_names = ["g0", "g1", "g2"]
        # Build four samples, each on the same regular grid.
        samples = []
        for _ in range(4):
            coords = (
                np.stack(
                    np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"),
                    axis=-1,
                )
                .reshape(-1, 2)
                .astype(float)
            )
            X = rng.uniform(0.1, 1.0, size=(ny * nx, 3))
            a = ad.AnnData(X=X)
            a.var_names = gene_names
            a.obsm["spatial"] = coords
            samples.append(a)
        # Zero gene 0 in samples 0 and 1 → presence_ will drop it there.
        for i in (0, 1):
            samples[i].X[:, 0] = 0.0
        groups = np.array([0, 0, 1, 1])
        cmp = ComparatorIrregular(
            samples,
            gene_names,
            presence_threshold=0.5,
        ).compute_spectra()
        assert cmp.presence_.shape == (4, 3)
        # Gene 0 should be absent in the two samples we zeroed, present elsewhere.
        assert not cmp.presence_[0, 0]
        assert not cmp.presence_[1, 0]
        assert cmp.presence_[2, 0]
        # Running test_diff_freq should dispatch to the masked path and return
        # n_obs_A / n_obs_B columns.
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=20, random_state=0)
        assert {"n_obs_A", "n_obs_B"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# normalize_shape kwarg on the comparison functions
# ---------------------------------------------------------------------------


def _stub_spectra_groups(seed: int = 0, n_a: int = 4, n_b: int = 4, n_genes: int = 30, K: int = 12):
    rng = np.random.default_rng(seed)
    spec = rng.uniform(0.5, 2.0, size=(n_a + n_b, n_genes, K))
    groups = np.array([0] * n_a + [1] * n_b)
    return spec, groups


class TestNormalizeShapeKwargCompareTwoGroups:
    """``normalize_shape`` kwarg on ``compare_two_groups``."""

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_default_false_unchanged(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_a = compare_two_groups(spec, groups, **kw)
        df_b = compare_two_groups(spec, groups, **kw, normalize_shape=False)
        # Drop the per-bin object column for welch_t_cauchy (object dtype
        # confuses pandas equality assertions).
        for c in ("P_value_per_bin",):
            df_a = df_a.drop(columns=c, errors="ignore")
            df_b = df_b.drop(columns=c, errors="ignore")
        pd.testing.assert_frame_equal(df_a, df_b)

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_kwarg_true_matches_manual(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_kw = compare_two_groups(spec, groups, **kw, normalize_shape=True)
        df_manual = compare_two_groups(normalize_shape(spec), groups, **kw)
        # Sort by Feature so row ordering is consistent between calls.
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

    def test_calibrated_under_h0_log_l2_wald(self):
        # Random spectra + random labels → P-values from the shape path
        # should be approximately uniform on [0, 1] under H0.
        rng = np.random.default_rng(123)
        spec = rng.uniform(0.5, 2.0, size=(8, 200, 12))
        groups = np.array([0, 1] * 4)
        df = compare_two_groups(
            spec,
            groups,
            statistic="log_l2",
            null="wald",
            normalize_shape=True,
        )
        ks_stat = kstest(df["P_value"].to_numpy(), "uniform").pvalue
        assert ks_stat > 1e-3, f"shape-path H0 p-value ks-uniform p={ks_stat:.3g}"


class TestNormalizeShapeKwargCompareTwoGroupsMasked:
    """``normalize_shape`` kwarg on ``compare_two_groups_masked``."""

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_default_false_unchanged(self, statistic, null):
        spec, groups = _stub_spectra_groups()
        presence = np.ones((spec.shape[0], spec.shape[1]), dtype=bool)
        kw = {"statistic": statistic, "null": null, "n_perm": 200, "random_state": 0}
        df_a = compare_two_groups_masked(spec, groups, presence, **kw)
        df_b = compare_two_groups_masked(spec, groups, presence, **kw, normalize_shape=False)
        for c in ("P_value_per_bin",):
            df_a = df_a.drop(columns=c, errors="ignore")
            df_b = df_b.drop(columns=c, errors="ignore")
        pd.testing.assert_frame_equal(df_a, df_b)

    @pytest.mark.parametrize(
        "statistic,null",
        [
            ("log_l2", "wald"),
            ("log_l2", "permutation"),
            ("welch_t_cauchy", "permutation"),
        ],
    )
    def test_kwarg_true_matches_manual(self, statistic, null):
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


class TestNormalizeShapeKwargCompareDesigns:
    """``normalize_shape`` kwarg on ``compare_glm``."""

    def test_default_false_unchanged(self):
        spec, _ = _stub_spectra_groups()
        design = pd.DataFrame({"x": np.arange(spec.shape[0], dtype=float)})
        df_a = compare_glm(spec, design, "x", statistic="log_l2", null="wald")
        df_b = compare_glm(
            spec, design, "x", statistic="log_l2", null="wald", normalize_shape=False
        )
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_kwarg_true_matches_manual(self):
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

    def test_calibrated_under_h0(self):
        rng = np.random.default_rng(7)
        spec = rng.uniform(0.5, 2.0, size=(10, 200, 12))
        design = pd.DataFrame({"x": rng.standard_normal(10)})
        df = compare_glm(spec, design, "x", statistic="log_l2", null="wald", normalize_shape=True)
        ks_p = kstest(df["P_value"].to_numpy(), "uniform").pvalue
        assert ks_p > 1e-3, f"compare_glm shape-path H0 ks-uniform p={ks_p:.3g}"


# ---------------------------------------------------------------------------
# Unified normalize_* surface API: signatures + ImportError on old names
# ---------------------------------------------------------------------------


class TestNormalizationApiUnification:
    def test_old_names_removed(self):
        """The hard rename should make the old names unimportable."""
        import importlib

        mod = importlib.import_module("quadsv.comparators.multisample")
        for old in ("normalize_by_background", "residualize_against_covariates", "shape_normalize"):
            assert not hasattr(
                mod, old
            ), f"{old} should have been removed from multisample after the rename"
            assert (
                old not in mod.__all__
            ), f"{old} still listed in multisample.__all__ after the rename"

    def test_three_normalizers_share_eps_default(self):
        """All three normalize_* helpers expose ``eps=1e-12`` (used inside the
        per-function log/divide guard)."""
        import inspect

        for fn in (normalize_background, normalize_covariates, normalize_shape):
            sig = inspect.signature(fn)
            assert "eps" in sig.parameters, f"{fn.__name__} missing eps kwarg"
            assert (
                sig.parameters["eps"].default == 1e-12
            ), f"{fn.__name__} eps default differs from 1e-12"

    def test_normalize_covariates_output_strictly_positive(self):
        """Log-space formulation: output is exp(log P - X β̂), always > 0."""
        rng = np.random.default_rng(0)
        gene = rng.lognormal(size=(50, 10))
        cov = rng.lognormal(size=(3, 10))
        out = normalize_covariates(gene, cov)
        assert (out > 0).all(), "log-space normalize_covariates must return > 0"

    def test_normalize_covariates_commutes_with_background_in_log_space(self):
        """``bg`` is left-mult by a projection on the genes axis; ``cov`` is
        right-mult by a projection on the bins axis. They commute exactly
        on log-spectra. Verify via the public (exponentiated) outputs."""
        rng = np.random.default_rng(0)
        spec = rng.lognormal(size=(40, 12))  # (G, K)
        cov = rng.lognormal(size=(2, 12))  # (n_cov, K)
        order_a = normalize_covariates(normalize_background(spec, axis=-2), cov)
        order_b = normalize_background(normalize_covariates(spec, cov), axis=-2)
        np.testing.assert_allclose(order_a, order_b, rtol=1e-10, atol=1e-12)

    def test_normalize_covariates_no_covariates_reduces_to_geo_mean_centering(
        self,
    ):
        """With fit_intercept=True and no covariates, the output equals each
        gene's spectrum divided by its cross-bin geometric mean."""
        rng = np.random.default_rng(0)
        spec = rng.lognormal(size=(15, 9))
        out = normalize_covariates(spec, np.zeros((0, spec.shape[-1])))
        geo_mean_per_gene = np.exp(np.mean(np.log(spec + 1e-12), axis=-1, keepdims=True))
        np.testing.assert_allclose(out, spec / geo_mean_per_gene, rtol=1e-10)

    def test_normalize_background_axis_default_unchanged(self):
        """Passing axis=-2 explicitly matches omitting it (default)."""
        rng = np.random.default_rng(0)
        spec = rng.uniform(0.5, 2.0, size=(2, 5, 8))
        np.testing.assert_array_equal(
            normalize_background(spec),
            normalize_background(spec, axis=-2),
        )

    def test_comparator_old_method_names_removed(self):
        """The four retired Comparator method names must not be reachable
        as instance attributes (rename-guard for breaking API changes)."""
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        for old in ("test_pattern", "test_expression", "test", "normalize_shape"):
            assert not hasattr(cmp, old), f"Comparator.{old} should have been retired in the rename"

    def test_normalize_covariates_first_arg_named_spectra(self):
        """First-arg rename gene_spectra → spectra. Old keyword should fail."""
        rng = np.random.default_rng(0)
        gene = rng.uniform(size=(20, 8))
        cov = rng.uniform(size=(2, 8))
        # Named-keyword call with the new name works:
        out_kw = normalize_covariates(spectra=gene, covariate_spectra=cov)
        # Positional call still works:
        out_pos = normalize_covariates(gene, cov)
        np.testing.assert_array_equal(out_kw, out_pos)
        # Old keyword name is gone:
        with pytest.raises(TypeError):
            normalize_covariates(gene_spectra=gene, covariate_spectra=cov)
