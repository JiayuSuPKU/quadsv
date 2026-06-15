"""Tests for comparator statistical comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from quadsv.comparators.multisample import (
    compare_glm,
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
)
from quadsv.comparators.normalization import normalize_shape


class TestTwoGroupNullCalibration:
    """Permutation-null calibration checks."""

    def test_log_l2_pvalues_are_uniform_under_h0(self):
        rng = np.random.default_rng(42)
        n_samples, n_genes, K = 8, 200, 12
        spectra = rng.uniform(0.5, 5.0, size=(n_samples, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Force the sampling path (max_exact_permutations=0) so the KS test can check
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
            max_exact_permutations=0,
        )
        _, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
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


class TestLogL2AnalyticNull:
    """Analytic Liu mixture null for ``log_l2``."""

    def test_synthetic_h0_pvalues_are_uniform(self):
        rng = np.random.default_rng(0)
        n_a, n_b, K = 4, 4, 30
        spectra = np.exp(rng.standard_normal((n_a + n_b, 5000, K)))
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        _, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic p-values not uniform under H0: KS p={ks_p:.4f}"
        assert 0.45 < df["P_value"].mean() < 0.55

    def test_analytic_breaks_permutation_floor_on_implanted_signal(self):
        """Analytic p-values can beat the exact-permutation floor on strong signals."""
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
        df_analytic = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        for g in ("0", "1", "2", "3", "4"):
            p_perm = df_perm.loc[df_perm["Feature"] == g, "P_value"].iloc[0]
            p_analytic = df_analytic.loc[df_analytic["Feature"] == g, "P_value"].iloc[0]
            assert (
                p_analytic < p_perm
            ), f"gene {g}: analytic={p_analytic:.3g} not < perm={p_perm:.3g}"
            assert p_analytic < 1.0 / 71.0, f"gene {g}: analytic={p_analytic:.3g} above perm floor"

    def test_analytic_argument_is_ignored_for_welch_t_cauchy(self):
        """``welch_t_cauchy`` ignores ``null`` and still returns its schema."""
        rng = np.random.default_rng(3)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic="welch_t_cauchy", null="analytic")
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()

    def test_unknown_null_raises(self):
        rng = np.random.default_rng(4)
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="Unknown null"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="bootstrap")

    def test_small_df_emits_user_warning(self):
        """Analytic null warns when residual df makes variance estimates unstable."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((3, 50, 20)))
        groups = np.array([0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='analytic' at residual df=1"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        spectra = np.exp(rng.standard_normal((4, 50, 20)))
        groups = np.array([0, 0, 1, 1])
        with pytest.warns(UserWarning, match="log_l2 \\+ null='analytic' at residual df=2"):
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        spectra = np.exp(rng.standard_normal((6, 50, 20)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")

    def test_full_sigma_calibrates_under_correlated_bins(self):
        """Full-covariance analytic p-values should stay calibrated with correlated bins."""
        rng = np.random.default_rng(42)
        n_a, n_b, G, K = 4, 4, 2000, 30
        # Shared noise creates near rank-1 covariance across bins.
        shared = rng.standard_normal((n_a + n_b, G, 1)) * 1.0
        per_bin = rng.standard_normal((n_a + n_b, G, K)) * 0.2
        log_y = shared + per_bin
        spectra = np.exp(log_y)
        groups = np.array([0] * n_a + [1] * n_b)
        df = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        ks_p = kstest(df["P_value"].to_numpy(), "uniform")[1]
        fpr05 = (df["P_value"] < 0.05).mean()
        assert (
            fpr05 < 0.10
        ), f"Full-Sigma analytic p-values should be near-calibrated; got Pr(p<.05)={fpr05:.3f}"
        assert (
            ks_p > 1e-3 or fpr05 < 0.07
        ), f"Full-Sigma analytic p-values should be roughly uniform; KS_p={ks_p:.3g} fpr05={fpr05:.3f}"

    def test_masked_analytic_parity_with_unmasked_when_complete(self):
        """All-present masks should match the unmasked analytic path."""
        rng = np.random.default_rng(5)
        n_samples, n_genes, K = 6, 100, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((n_samples, n_genes), dtype=bool)
        df_un = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_m = (
            compare_two_groups_masked(
                spectra, groups, presence, statistic="log_l2", null="analytic"
            )
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            df_un["Statistic"].to_numpy(), df_m["Statistic"].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            df_un["P_value"].to_numpy(), df_m["P_value"].to_numpy(), atol=1e-12
        )

    def test_masked_analytic_calibration_under_synthetic_missingness(self):
        """Random missingness should not strongly distort H0 false positives."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 8, 800, 20
        spectra = np.exp(rng.standard_normal((n_samples, n_genes, K)))
        groups = np.array([0] * 4 + [1] * 4)

        df_full = compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
        fpr_full = (df_full["P_value"] < 0.05).mean()

        presence = rng.uniform(size=(n_samples, n_genes)) >= 0.25
        df_m = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            null="analytic",
            min_samples_per_group=2,
        )
        valid = df_m["P_value"].notna()
        assert valid.sum() > 100, f"only {valid.sum()} testable genes after masking"
        fpr_m = (df_m.loc[valid, "P_value"] < 0.05).mean()

        assert 0.0 < fpr_m < 0.20, f"masked FPR={fpr_m:.3f} far from nominal"
        assert (
            0.5 * fpr_full < fpr_m < 2.0 * fpr_full + 0.02
        ), f"masked vs unmasked FPR mismatch: masked={fpr_m:.3f} unmasked={fpr_full:.3f}"

    def test_masked_analytic_skips_genes_with_insufficient_presence(self):
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
            null="analytic",
            min_samples_per_group=2,
        )
        row = df[df["Feature"] == "0"].iloc[0]
        assert np.isnan(row["P_value"]), f"expected NaN, got {row['P_value']}"
        assert row["n_obs_A"] == 1
        valid_genes = df[df["Feature"] != "0"]
        assert valid_genes["P_value"].notna().all()

    def test_masked_analytic_raises_when_no_eligible_genes(self):
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
                null="analytic",
                min_samples_per_group=2,
            )

    def test_masked_analytic_argument_is_ignored_for_welch_t_cauchy(self):
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
            null="analytic",
        )
        assert "P_value_per_bin" in df.columns
        assert df["P_value"].between(0, 1).all()


class TestCompareGLM:
    """GLM analytic comparisons for DataFrame, dict, and ndarray designs."""

    def test_binary_design_matches_groups_path_byte_close(self):
        """A binary GLM design should match the two-group analytic path."""
        rng = np.random.default_rng(0)
        spectra = np.exp(rng.standard_normal((8, 200, 25)))
        groups = np.array(["WT"] * 4 + ["TG"] * 4)

        df_g = (
            compare_two_groups(spectra, groups, statistic="log_l2", null="analytic")
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        design = pd.DataFrame({"genotype": groups})
        df_d = (
            compare_glm(spectra, design, contrast="genotype")
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
        rng = np.random.default_rng(7)
        n, n_genes, K = 8, 100, 25
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros((n_genes, K))
        beta[:5, :3] = 5.0
        log_y = beta[None, :, :] * x[:, None, None] + 0.3 * rng.standard_normal((n, n_genes, K))
        spectra = np.exp(log_y)
        design = pd.DataFrame({"time": x})
        df = compare_glm(spectra, design, contrast="time")
        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10, f"missing planted: {top10}"

    def test_dict_contrast_normalization(self):
        """A dict contrast should match the equivalent column-name contrast."""
        rng = np.random.default_rng(2)
        spectra = np.exp(rng.standard_normal((8, 30, 20)))
        design = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": np.arange(8)})
        df_dict = (
            compare_glm(spectra, design, contrast={"a": 1.0})
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_str = (
            compare_glm(spectra, design, contrast="a").sort_values("Feature").reset_index(drop=True)
        )
        np.testing.assert_allclose(df_dict["P_value"].to_numpy(), df_str["P_value"].to_numpy())

    def test_ndarray_design_with_intercept_only(self):
        rng = np.random.default_rng(3)
        n, n_genes, K = 6, 20, 18
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X = np.column_stack([np.ones(n), [0, 0, 0, 1, 1, 1]])
        df = compare_glm(spectra, X, contrast=np.array([0.0, 1.0]))
        assert df.shape[0] == n_genes
        assert df["P_value"].between(0, 1).all()

    def test_rank_deficient_estimable_contrast_matches_collapsed_design(self):
        rng = np.random.default_rng(4)
        n, n_genes, K = 8, 20, 12
        x = np.linspace(-1.0, 1.0, n)
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X_full = np.column_stack([np.ones(n), x])
        X_duplicate = np.column_stack([np.ones(n), x, x])

        df_full = (
            compare_glm(spectra, X_full, contrast=np.array([0.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_duplicate = (
            compare_glm(spectra, X_duplicate, contrast=np.array([0.0, 1.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_duplicate["Statistic"], df_full["Statistic"], rtol=1e-12)
        np.testing.assert_allclose(df_duplicate["P_value"], df_full["P_value"], rtol=1e-12)

    def test_rank_deficient_nonestimable_contrast_raises(self):
        rng = np.random.default_rng(5)
        n, n_genes, K = 8, 10, 6
        x = np.linspace(0.0, 1.0, n)
        spectra = np.exp(rng.standard_normal((n, n_genes, K)))
        X = np.column_stack([np.ones(n), x, x])

        with pytest.raises(ValueError, match="not estimable"):
            compare_glm(spectra, X, contrast=np.array([0.0, 1.0, 0.0]))

    def test_glm_functions_do_not_expose_statistic_or_null_arguments(self):
        """GLM primitives are fixed log-L2 analytic tests."""
        import inspect

        for fn in (compare_glm, compare_glm_masked):
            sig = inspect.signature(fn)
            assert "statistic" not in sig.parameters
            assert "null" not in sig.parameters

    def test_masked_glm_matches_unmasked_when_complete(self):
        """A fully present mask should recover the ordinary GLM analytic result."""
        rng = np.random.default_rng(11)
        spectra = np.exp(rng.standard_normal((8, 80, 18)))
        design = pd.DataFrame({"time": np.linspace(0.0, 1.0, 8)})
        presence = np.ones((8, 80), dtype=bool)

        df_full = (
            compare_glm(spectra, design, contrast="time")
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_mask = (
            compare_glm_masked(spectra, design, "time", presence)
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_full["Statistic"], df_mask["Statistic"], atol=1e-12)
        np.testing.assert_allclose(df_full["P_value"], df_mask["P_value"], atol=1e-12)
        assert (df_mask["n_obs"] == 8).all()
        assert (df_mask["df_resid"] == 6).all()

    def test_masked_glm_skips_unestimable_gene_contrast(self):
        """Masking can make a contrast unestimable for a single gene."""
        rng = np.random.default_rng(12)
        spectra = np.exp(rng.standard_normal((6, 12, 10)))
        design = pd.DataFrame({"group": [0, 0, 0, 1, 1, 1]})
        presence = np.ones((6, 12), dtype=bool)
        presence[3:, 0] = False  # gene 0 has no observed samples with group == 1

        df = compare_glm_masked(spectra, design, "group", presence)
        skipped = df.set_index("Feature").loc["0"]

        assert skipped["n_obs"] == 3
        assert skipped["df_resid"] == 2
        assert np.isnan(skipped["Statistic"])
        assert np.isnan(skipped["P_value"])
        assert df.loc[df["Feature"] != "0", "P_value"].notna().all()


class TestTwoGroupSignalDetection:
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
        with pytest.raises(ValueError, match="length n_bins="):
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


class TestScalarTwoGroupComparison:
    """Scalar differential-expression calibration and power."""

    def test_welch_analytic_is_uniform_under_h0(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 1000
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(values, groups)
        _, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"analytic-Welch p-values not uniform under H0, KS p={ks_p:.4f}"

    def test_welch_analytic_matches_scipy(self):
        from scipy.stats import ttest_ind

        rng = np.random.default_rng(2)
        n_per, n_genes = 5, 50
        a = rng.normal(0.0, 1.0, size=(n_per, n_genes))
        b = rng.normal(0.3, 1.5, size=(n_per, n_genes))
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups)
        df = df.set_index("Feature").loc[[str(i) for i in range(n_genes)]]
        scipy_p = ttest_ind(a, b, equal_var=False, axis=0).pvalue
        np.testing.assert_allclose(df["P_value"].to_numpy(), scipy_p, rtol=1e-9, atol=1e-300)

    def test_implanted_mean_shift_recovered(self):
        rng = np.random.default_rng(7)
        n_per, n_genes = 6, 40
        a = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b[:, :5] += 2.0
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups)
        top5 = set(df.head(5).Feature.astype(str).tolist())
        assert top5 == {"0", "1", "2", "3", "4"}

    def test_log_expression_matches_manual_log_values(self):
        rng = np.random.default_rng(8)
        values = rng.lognormal(mean=1.0, sigma=0.2, size=(8, 6))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        df_logged = compare_two_groups_scalar(values, groups, log_expression=True)
        df_manual = compare_two_groups_scalar(np.log(values + 1e-12), groups)

        df_logged = df_logged.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df_logged["Mean_diff"], df_manual["Mean_diff"], rtol=1e-12)
        np.testing.assert_allclose(df_logged["P_value"], df_manual["P_value"], rtol=1e-12)

    def test_log_expression_rejects_nonpositive_values(self):
        values = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 2.5]])
        groups = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="log_expression=True"):
            compare_two_groups_scalar(values, groups, log_expression=True)


class TestCompareGLMScalar:
    """Scalar GLM tests for continuous designs and logged responses."""

    def test_scalar_functions_do_not_expose_null_argument(self):
        """Scalar DE uses fixed t-distribution nulls, not a null selector."""
        import inspect

        assert "null" not in inspect.signature(compare_two_groups_scalar).parameters
        assert "null" not in inspect.signature(compare_glm_scalar).parameters

    def test_matches_manual_ols_contrast(self):
        from scipy.stats import t as t_dist

        rng = np.random.default_rng(11)
        n, n_genes = 9, 6
        x = np.linspace(-1.0, 1.0, n)
        values = rng.normal(size=(n, n_genes)) + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        design = pd.DataFrame({"x": x})

        df = compare_glm_scalar(values, design, contrast="x").set_index("Feature")

        X = np.column_stack([np.ones(n), x])
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ values
        resid = values - X @ beta
        df_resid = n - np.linalg.matrix_rank(X)
        sigma2 = np.sum(resid**2, axis=0) / df_resid
        se = np.sqrt(sigma2 * xtx_inv[1, 1])
        t_stat = beta[1] / se
        pvals = 2.0 * t_dist.sf(np.abs(t_stat), df_resid)

        ordered = df.loc[[str(i) for i in range(n_genes)]]
        np.testing.assert_allclose(ordered["Estimate"].to_numpy(), beta[1], rtol=1e-12)
        np.testing.assert_allclose(ordered["P_value"].to_numpy(), pvals, rtol=1e-12)

    def test_continuous_contrast_recovers_planted_scalar_signal(self):
        rng = np.random.default_rng(12)
        n, n_genes = 12, 50
        x = np.linspace(0.0, 1.0, n)
        beta = np.zeros(n_genes)
        beta[:5] = 2.0
        values = 0.2 * rng.standard_normal((n, n_genes)) + x[:, None] * beta[None, :]

        df = compare_glm_scalar(values, pd.DataFrame({"time": x}), contrast="time")

        top10 = set(df.head(10)["Feature"].tolist())
        assert {"0", "1", "2", "3", "4"} <= top10

    def test_log_expression_matches_manual_log_response(self):
        rng = np.random.default_rng(13)
        n, n_genes = 10, 8
        x = np.linspace(0.0, 1.0, n)
        log_values = 1.0 + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        values = np.exp(log_values + 0.05 * rng.standard_normal((n, n_genes)))
        design = pd.DataFrame({"x": x})

        df_logged = compare_glm_scalar(values, design, "x", log_expression=True)
        df_manual = compare_glm_scalar(np.log(values + 1e-12), design, "x")

        df_logged = df_logged.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df_logged["Estimate"], df_manual["Estimate"], rtol=1e-12)
        np.testing.assert_allclose(df_logged["P_value"], df_manual["P_value"], rtol=1e-12)

    def test_log_expression_rejects_nonpositive_values(self):
        values = np.array([[1.0, 2.0], [0.5, -1.0], [1.5, 2.5]])
        design = np.column_stack([np.ones(3), np.arange(3)])
        with pytest.raises(ValueError, match="log_expression=True"):
            compare_glm_scalar(values, design, np.array([0.0, 1.0]), log_expression=True)

    def test_rank_deficient_estimable_contrast_matches_collapsed_design(self):
        rng = np.random.default_rng(14)
        n, n_genes = 9, 7
        x = np.linspace(-1.0, 1.0, n)
        values = rng.normal(size=(n, n_genes)) + x[:, None] * np.linspace(0.0, 1.0, n_genes)
        X_full = np.column_stack([np.ones(n), x])
        X_duplicate = np.column_stack([np.ones(n), x, x])

        df_full = (
            compare_glm_scalar(values, X_full, contrast=np.array([0.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )
        df_duplicate = (
            compare_glm_scalar(values, X_duplicate, contrast=np.array([0.0, 1.0, 1.0]))
            .sort_values("Feature")
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(df_duplicate["Estimate"], df_full["Estimate"], rtol=1e-12)
        np.testing.assert_allclose(df_duplicate["P_value"], df_full["P_value"], rtol=1e-12)

    def test_rank_deficient_nonestimable_contrast_raises(self):
        values = np.arange(24, dtype=float).reshape(8, 3)
        x = np.linspace(0.0, 1.0, values.shape[0])
        X = np.column_stack([np.ones(values.shape[0]), x, x])

        with pytest.raises(ValueError, match="not estimable"):
            compare_glm_scalar(values, X, contrast=np.array([0.0, 1.0, 0.0]))


class TestMaskedTwoGroupComparison:
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
            ("log_l2", "analytic"),
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
            ("log_l2", "analytic"),
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
        df_kw = compare_glm(spec, design, "x", normalize_shape=True)
        df_manual = compare_glm(normalize_shape(spec), design, "x")
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_masked_glm_option_matches_manual_normalization(self):
        spec, _ = _stub_spectra_groups()
        design = pd.DataFrame({"x": np.arange(spec.shape[0], dtype=float)})
        presence = np.ones((spec.shape[0], spec.shape[1]), dtype=bool)
        df_kw = compare_glm_masked(spec, design, "x", presence, normalize_shape=True)
        df_manual = compare_glm_masked(normalize_shape(spec), design, "x", presence)
        df_kw = df_kw.sort_values("Feature").reset_index(drop=True)
        df_manual = df_manual.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(
            df_kw["P_value"].to_numpy(),
            df_manual["P_value"].to_numpy(),
            rtol=1e-12,
            atol=1e-15,
        )
