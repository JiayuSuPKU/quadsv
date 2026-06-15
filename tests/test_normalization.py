"""Tests for comparator spectrum normalization primitives."""

from __future__ import annotations

import numpy as np
import pytest

from quadsv.comparators.normalization import (
    normalize_background,
    normalize_covariates,
    normalize_shape,
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

    def test_normalizers_are_not_reexported_from_multisample(self):
        import quadsv.comparators.multisample as multisample

        for name in ("normalize_background", "normalize_covariates", "normalize_shape"):
            assert name not in multisample.__all__
            assert not hasattr(multisample, name)
