"""Tests for the public comparator classes."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pytest

# Suppress noisy ome_zarr / dask deprecation chatter during import.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import shapely.geometry as sg
import spatialdata as sd
from geopandas import GeoDataFrame
from spatialdata.models import ShapesModel, TableModel

from quadsv.comparators import ComparatorGrid, ComparatorIrregular
from quadsv.comparators.multisample import (
    compare_glm_masked,
    compare_glm_scalar,
    compare_two_groups,
    compare_two_groups_scalar,
)

# Standalone comparison helpers are the oracles for wrapper-level dispatch tests.


class _ArraySelection:
    """Small object that mimics xarray's ``.sel(...).values`` result."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _ArrayRaster:
    """Minimal xarray-like raster used to test ComparatorGrid directly."""

    def __init__(self, values: np.ndarray, gene_names: list[str]) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self.shape = self._values.shape
        self._name_to_idx = {g: i for i, g in enumerate(gene_names)}

    def sel(self, *, c):
        names = [c] if isinstance(c, str) else list(c)
        return _ArraySelection(self._values[[self._name_to_idx[g] for g in names]])


def _make_table_only_sdata(gene_names: list[str]) -> sd.SpatialData:
    """Create the table metadata ComparatorGrid needs when rasterization is stubbed."""

    table = ad.AnnData(X=np.zeros((1, len(gene_names)), dtype=np.float64))
    table.var_names = gene_names
    return sd.SpatialData(tables={"table": table})


def _install_grid_rasters(monkeypatch, rasters: list[np.ndarray], gene_names: list[str]):
    """Patch ComparatorGrid rasterization to return supplied in-memory arrays."""

    samples = [_make_table_only_sdata(gene_names) for _ in rasters]
    by_sample_id = {id(sample): raster for sample, raster in zip(samples, rasters, strict=True)}

    def fake_rasterize_one(self, sdata):
        return _ArrayRaster(by_sample_id[id(sdata)], self.gene_names)

    monkeypatch.setattr(ComparatorGrid, "_rasterize_one", fake_rasterize_one)
    return samples


def _grid_sample_to_adata(sample: np.ndarray, gene_names, spacing=(1.0, 1.0)):
    """Convert a regular ``(gene, y, x)`` grid into AnnData with spatial coords."""

    n_genes, ny, nx = sample.shape
    X = sample.reshape(n_genes, ny * nx).T
    # Coordinates follow the same row/column ordering as the flattened expression matrix.
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


def _grid_samples_to_adata(samples, gene_names, spacings=None):
    """Convert multiple regular grids into AnnData samples."""

    spacings = spacings if spacings is not None else [(1.0, 1.0)] * len(samples)
    return [
        _grid_sample_to_adata(s, gene_names, spacing=spacings[i]) for i, s in enumerate(samples)
    ]


def _make_bin_sdata(
    ny: int = 16,
    nx: int = 16,
    gene_values: dict[str, np.ndarray] | None = None,
    rng_seed: int = 0,
) -> sd.SpatialData:
    """Build a minimal SpatialData with square-bin Shapes + an AnnData table.

    The bins are a regular ``ny × nx`` square grid with unit pitch;
    ``col_key`` / ``row_key`` / ``value_key`` are populated on the table's
    ``.obs``. ``gene_values`` maps gene name → ``(ny, nx)`` pattern; when None
    a pair of low-frequency sinusoids is used so :class:`ComparatorGrid`
    has non-trivial spectra to compute.
    """
    rng = np.random.default_rng(rng_seed)
    if gene_values is None:
        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        gene_values = {
            "g0": np.sin(2 * np.pi * y / ny),
            "g1": np.cos(2 * np.pi * x / nx),
            "g2": rng.standard_normal((ny, nx)),
        }
    gene_names = list(gene_values)
    n_genes = len(gene_names)
    n_bins = ny * nx

    # Per-bin square polygons (unit pitch, origin at 0, 0).
    polys, row_idx, col_idx = [], [], []
    for r in range(ny):
        for c in range(nx):
            polys.append(sg.box(c, r, c + 1, r + 1))
            row_idx.append(r)
            col_idx.append(c)
    shapes = ShapesModel.parse(
        GeoDataFrame(
            {"geometry": polys},
            index=[f"bin_{i}" for i in range(n_bins)],
        )
    )

    # Per-bin expression table.
    X = np.zeros((n_bins, n_genes), dtype=np.float64)
    for gi, g in enumerate(gene_names):
        X[:, gi] = gene_values[g].reshape(-1)
    obs = pd.DataFrame(
        {
            "row_idx": row_idx,
            "col_idx": col_idx,
            "region": pd.Categorical(["bins"] * n_bins),
            "instance_id": [f"bin_{i}" for i in range(n_bins)],
        },
        index=[f"bin_{i}" for i in range(n_bins)],
    )
    table = ad.AnnData(X=X, obs=obs)
    table.var_names = gene_names
    table = TableModel.parse(
        table,
        region="bins",
        region_key="region",
        instance_key="instance_id",
    )

    sdata = sd.SpatialData(shapes={"bins": shapes}, tables={"table": table})
    return sdata


class TestComparatorGrid:
    """Coverage for the SpatialData-backed comparator surface."""

    def _build_samples(self, n_samples: int = 4, ny: int = 16, nx: int = 16):
        """Build small SpatialData samples with one group-dependent gene."""

        samples = []
        for i in range(n_samples):
            y_stripes = np.broadcast_to(
                np.sin(2 * np.pi * np.arange(ny)[:, None] / ny),
                (ny, nx),
            )
            x_stripes = np.broadcast_to(
                np.cos(2 * np.pi * np.arange(nx)[None, :] / nx),
                (ny, nx),
            )
            gene_values = {
                "g0": y_stripes,
                "g1": x_stripes,
                "g2": (
                    np.broadcast_to(
                        np.sin(2 * np.pi * np.arange(ny)[:, None] / ny + 0.3 * i),
                        (ny, nx),
                    )
                    if i < n_samples // 2
                    else np.broadcast_to(
                        np.cos(2 * np.pi * np.arange(nx)[None, :] / nx + 0.1 * i),
                        (ny, nx),
                    )
                ),
            }
            samples.append(_make_bin_sdata(ny, nx, gene_values, rng_seed=i))
        return samples

    def test_fit_populates_core_attributes(self):
        try:
            samples = self._build_samples()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"SpatialData fixture unavailable: {exc}")

        groups = np.array([0, 0, 1, 1])
        try:
            cmp = ComparatorGrid(
                samples,
                bins="bins",
                table_name="table",
                col_key="col_idx",
                row_key="row_idx",
                n_radial_bins=6,
                fft_chunk_size=2,
            )
        except TypeError:
            pytest.skip("ComparatorGrid smoke test not runnable on this spatialdata version.")
        cmp.compute_spectra(progress=False)
        assert cmp.spectra_ is not None
        assert cmp.spectra_.shape[0] == len(samples)
        assert cmp.spectra_.shape[1] == 3  # n_genes
        assert cmp.dc_.shape == (len(samples), 3)
        assert cmp.presence_.shape == (len(samples), 3)
        # Pattern test runs without error and returns a frame with a row per gene.
        df = cmp.test_diff_freq(groups, statistic="welch_t_cauchy")
        assert df.shape[0] == 3
        assert df["P_value"].between(0, 1).all()


class TestComparatorGridFeatures:
    """ComparatorGrid feature extraction options that do not need real rasterization."""

    def test_radial_auto_chunk_with_per_sample_spacing(self, monkeypatch):
        rng = np.random.default_rng(10)
        gene_names = ["g0", "g1", "g2"]
        rasters = [
            rng.standard_normal((3, 8, 10)),
            rng.standard_normal((3, 10, 8)),
        ]
        samples = _install_grid_rasters(monkeypatch, rasters, gene_names)
        spacings = [(0.5, 0.25), (1.0, 0.75)]

        cmp = ComparatorGrid(
            samples,
            bins="bins",
            table_name="table",
            col_key="col_idx",
            row_key="row_idx",
            gene_names=gene_names,
            feature_mode="radial",
            n_radial_bins=5,
            spacing=spacings,
            fft_chunk_size="auto",
        ).compute_spectra(n_jobs=1, progress=False)

        assert cmp._grid_shapes == [(8, 10), (10, 8)]
        assert cmp._spacings == spacings
        assert 8 <= cmp._fft_chunk_size <= cmp._auto_chunk_cap
        assert cmp.spectra_.shape == (2, 3, 5)
        expected_f_max = min(1.0 / (2.0 * max(dy, dx)) for dy, dx in spacings)
        assert cmp.freq_edges[-1] == pytest.approx(expected_f_max * (1.0 + 1e-9))

    def test_2d_auto_chunk_with_per_sample_spacing(self, monkeypatch):
        rng = np.random.default_rng(11)
        gene_names = ["g0", "g1", "g2"]
        rasters = [
            rng.standard_normal((3, 9, 11)),
            rng.standard_normal((3, 11, 9)),
        ]
        samples = _install_grid_rasters(monkeypatch, rasters, gene_names)
        spacings = [(0.4, 0.5), (0.8, 0.6)]

        cmp = ComparatorGrid(
            samples,
            bins="bins",
            table_name="table",
            col_key="col_idx",
            row_key="row_idx",
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=4,
            n_theta_bins=8,
            spacing=spacings,
            fft_chunk_size="auto",
        ).compute_spectra(n_jobs=1, landmark_genes=["g0"], progress=False)

        assert cmp._grid_shapes == [(9, 11), (11, 9)]
        assert cmp._spacings == spacings
        assert cmp._n_theta_bins == 8
        assert 8 <= cmp._fft_chunk_size <= cmp._auto_chunk_cap
        assert cmp.rotation_angles_.shape == (2,)
        assert cmp.rotation_angles_[0] == pytest.approx(0.0)
        assert cmp._raw_2d_spectra is None
        assert cmp.spectra_.shape == (2, 3, 4 * 8)
        assert np.isfinite(cmp.spectra_).all()

    def test_radial_and_2d_use_sample_parallel_dispatch(self, monkeypatch):
        import quadsv.comparators.base as base_mod

        calls = []

        class FakeParallel:
            def __init__(self, *, n_jobs, prefer):
                self.n_jobs = n_jobs
                self.prefer = prefer
                calls.append(self)

            def __call__(self, tasks):
                return [func(*args, **kwargs) for func, args, kwargs in tasks]

        monkeypatch.setattr(base_mod, "Parallel", FakeParallel)

        rng = np.random.default_rng(12)
        gene_names = ["g0", "g1"]
        rasters = [rng.standard_normal((2, 8, 8)) for _ in range(3)]
        samples = _install_grid_rasters(monkeypatch, rasters, gene_names)

        ComparatorGrid(
            samples,
            bins="bins",
            table_name="table",
            col_key="col_idx",
            row_key="row_idx",
            gene_names=gene_names,
            feature_mode="radial",
            n_radial_bins=3,
            fft_chunk_size=1,
        ).compute_spectra(n_jobs=2, progress=False)
        assert [c.n_jobs for c in calls] == [2]
        assert calls[0].prefer == "threads"

        calls.clear()
        cmp = ComparatorGrid(
            samples,
            bins="bins",
            table_name="table",
            col_key="col_idx",
            row_key="row_idx",
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=3,
            n_theta_bins=4,
            fft_chunk_size=1,
        ).compute_spectra(n_jobs=2, landmark_genes=["g0"], progress=False)
        assert [c.n_jobs for c in calls] == [2, 2]
        assert all(c.prefer == "threads" for c in calls)
        assert cmp.spectra_.shape == (3, 2, 3 * 4)

    def test_subset_returns_fitted_comparator_view(self, monkeypatch):
        rng = np.random.default_rng(13)
        gene_names = ["g0", "g1"]
        rasters = [rng.standard_normal((2, 8, 8)) for _ in range(3)]
        samples = _install_grid_rasters(monkeypatch, rasters, gene_names)
        spacings = [(1.0, 1.0), (0.8, 1.0), (1.0, 0.8)]
        cmp = ComparatorGrid(
            samples,
            bins="bins",
            table_name="table",
            col_key="col_idx",
            row_key="row_idx",
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=3,
            n_theta_bins=4,
            spacing=spacings,
            fft_chunk_size=1,
        ).compute_spectra(n_jobs=1, landmark_genes=["g0"], progress=False)
        cmp._raw_2d_spectra = [np.asarray([i]) for i in range(len(cmp.samples))]

        sub = cmp.subset([2, 0])

        assert isinstance(sub, ComparatorGrid)
        assert sub is not cmp
        assert [id(sample) for sample in sub.samples] == [
            id(cmp.samples[2]),
            id(cmp.samples[0]),
        ]
        np.testing.assert_array_equal(sub.spectra_, cmp.spectra_[[2, 0]])
        np.testing.assert_array_equal(sub.dc_, cmp.dc_[[2, 0]])
        np.testing.assert_array_equal(sub.presence_, cmp.presence_[[2, 0]])
        np.testing.assert_array_equal(sub.rotation_angles_, cmp.rotation_angles_[[2, 0]])
        assert sub._grid_shapes == [(8, 8), (8, 8)]
        assert sub._spacings == [spacings[2], spacings[0]]
        assert [int(x[0]) for x in sub._raw_2d_spectra] == [2, 0]

        mask_sub = cmp.subset(np.array([False, True, True]))
        np.testing.assert_array_equal(mask_sub.spectra_, cmp.spectra_[[1, 2]])


class TestComparatorIrregularImport:
    """Constructor validation for the AnnData-backed comparator."""

    def test_rejects_non_anndata(self):
        with pytest.raises(TypeError, match="anndata.AnnData"):
            ComparatorIrregular([np.zeros((4, 3))])


class TestComparatorIrregularFeatures:
    """ComparatorIrregular grid/spacing/chunk behavior for radial and 2D features."""

    def test_radial_auto_chunk_with_per_sample_grid_spacing(self):
        rng = np.random.default_rng(1)
        gene_names = ["g0", "g1", "g2"]
        samples = [rng.standard_normal((3, 8, 8)) for _ in range(2)]
        adatas = _grid_samples_to_adata(samples, gene_names)
        grid_shapes = [(12, 14), (10, 16)]
        spacings = [(0.5, 0.25), (1.0, 0.75)]

        cmp = ComparatorIrregular(
            adatas,
            gene_names=gene_names,
            feature_mode="radial",
            grid_shape=grid_shapes,
            spacing=spacings,
            n_radial_bins=6,
            nufft_chunk_size="auto",
        )
        assert cmp._grid_shapes == grid_shapes
        assert cmp._spacings == spacings
        assert 8 <= cmp._nufft_chunk_size <= cmp._auto_chunk_cap

        cmp.compute_spectra(n_jobs=1, progress=False)
        assert cmp.spectra_.shape == (2, 3, 6)
        expected_f_max = min(1.0 / (2.0 * max(dy, dx)) for dy, dx in spacings)
        assert cmp.freq_edges[-1] == pytest.approx(expected_f_max * (1.0 + 1e-9))

    def test_2d_auto_chunk_with_per_sample_grid_spacing(self):
        rng = np.random.default_rng(3)
        gene_names = ["g0", "g1", "g2"]
        samples = [rng.standard_normal((3, 10, 10)) for _ in range(3)]
        adatas = _grid_samples_to_adata(samples, gene_names)
        grid_shapes = [(10, 10), (12, 10), (10, 12)]
        spacings = [(1.0, 1.0), (0.75, 1.0), (1.0, 0.75)]

        cmp = ComparatorIrregular(
            adatas,
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=4,
            n_theta_bins=6,
            grid_shape=grid_shapes,
            spacing=spacings,
            nufft_chunk_size="auto",
        ).compute_spectra(n_jobs=1, landmark_genes=["g0"], progress=False)

        assert cmp._grid_shapes == grid_shapes
        assert cmp._spacings == spacings
        assert cmp._n_theta_bins == 6
        assert 8 <= cmp._nufft_chunk_size <= cmp._auto_chunk_cap
        assert cmp._raw_2d_spectra is None
        assert cmp.rotation_angles_.shape == (3,)
        assert cmp.rotation_angles_[0] == pytest.approx(0.0)
        assert cmp.spectra_.shape == (3, 3, 4 * 6)
        assert np.isfinite(cmp.spectra_).all()

        covariates = [rng.standard_normal((1, 10, 10)) for _ in range(3)]
        before_shape = cmp.spectra_.shape
        ret = cmp.normalize_covariates(covariates)
        assert ret is cmp
        assert cmp.spectra_.shape == before_shape
        assert np.isfinite(cmp.spectra_).all()

    def test_2d_uses_sample_parallel_dispatch(self, monkeypatch):
        import quadsv.comparators.base as base_mod

        calls = []

        class FakeParallel:
            def __init__(self, *, n_jobs, prefer):
                self.n_jobs = n_jobs
                self.prefer = prefer
                calls.append(self)

            def __call__(self, tasks):
                return [func(*args, **kwargs) for func, args, kwargs in tasks]

        monkeypatch.setattr(base_mod, "Parallel", FakeParallel)

        rng = np.random.default_rng(9)
        gene_names = ["g0", "g1"]
        samples = [rng.standard_normal((2, 8, 8)) for _ in range(3)]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names),
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=3,
            n_theta_bins=4,
            grid_shape=(8, 8),
            spacing=(1.0, 1.0),
            nufft_chunk_size=1,
        ).compute_spectra(n_jobs=2, landmark_genes=["g0"], progress=False)

        assert [c.n_jobs for c in calls] == [2, 2]
        assert all(c.prefer == "threads" for c in calls)
        assert cmp.spectra_.shape == (3, 2, 3 * 4)

    def test_rejects_non_positive_theta_bins(self):
        rng = np.random.default_rng(2)
        gene_names = ["g0", "g1"]
        samples = [rng.standard_normal((2, 6, 6)) for _ in range(2)]
        adatas = _grid_samples_to_adata(samples, gene_names)
        with pytest.raises(ValueError, match="n_theta_bins"):
            ComparatorIrregular(
                adatas,
                gene_names=gene_names,
                feature_mode="2d",
                n_theta_bins=0,
            )

    def test_rejects_mismatched_grid_shape_count(self):
        rng = np.random.default_rng(2)
        gene_names = ["g0", "g1"]
        samples = [rng.standard_normal((2, 6, 6)) for _ in range(2)]
        adatas = _grid_samples_to_adata(samples, gene_names)
        with pytest.raises(ValueError, match="per-sample grid_shape has 3 rows"):
            ComparatorIrregular(
                adatas,
                gene_names=gene_names,
                grid_shape=[(6, 6), (6, 6), (6, 6)],
                spacing=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            )

    def test_2d_rejects_unknown_landmark_gene(self):
        rng = np.random.default_rng(4)
        gene_names = ["g0", "g1"]
        samples = [rng.standard_normal((2, 8, 8)) for _ in range(2)]
        adatas = _grid_samples_to_adata(samples, gene_names)
        cmp = ComparatorIrregular(
            adatas,
            gene_names=gene_names,
            feature_mode="2d",
            n_radial_bins=3,
            grid_shape=(8, 8),
            spacing=(1.0, 1.0),
            nufft_chunk_size=1,
        )
        with pytest.raises(KeyError, match="landmark_genes"):
            cmp.compute_spectra(n_jobs=1, landmark_genes=["missing"], progress=False)


class TestComparatorIrregularDesign:
    """Design validation owned by the comparator wrapper."""

    def test_rejects_non_binary_1d_design(self):
        rng = np.random.default_rng(0)
        samples = [
            _grid_sample_to_adata(
                rng.uniform(size=(5, 8, 8)),
                gene_names=[f"g{i}" for i in range(5)],
            )
            for _ in range(4)
        ]
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False)
        with pytest.raises(ValueError, match="exactly two distinct labels"):
            cmp.test_diff_freq(np.array([0, 1, 2, 2]))


class TestComparatorIrregularEffectiveRank:
    """Comparator-level accessors for effective-rank summaries."""

    def test_effective_rank_within_group(self):
        rng = np.random.default_rng(4)
        n_per = 4
        adatas = [
            _grid_sample_to_adata(
                rng.uniform(size=(200, 8, 8)),
                gene_names=[f"g{j}" for j in range(200)],
            )
            for _ in range(2 * n_per)
        ]
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

    def test_effective_rank_per_sample(self):
        rng = np.random.default_rng(5)
        n_per = 3
        n_total = 2 * n_per
        adatas = [
            _grid_sample_to_adata(
                rng.uniform(size=(150, 8, 8)),
                gene_names=[f"g{j}" for j in range(150)],
            )
            for _ in range(n_total)
        ]
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


class TestComparatorIrregularWorkflow:
    """End-to-end ComparatorIrregular workflows through public methods."""

    def test_radial_workflow_ranks_implanted_gene_first(self):
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
            ComparatorIrregular(_grid_samples_to_adata(samples, gene_names), gene_names)
            .compute_spectra(progress=False)
            .normalize_background()
        )
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=300, random_state=0)
        assert df["Feature"].iloc[0] == "g0", f"expected g0 first, got {df.head().to_dict()}"

    def test_covariate_residualization_keeps_test_path_usable(self):
        rng = np.random.default_rng(0)
        n_per = 3
        ny = nx = 16
        gene_names = [f"g{i}" for i in range(4)]
        samples = [rng.standard_normal((4, ny, nx)) for _ in range(2 * n_per)]
        covariates = [rng.standard_normal((1, ny, nx)) for _ in range(2 * n_per)]
        groups = np.array([0] * n_per + [1] * n_per)
        cmp = ComparatorIrregular(_grid_samples_to_adata(samples, gene_names), gene_names)
        cmp.compute_spectra(progress=False).normalize_covariates(covariates)
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=50, random_state=0)
        assert df.shape[0] == 4

    def test_diff_freq_requires_spectra(self):
        gene_names = ["a", "b"]
        adatas = _grid_samples_to_adata([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        with pytest.raises(RuntimeError, match=r"\.compute_spectra\(\)"):
            cmp.test_diff_freq(np.array([0, 1]))

    def test_diff_freq_glm_rejects_binary_only_selectors(self):
        gene_names = ["a", "b"]
        samples = [np.ones((2, 4, 4)) * (i + 1.0) for i in range(4)]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names), gene_names
        ).compute_spectra(progress=False)
        design = pd.DataFrame({"x": np.arange(4, dtype=float)})

        with pytest.raises(ValueError, match="statistic='log_l2'"):
            cmp.test_diff_freq(design, contrast="x", statistic="welch_t_cauchy")
        with pytest.raises(NotImplementedError, match="null='analytic'"):
            cmp.test_diff_freq(design, contrast="x", null="permutation")


class TestComparatorIrregularNormalizeCovariates:
    """Comparator-level covariate normalization input modes."""

    @staticmethod
    def _build_samples(n_samples=4, n_spots=200, n_genes=4, seed=0):
        """Build AnnData samples with numeric obs, categorical obs, and genes."""

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

    def test_obs_keys_residualize_spectra(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False).normalize_background()
        before = cmp.spectra_.copy()
        ret = cmp.normalize_covariates(["cov_a", "cov_b"])
        assert ret is cmp
        assert not np.array_equal(cmp.spectra_, before)
        assert cmp.spectra_.shape == before.shape

    def test_missing_covariate_key_raises(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False)
        with pytest.raises(KeyError, match="in neither obs.columns nor"):
            cmp.normalize_covariates(["does_not_exist"])

    def test_obs_key_takes_precedence_over_var_name(self):
        samples = self._build_samples()
        for a in samples:
            a.obs["g0"] = np.linspace(0.0, 1.0, a.n_obs)
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False).normalize_background()
        before = cmp.spectra_.copy()
        cmp.normalize_covariates(["g0"])
        assert not np.array_equal(cmp.spectra_, before)

    def test_mixed_obs_and_var_name_keys_residualize_spectra(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False).normalize_background()
        before = cmp.spectra_.copy()
        cmp.normalize_covariates(["cov_a", "g0"])
        assert not np.array_equal(cmp.spectra_, before)

    def test_non_numeric_obs_key_raises(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False)
        with pytest.raises(ValueError, match="cannot be cast to float"):
            cmp.normalize_covariates(["batch"])

    def test_array_covariates_still_work(self):
        rng = np.random.default_rng(1)
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False).normalize_background()
        before = cmp.spectra_.copy()
        arrays = [rng.standard_normal((1, 8, 8)) for _ in samples]
        cmp.normalize_covariates(arrays)
        assert not np.array_equal(cmp.spectra_, before)

    def test_rejects_empty_covariate_sequence(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False)
        with pytest.raises(ValueError, match="non-empty"):
            cmp.normalize_covariates([])

    def test_rejects_mixed_key_and_array_covariates(self):
        samples = self._build_samples()
        cmp = ComparatorIrregular(samples).compute_spectra(progress=False)
        with pytest.raises(TypeError, match="Mixed str and non-str"):
            cmp.normalize_covariates(["cov_a", np.zeros((1, 8, 8))])


class TestComparatorIrregularDcAccess:
    """Comparator-level access to DC means and expression testing."""

    def test_compute_spectra_populates_dc_means(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((3, 8, 10)) + s for s in range(4)]
        gene_names = ["a", "b", "c"]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names), gene_names
        ).compute_spectra(progress=False)
        assert cmp.dc_ is not None
        assert cmp.dc_.shape == (4, 3)
        expected = np.array([samples[i].mean(axis=(1, 2)) for i in range(4)])
        np.testing.assert_allclose(cmp.dc_, expected, rtol=1e-12)

    def test_diff_expr_requires_spectra(self):
        gene_names = ["a", "b"]
        adatas = _grid_samples_to_adata([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        with pytest.raises(RuntimeError, match=r"\.compute_spectra\(\)"):
            cmp.test_diff_expr(np.array([0, 1]))

    def test_diff_expr_supports_glm_scalar_design(self):
        rng = np.random.default_rng(4)
        x = np.linspace(0.0, 1.0, 8)
        gene_names = ["g0", "g1", "g2"]
        samples = []
        for xi in x:
            sample = 0.05 * rng.standard_normal((3, 6, 6))
            sample[0] += 2.0 * xi
            samples.append(sample)
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names),
            gene_names=gene_names,
        ).compute_spectra(progress=False)

        df = cmp.test_diff_expr(pd.DataFrame({"dose": x}), contrast="dose")

        assert df.iloc[0]["Feature"] == "g0"
        assert df.iloc[0]["Estimate"] > 1.0
        assert df.iloc[0]["P_value"] < 0.01

    def test_diff_expr_log_expression_applies_to_two_group_path(self):
        gene_names = ["g0", "g1"]
        groups = np.array([0, 0, 1, 1])
        dc_values = np.array(
            [
                [1.0, 2.0],
                [1.2, 1.9],
                [2.4, 2.1],
                [2.6, 2.0],
            ]
        )
        samples = [np.array([np.full((6, 6), value) for value in row]) for row in dc_values]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names),
            gene_names=gene_names,
        ).compute_spectra(progress=False)

        df = cmp.test_diff_expr(groups, log_expression=True)
        expected = compare_two_groups_scalar(
            np.log(cmp.dc_ + 1e-12),
            groups,
            gene_names=gene_names,
        )

        df = df.sort_values("Feature").reset_index(drop=True)
        expected = expected.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df["Mean_diff"], expected["Mean_diff"], rtol=1e-12)
        np.testing.assert_allclose(df["P_value"], expected["P_value"], rtol=1e-12)

    def test_diff_expr_log_expression_applies_to_glm_path(self):
        gene_names = ["g0", "g1"]
        x = np.linspace(0.0, 1.0, 6)
        dc_values = np.column_stack([np.exp(1.0 + 0.8 * x), np.exp(1.5 + 0.05 * x)])
        samples = [np.array([np.full((6, 6), value) for value in row]) for row in dc_values]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names),
            gene_names=gene_names,
        ).compute_spectra(progress=False)
        design = pd.DataFrame({"dose": x})

        df = cmp.test_diff_expr(design, contrast="dose", log_expression=True)
        expected = compare_glm_scalar(
            np.log(cmp.dc_ + 1e-12),
            design,
            "dose",
            gene_names=gene_names,
        )

        df = df.sort_values("Feature").reset_index(drop=True)
        expected = expected.sort_values("Feature").reset_index(drop=True)
        np.testing.assert_allclose(df["Estimate"], expected["Estimate"], rtol=1e-12)
        np.testing.assert_allclose(df["P_value"], expected["P_value"], rtol=1e-12)


class TestComparatorIrregularDcPatternSeparation:
    """Mean shifts should affect expression tests without driving pattern tests."""

    def test_mean_shift_affects_expression_not_pattern(self):
        rng = np.random.default_rng(0)
        n_per = 5
        ny = nx = 24
        n_genes = 6
        pattern = rng.standard_normal((n_genes, ny, nx))

        samples = [
            pattern + 0.05 * rng.standard_normal((n_genes, ny, nx)) for _ in range(2 * n_per)
        ]
        for i in range(n_per, 2 * n_per):
            samples[i][0] += 10.0

        groups = np.array([0] * n_per + [1] * n_per)
        gene_names = [f"g{i}" for i in range(n_genes)]
        cmp = ComparatorIrregular(
            _grid_samples_to_adata(samples, gene_names),
            gene_names=gene_names,
            n_radial_bins=8,
        ).compute_spectra(progress=False)

        de = cmp.test_diff_expr(groups)
        pattern_df = cmp.test_diff_freq(groups, n_perm=400, random_state=0)

        de_g0 = de.set_index("Feature").loc["g0"]
        pat_g0 = pattern_df.set_index("Feature").loc["g0"]
        assert de.Feature.iloc[0] == "g0"
        assert de_g0.P_value < 0.05
        assert pat_g0.P_value > 0.05


class TestComparatorIrregularNormalizeShape:
    """Comparator wrapper behavior for ``normalize_shape=True``."""

    def test_diff_freq_normalize_shape_matches_helper_and_preserves_spectra(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((4, 12, 14)) for _ in range(6)]
        groups = np.array([0, 0, 0, 1, 1, 1])
        gene_names = ["g0", "g1", "g2", "g3"]
        cmp = (
            ComparatorIrregular(
                _grid_samples_to_adata(samples, gene_names), gene_names, n_radial_bins=8
            )
            .compute_spectra(progress=False)
            .normalize_background()
        )
        before = cmp.spectra_.copy()
        df_kw = cmp.test_diff_freq(
            groups, statistic="log_l2", null="analytic", normalize_shape=True
        )
        # The lower-level helper is the oracle for wrapper dispatch correctness.
        df_manual = compare_two_groups(
            cmp.spectra_,
            groups,
            gene_names=cmp.gene_names,
            statistic="log_l2",
            null="analytic",
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
        np.testing.assert_array_equal(cmp.spectra_, before)


class TestComparatorIrregularIncompleteData:
    """Presence masks should route comparator tests through masked statistics."""

    def test_presence_threshold_uses_masked_comparison(self):
        rng = np.random.default_rng(0)
        ny = nx = 12
        gene_names = ["g0", "g1", "g2"]
        samples = []
        coords = (
            np.stack(
                np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"),
                axis=-1,
            )
            .reshape(-1, 2)
            .astype(float)
        )
        for _ in range(6):
            X = rng.uniform(0.1, 1.0, size=(ny * nx, 3))
            a = ad.AnnData(X=X)
            a.var_names = gene_names
            a.obsm["spatial"] = coords
            samples.append(a)
        for i in (0, 1):
            samples[i].X[:, 0] = 0.0
        groups = np.array([0, 0, 0, 1, 1, 1])
        cmp = ComparatorIrregular(
            samples,
            gene_names,
            presence_threshold=0.5,
        ).compute_spectra(progress=False)
        assert cmp.presence_.shape == (6, 3)
        assert not cmp.presence_[0, 0]
        assert not cmp.presence_[1, 0]
        assert cmp.presence_[2, 0]
        df = cmp.test_diff_freq(groups, statistic="log_l2", n_perm=20, random_state=0)
        assert {"n_obs_A", "n_obs_B"}.issubset(df.columns)

    def test_presence_threshold_uses_masked_glm_comparison(self):
        rng = np.random.default_rng(1)
        ny = nx = 12
        gene_names = ["g0", "g1", "g2"]
        coords = (
            np.stack(np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"), axis=-1)
            .reshape(-1, 2)
            .astype(float)
        )
        samples = []
        for _ in range(6):
            X = rng.uniform(0.1, 1.0, size=(ny * nx, 3))
            a = ad.AnnData(X=X)
            a.var_names = gene_names
            a.obsm["spatial"] = coords
            samples.append(a)
        for i in (0, 1):
            samples[i].X[:, 0] = 0.0

        design = pd.DataFrame({"time": np.linspace(0.0, 1.0, 6)})
        cmp = ComparatorIrregular(
            samples,
            gene_names,
            presence_threshold=0.5,
        ).compute_spectra(progress=False)

        df = cmp.test_diff_freq(design, contrast="time", statistic="log_l2")
        expected = compare_glm_masked(
            cmp.spectra_,
            design,
            "time",
            cmp.presence_,
            gene_names=cmp.gene_names,
        )

        assert {"n_obs", "df_resid"}.issubset(df.columns)
        pd.testing.assert_frame_equal(
            df.sort_values("Feature").reset_index(drop=True),
            expected.sort_values("Feature").reset_index(drop=True),
        )


class TestComparatorIrregularApiUnification:
    """Rename guards for retired Comparator method names."""

    def test_retired_method_names_are_absent(self):
        gene_names = ["a", "b"]
        adatas = _grid_samples_to_adata([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, gene_names)
        for old in ("test_pattern", "test_expression", "test", "normalize_shape"):
            assert not hasattr(cmp, old), f"Comparator.{old} should have been retired in the rename"


class TestComparatorCrossConsistency:
    """ComparatorGrid and ComparatorIrregular should agree on planted signal ranking."""

    def _paired_samples(self, ny: int = 16, nx: int = 16, n_per_group: int = 3, seed: int = 0):
        """Build matched grid and AnnData inputs for cross-comparator checks."""

        rng = np.random.default_rng(seed)
        grids = []
        groups = []
        y_col = np.arange(ny)[:, None]
        x_row = np.arange(nx)[None, :]
        stripes_y = np.broadcast_to(np.sin(2 * np.pi * y_col / ny), (ny, nx))
        stripes_x = np.broadcast_to(np.cos(2 * np.pi * x_row / nx), (ny, nx))
        stripes_hi = np.broadcast_to(np.sin(2 * np.pi * y_col / (ny / 2)), (ny, nx))
        for gi in range(2 * n_per_group):
            group = 0 if gi < n_per_group else 1
            g0 = stripes_y if group == 0 else stripes_hi
            g1 = rng.standard_normal((ny, nx))
            g2 = stripes_x
            grids.append(np.stack([g0, g1, g2], axis=0))
            groups.append(group)
        return grids, np.array(groups)

    def test_grid_and_irregular_rank_planted_gene_consistently(self):
        from scipy.stats import spearmanr

        grids, groups = self._paired_samples()
        gene_names = [f"g{i}" for i in range(grids[0].shape[0])]

        adatas = _grid_samples_to_adata(grids, gene_names)
        cmp_n = ComparatorIrregular(adatas, gene_names)
        cmp_n.compute_spectra(progress=False)
        df_n = cmp_n.test_diff_freq(groups, n_perm=200, random_state=0)

        try:
            samples_sd = []
            for grid_arr in grids:
                sample = _make_bin_sdata(
                    ny=grid_arr.shape[1],
                    nx=grid_arr.shape[2],
                    gene_values={g: grid_arr[i] for i, g in enumerate(gene_names)},
                )
                samples_sd.append(sample)
            cmp_g = ComparatorGrid(
                samples_sd,
                bins="bins",
                table_name="table",
                col_key="col_idx",
                row_key="row_idx",
                n_radial_bins=6,
                fft_chunk_size=2,
            )
        except Exception as exc:  # pragma: no cover — depends on spatialdata version
            pytest.skip(f"ComparatorGrid fixture unavailable: {type(exc).__name__}")
        cmp_g.compute_spectra(progress=False)
        df_g = cmp_g.test_diff_freq(groups, n_perm=200, random_state=0)

        # Normalize: both frames are indexed by/contain Feature.
        def _by_feature(df):
            return df.set_index("Feature") if "Feature" in df.columns else df

        df_n = _by_feature(df_n)
        df_g = _by_feature(df_g)

        # ``g0`` (the planted group difference) must be significant on both.
        assert df_n.loc["g0", "P_value"] < 0.2, f"NUFFT P(g0) = {df_n.loc['g0', 'P_value']:.3f}"
        assert df_g.loc["g0", "P_value"] < 0.2, f"FFT P(g0)  = {df_g.loc['g0', 'P_value']:.3f}"

        # Spearman rank-correlation on the 3-gene p-value vector.
        common = sorted(set(df_n.index) & set(df_g.index))
        p_n = df_n.loc[common, "P_value"].to_numpy()
        p_g = df_g.loc[common, "P_value"].to_numpy()
        rho, _ = spearmanr(p_n, p_g)
        assert rho >= 0.5, f"Spearman(P_nufft, P_fft) = {rho:.2f}"
