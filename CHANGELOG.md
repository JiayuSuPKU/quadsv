# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **GLM design API for cross-sample pattern comparison.** New public
  `compare_glm(spectra, design, contrast, …)` generalises the
  two-group test to arbitrary OLS designs (binary, continuous,
  multi-factor) with an analytic Wald test. The two-group case is
  recovered exactly. `Comparator.test_diff_freq(...)` gains a
  `contrast=` argument (column name, dict, or contrast vector).
- **Analytic null for `log_l2`** (`null="analytic"`) on
  `compare_two_groups`, `compare_two_groups_masked`, and
  `compare_glm`. Per-gene statistic is integrated via Liu's
  approximation against a pooled-across-genes **full** within-group
  Σ (a single 30×30 eigendecomposition before each Liu integration);
  bypasses the small-n permutation BH-floor while keeping mean
  within-group null FPR at ~0.012 across the three benchmark panels.
  Emits a `UserWarning` at residual df < 3. The masked variant uses
  a mask-aware pooled estimator with per-gene noncentrality scaling
  so genes with different observed cohorts get correctly-scaled
  eigenvalues.
- **Analytic Welch t tests for `compare_two_groups_scalar`**.
  Computes per-gene two-sided p-values from the
  Welch-Satterthwaite t-distribution; the scalar DE companion uses
  this fixed analytic null rather than exposing a `null=` selector.
- **`normalize_shape: bool = False` keyword** on every spectrum-input
  comparison test (`compare_two_groups`, `compare_two_groups_masked`,
  `compare_glm`). When True, divides each per-(sample, gene)
  spectrum by its sum along the frequency axis before the statistic
  is computed, so the test fires only on shape-only redistribution
  of power across radial frequencies. Statistic-agnostic; default
  False preserves prior behaviour.
- **Comparator null-covariance and effective-rank diagnostics**:
  `Comparator.effective_rank(weights=None)` for per-sample heterogeneity 
  in the gene spectrum cross-frequency covariance; 
  `Comparator.estimate_null_covariance(design, contrast=...)`for the
  pooled log-spectrum covariance, weighted covariance, scaled Liu
  eigenvalues, effective rank, residual df, and masked-path eligibility
  metadata used by `test_diff_freq(..., statistic="log_l2", null="analytic")`.
- **Top-level convenience exports**:
  `quadsv.Detector(data, …)` and `quadsv.Comparator(data_list, …)`
  factories that dispatch on `AnnData` vs `SpatialData`;
  `quadsv.compute_null_params`, `quadsv.auto_chunk_size`,
  `quadsv.liu_sf` promoted to top level (canonical
  `quadsv.statistics` paths still work).
- **Public-API freeze test** (`tests/test_public_api.py`) snapshots
  `__all__`, docstring presence, canonical-path identity, and
  asserts removed legacy paths raise `ModuleNotFoundError`.
- **Convenience input modes for `Comparator.normalize_covariates`.**
  In addition to the existing per-sample
  `Sequence[np.ndarray]` of pre-rasterized
  `(n_covariates, ny, nx)` images, the method now accepts a shared
  `Sequence[str]` of column names; the subclass interprets it
  natively:

    * `ComparatorIrregular` looks each key up in `adata.obs.columns`
      first, then `adata.var_names` (preferring obs on collision);
      the resolved per-spot vector is NUFFTed directly onto the
      sample's k-grid — so the same call accepts deconvolved
      cell-type proportion columns *and* per-gene expression columns
      (e.g., a housekeeping or marker gene) interchangeably.
    * `ComparatorGrid` forwards the keys as `value_key=` to
      `spatialdata.rasterize_bins`, so any combination of `.obs`
      columns and `var_names` in the comparator's table works.

  Mode is detected from the first element's type. Both paths reduce
  to the same `(n_covariates, K)` per-sample features fed into the
  log-space residualization, so the math is identical — only the
  input boilerplate is different.

### Changed
- **Breaking: package layout migrated to `src/quadsv/`** with the
  four conceptual layers as physical subpackages —
  `quadsv.kernels.{fft,nufft}`,
  `quadsv.detectors.{base,irregular,grid}`,
  `quadsv.comparators.{__init__,multisample}`. `import quadsv` and
  `from quadsv import …` keep working; editable installs must be
  reissued (`pip install -e ".[dev]"`). Lint / format commands now
  target `src/ tests/`.
- **Breaking: unified `normalize_*` surface API in
  `quadsv.comparators.multisample`** (no aliases):
    * `normalize_by_background` → `normalize_background`
    * `residualize_against_covariates` → `normalize_covariates`
    * `shape_normalize` → `normalize_shape`
  Consistent first-arg `spectra`, keyword-only after, `eps=1e-12`
  default on every helper, and NumPy-style docstrings with LaTeX
  math. `normalize_covariates`'s first positional arg is renamed
  `gene_spectra` → `spectra`, and its implementation now operates in
  **log-space**: it residualises `log(spectra + ε)` against
  `[1, log(C^T + ε)]` and exponentiates, so the output stays strictly
  positive and composes cleanly with downstream `log_l2` tests.
  Log-space `normalize_covariates` also **commutes exactly** with
  `normalize_background` (left- vs right-multiplication of the
  log-spectrum matrix by orthogonal-projection matrices on disjoint
  axes), so the two can be applied in either order. The remaining
  chainable comparator instance method follows the rename:
    * `.residualize()` → `.normalize_covariates()`
- **Breaking: `Comparator.fit()` renamed to `Comparator.compute_spectra()`**.
  The method computes per-sample radial-binned power spectra rather than
  fitting model parameters; the new name describes the operation
  directly and matches the codebase's verb-first method convention. All
  three keyword arguments (`n_jobs`, `landmark_genes`, `progress`) and
  the chainable `return self` behaviour are unchanged.
- **Breaking: `design` moved from Comparator constructor to test
  time.** The cross-sample contrast is no longer a construction
  argument — it is supplied directly to `.test_diff_freq(design, ...)`
  / `.test_diff_expr(design, ...)` (positional first arg). A single fitted comparator can now
  serve any number of unrelated contrasts on the same `spectra_`
  without recomputing per-sample spectra. `min_samples_per_group`
  follows `design` to `test_diff_freq` (kwarg) since it's a property
  of the design's group sizes, not of the spectra. `design` accepts
  the same three forms as before:
    * 1-D array / Series of binary labels → two-sample dispatch
      (`compare_two_groups` / `compare_two_groups_masked`);
    * 2-D `np.ndarray` of shape `(n_samples, p)` → GLM design matrix,
      used verbatim by `compare_glm`;
    * `pandas.DataFrame` → GLM design, patsy-encoded by
      `compare_glm`.
- **Breaking: default `null` switched from `"permutation"` to
  `"analytic"` across the spectral comparison surface** —
  `Comparator.test_diff_freq`,
  `quadsv.comparators.multisample.compare_two_groups`, and
  `quadsv.comparators.multisample.compare_two_groups_masked`. The
  analytic Wald test (Liu mixture-χ² null) bypasses the small-n permutation
  BH-floor and is the only path that works on every dispatch target
  (binary permutation/analytic + GLM analytic), so it makes a single sensible
  package-wide default. Callers who want the permutation null must
  now pass `null="permutation"` explicitly. As a related
  ergonomic fix, `compare_two_groups{,_masked}(statistic="welch_t_cauchy",
  null="analytic")` no longer raises — `welch_t_cauchy` carries its own
  analytic null (documented as ignoring the `null` kwarg) so the
  package default `null="analytic"` is treated as a no-op for that
  statistic.
- **Breaking: statistical-test naming cleanup** in
  `quadsv.comparators.multisample` and the corresponding
  `Comparator.test_diff_*` methods:
    * **`compare_designs` → `compare_glm`.** The plural form was
      awkward (one design per call); `compare_glm` names the test
      family at the call site and parallels the binary
      `compare_two_groups` cleanly.
    * **Statistic `"cauchy_welch"` → `"welch_t_cauchy"`.** The new
      token reads in pipeline order (per-bin Welch t first, gene-level
      Cauchy combination second) and disambiguates from naming the
      gene-level aggregator alone.
    * **Scalar DE `null=` selector retired** on
      `compare_two_groups_scalar` / `Comparator.test_diff_expr`.
      Scalar DE now always uses the analytic Welch-Satterthwaite
      t-distribution null.
    * **`null="liu"` alias retired.** The `liu` token referred to
      the numerical algorithm used to integrate the analytic χ² mixture
      tail (see `quadsv.statistics.liu_sf`), not a separate
      statistical concept. Single canonical token: `analytic`.
- **Breaking: Comparator attribute surface narrowed** (sklearn-style
  moderate-privacy convention). The public surface is now `samples`,
  `gene_names`, `feature_mode`, `freq_edges`, plus the
  trailing-underscore fitted attributes (`spectra_`, `dc_`,
  `presence_`, `rotation_angles_`). `design`/`groups_` are no longer
  carried as instance state — the comparator is design-agnostic.
  Internal config knobs that were inadvertently public are now
  single-underscore-prefixed: `_n_radial_bins`, `_fft_solver`,
  `_workers`, `_presence_threshold`, `_spacings`, `_grid_shapes`,
  `_spectrum_fft_solver`, `_fft_chunk_size`, `_spacing_override`,
  `_bins`, `_table_name`, `_col_key`, `_row_key`, `_value_key`.
- **Breaking: Comparator test methods renamed and aligned with the
  standalone `compare_*` API** in `quadsv.comparators.multisample`:
    * `.test_pattern()`    → `.test_diff_freq()` — gains a new
      `normalize_shape: bool = False` keyword, forwarded to its
      dispatch target (`compare_two_groups`,
      `compare_two_groups_masked`, or `compare_glm`) so users get
      the shape-only DF path without mutating `cmp.spectra_`.
    * `.test_expression()` → `.test_diff_expr()` — uses analytic
      t-distribution tests for scalar DE and supports both binary
      two-group and GLM contrast designs.

### Removed
- **Breaking: `groups=` / `design=` constructor kwargs on
  `ComparatorIrregular` and `ComparatorGrid` are gone.** Supply the
  1-D labels or design matrix to the test method instead
  (`cmp.test_diff_freq(design, ...)`, `cmp.test_diff_expr(design,
  ...)`). The comparator no longer carries design state; one fitted
  comparator can serve any number of contrasts on the same spectra.
- **Breaking: `Comparator.shape_normalize()` chainable method
  retired.** Use the equivalent
  `cmp.test_diff_freq(..., normalize_shape=True)` keyword path for the
  one-shot non-destructive test, or call
  `quadsv.comparators.multisample.normalize_shape(cmp.spectra_)`
  directly to obtain the standalone transform. The previous in-place
  method silently mutated `cmp.spectra_` and surprised subsequent
  `.test_diff_freq()`/`.test_diff_expr()` calls on the same comparator.
- **Breaking: the `test = test_pattern` alias retired.** Use the
  explicit `cmp.test_diff_freq(...)` (or `cmp.test_diff_expr(...)` for
  the DE companion); the unqualified `cmp.test()` was ambiguous once
  the API exposed two complementary tests.
- **Breaking: `center` argument retired** across the comparator API.
  `ComparatorIrregular`, `ComparatorGrid`, and
  `compute_sample_spectrum` no longer accept `center`. Per-gene
  mean centring (the previous default) is now the only spectrum
  normalisation path. The `_ZSCORE_CLIP` constant, the
  `zscore_clip` parameter, and the per-bin clamp in the NUFFT loop
  are deleted (~50 LOC).
- **Breaking: `benchmark_statistics` function and the matching
  `Comparator.benchmark()` method retired.** Invoke
  `compare_two_groups` directly with each `statistic=` value to A/B
  compare on the same fitted spectra (~95 LOC).
- **Breaking: `statistic="hotelling_lw"` and `statistic="mmd_rbf"`
  paths retired** from every comparison function. Both were
  impractically slow and consistently dominated on sensitivity by
  `log_l2 + null='analytic'` or `welch_t_cauchy`. `_AVAILABLE_STATISTICS`
  now reads `("log_l2", "welch_t_cauchy")`.
- **Breaking: six legacy-path shim modules removed** —
  `quadsv.fft`, `quadsv.nufft`, `quadsv.detector`,
  `quadsv.detector_grid`, `quadsv._detector_base`,
  `quadsv.multisample`. Use the canonical subpackage paths.
- **Breaking: backend ABCs `Kernel` and `MatrixKernelBase` no
  longer re-exported from top-level `quadsv`**. They live at
  `quadsv.kernels` and are intended for backend authors.

### Fixed
- CI workflow install step referenced non-existent extras
  (`[dev,test,spatial]` and `[docs,spatial]`); narrowed to the
  actual `[dev]` / `[docs]` extras in `pyproject.toml`.

## Release Process

- [ ] Run full test suite: `pytest tests/ --cov=quadsv`
- [ ] Check documentation builds: `sphinx-build -b html docs/ docs/_build/`
- [ ] Update version in `pyproject.toml`
- [ ] Update this CHANGELOG
- [ ] Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Build package: `python -m build`
- [ ] Upload to PyPI: `python -m twine upload dist/*`

## [0.1.0] - 2026-02-02

### Added
- Initial public release
- Q-test framework for univariate spatial pattern detection
- R-test framework for bivariate spatial co-expression
- Core kernel methods: Gaussian, Matérn, CAR, Graph Laplacian, Moran's I
- Implicit mode for scalable large-N computation (N > 5000)
- FFT acceleration for regular grid data (Visium HD)
- `DetectorIrregular` for AnnData integration (genome-wide SVG detection)
- `DetectorGrid` for large-scale Visium HD analysis
- Null approximation methods: CLT, Welch/Satterthwaite, Liu
- Comprehensive test suite (unit + integration tests)
- Tutorial test cases demonstrating all major workflows
- Complete documentation with quickstart and theory sections
- Support for Python 3.10, 3.11, 3.12

## [0.1.1]

### Fixed
- Fix type hinting issues in `quadsv.kernels` module
