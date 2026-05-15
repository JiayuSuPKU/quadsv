Cross-sample Comparison
=======================

Suppose you have two groups of spatial-omics samples (for example a
set of healthy controls and a set of cancer sections) and want to
ask which genes show the biggest spatial-pattern difference between
the groups. The :class:`~quadsv.ComparatorIrregular` and
:class:`~quadsv.ComparatorGrid` classes give you a frequency-domain
pipeline that does this without spatial registration. The default
null is an analytic Liu-approximation Wald test (``null="wald"``);
a label-permutation null is available on the binary path with
``null="permutation"``. The GLM (multi-column / continuous design)
path is Wald-only.

.. note::

   This API is under active development. Signatures may shift
   between minor releases.


Why frequency domain?
---------------------

The 2-D power spectrum :math:`|\hat x(k)|^2` of a rasterised gene
image is translation-invariant: shifting the image leaves the
spectrum unchanged. Radial averaging additionally makes the
representation rotation-invariant. Together these mean samples never
need to be spatially registered onto each other, which would
otherwise be a hard requirement when (for example) healthy and
cancer slides have no shared anatomy.


Five-step pipeline
------------------

:class:`~quadsv.ComparatorIrregular` chains five stages:

1. Per-sample 2-D power spectra (``compute_spectra``).
2. Reduction to a low-dimensional feature vector. The default is
   radial 1-D bins.
3. Background normalisation that cancels per-slide differences in
   gain and sensitivity (``normalize_background`` — geometric-mean
   spectrum across all genes per sample).
4. Optional residualisation against covariate spectra
   (``normalize_covariates``: cell-type maps, tissue-domain
   indicators, ...).
5. Per-gene two-group / GLM-contrast test
   (``test_diff_freq(design, ...)``) with BH-FDR correction. The
   ``design`` argument is supplied at test time, not at construction,
   so a single fitted comparator can serve any number of unrelated
   comparisons on the same spectra.

.. dropdown:: DC vs AC: separating expression level from pattern shape

   The pipeline always mean-centres each gene's spatial signal before
   the FFT, splitting the information cleanly into two orthogonal
   pieces.

   The **DC scalar** is the per-sample grid mean, i.e. total
   normalised expression. It is tested across groups with
   :meth:`~quadsv.ComparatorIrregular.test_diff_expr`, which runs
   an analytic Welch-Satterthwaite t-test by default
   (``null="wald"``; ``null="permutation"`` is also available) with
   BH-FDR. This is a spatially-aware differential-expression test.

   The **AC spectrum** is the pattern shape, with DC exactly zero.
   It is tested with
   :meth:`~quadsv.ComparatorIrregular.test_diff_freq` using one of
   the two statistics listed below.

   The two tests carry complementary information. A gene may be
   "only DE" (same pattern, different magnitude), "only pattern"
   (same total expression, different spatial layout), or both. Run
   them side by side and inspect where the hits overlap and where
   they separate.

The two pattern-test statistics ship out of the box and share a
common dispatch, so they are directly comparable:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Statistic
     - What it measures
   * - ``log_l2`` (default)
     - Quadratic form ``T² = D'WD`` on log-spectra differences.
       Supports analytic ``null="wald"`` (Liu mixture-χ² tail; the
       default on :meth:`~quadsv.ComparatorIrregular.test_diff_freq`)
       and ``null="permutation"`` on the binary path. The Wald null
       bypasses the BH-FDR floor that the exact permutation test
       hits at small per-arm n, and is the only path that works on
       multi-column / continuous designs.
   * - ``welch_t_cauchy``
     - Cauchy combination of per-bin Welch t-statistics. Analytic
       null is built in; remains well-calibrated at very small n.
       Binary path only.

Both run through :func:`quadsv.comparators.multisample.compare_two_groups`
(or :meth:`quadsv.ComparatorIrregular.test_diff_freq` for the class API);
flip ``statistic="log_l2"`` ↔ ``statistic="welch_t_cauchy"`` to compare on
the same fitted spectra.

Minimal pattern-comparison call
-------------------------------

If every sample is an :class:`anndata.AnnData` with shared
``var_names`` and coordinates in ``obsm["spatial"]``, the
:func:`~quadsv.Comparator` factory is enough:

.. code-block:: python

   import numpy as np
   from quadsv import Comparator

   design = np.array([0, 0, 0, 1, 1, 1])  # one label per sample
   cmp = (
       Comparator(samples)
       .compute_spectra(n_jobs=4)
       .normalize_background()
   )
   pattern_hits = cmp.test_diff_freq(
       design,
       statistic="log_l2",
       normalize_shape=True,
   )
   expression_hits = cmp.test_diff_expr(design)

``pattern_hits`` asks whether the spatial layout differs between
groups. ``expression_hits`` asks whether the sample-level means differ.
Running both keeps pattern-only changes separate from ordinary
differential expression.


Picking a class
---------------

Two backends, mirroring the detector layer:

- :class:`~quadsv.ComparatorIrregular` takes a list of
  :class:`anndata.AnnData` (irregular spots, common across Visium,
  Slide-seq, Stereo-seq, MERFISH). Spectra are computed with a
  batched type-1 NUFFT. Each sample keeps its own grid shape and
  spacing. Cross-sample comparability comes from radial binning in
  physical-frequency space.
- :class:`~quadsv.ComparatorGrid` takes a list of
  :class:`spatialdata.SpatialData` (regular rasterised bins, e.g.
  Visium HD). Spectra are computed with a single batched 2-D FFT
  per sample.

Sparse ``adata.X`` and layer matrices are not densified up front.
The spectrum loop converts exactly one gene column at a time. The
:func:`~quadsv.Comparator` factory dispatches between the two
classes based on the input list type. Mixed lists raise
``TypeError``.


Toy walkthrough (AnnData / NUFFT)
---------------------------------

Eight synthetic samples (4 per group) of 10 genes. Gene ``g0``
carries a low-frequency stripe pattern in group 1 only.

.. code-block:: python

   import anndata as ad
   import numpy as np
   from quadsv import ComparatorIrregular

   rng = np.random.default_rng(3)
   ny = nx = 32
   gene_names = [f"g{i}" for i in range(10)]

   def make_adata(group: int) -> ad.AnnData:
       yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
       coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
       X = rng.standard_normal((ny * nx, len(gene_names))) * 0.1
       if group == 1:
           X[:, 0] += 1.5 * np.sin(2 * np.pi * yy / 16.0).ravel()
       a = ad.AnnData(X=X)
       a.var_names = gene_names
       a.obsm["spatial"] = coords
       return a

   samples = [make_adata(0) for _ in range(4)] + [make_adata(1) for _ in range(4)]
   design = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 1-D labels → binary contrast

   cmp = (
       ComparatorIrregular(samples, gene_names)
       .compute_spectra()
       .normalize_background()
   )
   # Default null="wald" → analytic Liu-approximation Wald test.
   # Add null="permutation", n_perm=300, random_state=0 for the
   # label-permutation alternative.
   results = cmp.test_diff_freq(design, statistic="log_l2")
   print(results.head())

The implanted gene ``g0`` ranks first in the resulting table.


Walkthrough (SpatialData / FFT)
-------------------------------

For rasterised-grid samples, swap in :class:`~quadsv.ComparatorGrid`
and pass the same bin / table / coord keys you would pass to
:class:`~quadsv.DetectorGrid`:

.. code-block:: python

   import spatialdata as sd
   from quadsv import ComparatorGrid

   samples_sd = [sd.read_zarr(p) for p in paths_by_group]
   design = np.array([0] * len(paths_a) + [1] * len(paths_b))
   cmp = ComparatorGrid(
       samples_sd,
       bins="bin_shapes",            # SpatialElement name shared by every sdata
       table_name="counts",          # table inside each sdata
       col_key="array_col",          # obs column with bin-column indices
       row_key="array_row",          # obs column with bin-row indices
       value_key=None,               # None means rasterise expression off .X
       fft_chunk_size=256,           # genes per batched scipy.fft call
   ).compute_spectra().normalize_background()
   cmp.test_diff_freq(design, statistic="log_l2")


Mixed coordinate units (NUFFT path)
-----------------------------------

.. dropdown:: When samples ship coordinates in different physical units

   Some pipelines store coordinates in mixed units. For example one
   slide may be in μm and another in Visium full-resolution pixels
   at 0.35 μm/pixel. Pass ``unit_scales`` to convert each sample's
   raw coords into the common unit. Radial bins then come out in
   cycles per that unit on every sample:

   .. code-block:: python

      cmp = ComparatorIrregular(
          samples,
          gene_names=gene_names,
          unit_scales=[1.0, 0.35, 1.0, 0.35],
          spacing=(50.0, 50.0),       # common physical spacing, μm
          n_radial_bins=30,
      ).compute_spectra().normalize_background()
      cmp.test_diff_freq(design, statistic="log_l2")

   If ``grid_shape`` and ``spacing`` are left unset, each sample's
   k-grid is auto-inferred from its coords via
   :func:`quadsv.kernels.nufft._infer_grid_from_coords`.
   :func:`quadsv.kernels.nufft.power_spectrum_2d_nufft` is the
   lower-level primitive that runs one sample at a time.


Visium hex grids
----------------

For 10x Visium slides,
:func:`quadsv.utils.load_visium_sample` reads a Space Ranger output
directory into an :class:`anndata.AnnData`. You can feed that
:class:`~anndata.AnnData` directly to
:class:`~quadsv.ComparatorIrregular`. The NUFFT backend handles the
hex layout natively, no manual rasterisation needed. If you do want
the explicit hex-to-grid rasterisation,
:func:`quadsv.utils.visium_to_grid` returns a ``(n_genes, 78, 128)``
array and the physical spacing ``(dy, dx) = (100·√3/2, 50)`` μm per
cell for v1 Visium. The smallest resolvable pattern is roughly
``2 · 86.6 μm ≈ 173 μm`` along the coarser axis (the Nyquist
limit).


Choosing covariate maps for residualisation
-------------------------------------------

:meth:`~quadsv.ComparatorIrregular.normalize_covariates` takes one
of two shapes — a sequence of column-name strings shared across
samples, or a sequence of per-sample image arrays:

.. code-block:: python

   # 1. Shared column names — interpreted by the subclass.
   #    ComparatorIrregular: each key is looked up in adata.obs.columns
   #    first, then adata.var_names — so the same call accepts
   #    deconvolved cell-type proportion columns *and* per-gene
   #    expression columns (housekeeping / marker genes) interchangeably.
   cmp.normalize_covariates(["celltype_astro", "celltype_neuron", "MALAT1"])

   # ComparatorGrid: forward as `value_key=` to spatialdata.rasterize_bins
   # (works for .obs columns AND var_names in the comparator's table).
   cmp_g.normalize_covariates(["region_label", "MALAT1"])

   # 2. Pre-rasterized per-sample arrays of shape (n_covariates, ny_s, nx_s).
   #    Universal: works on either subclass; use when covariates aren't
   #    already attached to the sample containers.
   cmp.normalize_covariates(per_sample_arrays)

Useful covariate candidates:

- Cell-type proportion maps from a deconvolution tool such as
  Cell2location, CARD, or RCTD. One channel per cell type.
- Tissue-domain indicator maps from a spatial clustering method
  such as BayesSpace or GraphST.
- A composite "housekeeping" expression image to absorb depth
  gradients.

Residualisation is applied after background normalisation and
before testing.


See also
--------

- :doc:`/guides/quickstart` for the single-sample workflow.
- :doc:`/guides/scaling` for how the FFT and NUFFT routines scale.
- :class:`quadsv.ComparatorIrregular` and
  :class:`quadsv.ComparatorGrid` for the class reference.
- :func:`quadsv.comparators.multisample.compare_two_groups`,
  :func:`quadsv.comparators.multisample.compare_two_groups_masked`,
  and :func:`quadsv.comparators.multisample.compare_glm` for the
  array-level primitives.
- :func:`quadsv.kernels.fft.power_spectrum_2d` and
  :func:`quadsv.kernels.nufft.power_spectrum_2d_nufft` for the
  spectrum primitives.
