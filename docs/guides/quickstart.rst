Quick Start
===========

A 5-minute tour. ``quadsv`` does three things:

1. **Score one feature** with :func:`~quadsv.spatial_q_test`. Does
   its expression depend on space?
2. **Score every feature in a tissue** with :func:`~quadsv.Detector`.
   Which genes are spatially variable?
3. **Compare slides** with :func:`~quadsv.Comparator`. Do two
   groups of samples differ in spatial pattern?

The kernel you pass to the single-sample tests decides what kind of
spatial structure earns a high score. CAR and Matérn kernels reward
smooth gradients; a graph-Laplacian kernel rewards sharp differences
between neighbouring spots. See :doc:`/guides/kernels`. For
cross-sample comparisons, ``quadsv`` compares frequency-domain
pattern spectra so samples do not need to be spatially registered.


The four layers
---------------

Every name listed below is importable from the top-level package
with ``from quadsv import ...``.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Layer
     - What it does
     - Public names
   * - **Kernels**
     - Encode the spatial structure to look for.
     - :class:`~quadsv.MatrixKernel` (any coords or graph),
       :class:`~quadsv.FFTKernel` (regular grid),
       :class:`~quadsv.NUFFTKernel` (irregular 2-D coords).
       Backend authors can subclass
       :class:`quadsv.kernels.Kernel` or
       :class:`quadsv.kernels.MatrixKernelBase` for a custom
       backend; see :doc:`/guides/kernels`.
   * - **Tests**
     - Compute the test statistic and a p-value on a feature
       vector or batch.
     - :func:`~quadsv.spatial_q_test` (univariate),
       :func:`~quadsv.spatial_r_test` (bivariate),
       and helpers :func:`~quadsv.compute_null_params`,
       :func:`~quadsv.auto_chunk_size`, :func:`~quadsv.liu_sf`.
   * - **Detectors**
     - Genome-wide pattern screening on one sample.
     - :class:`~quadsv.DetectorIrregular`,
       :class:`~quadsv.DetectorGrid`, and the dispatch factory
       :func:`~quadsv.Detector`.
   * - **Comparators**
     - Cross-sample pattern comparison between groups of slides.
     - :class:`~quadsv.ComparatorIrregular`,
       :class:`~quadsv.ComparatorGrid`, and the dispatch factory
       :func:`~quadsv.Comparator`.


Test one feature
----------------

Score whether a gene's expression depends on space, given a kernel.

.. code-block:: python

   import numpy as np
   from quadsv import NUFFTKernel, spatial_q_test

   rng = np.random.default_rng(0)
   coords = rng.uniform(0, 20, size=(500, 2))
   gene = rng.standard_normal(500)

   kernel = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
   Q, pval = spatial_q_test(gene, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

Reading the result:

- **High Q with low p-value.** The gene's expression depends on
  location, in the way this kernel looks for. Here the kernel is
  Matérn, which looks for smooth large-scale gradients.
- **Low Q with high p-value.** The gene looks spatially independent
  under this kernel.

The kernel choice matters. Swap the Matérn for a graph-Laplacian
kernel and a gene that scored low above can score high if its
expression changes sharply between neighbouring spots. See
:doc:`/guides/kernels` for picking a kernel.

The same :func:`~quadsv.spatial_q_test` call works with any kernel
type. Pass a :class:`~quadsv.MatrixKernel` for an arbitrary
coordinate cloud or graph, an :class:`~quadsv.FFTKernel` for a
regular 2-D grid, or :class:`~quadsv.NUFFTKernel` for irregular 2-D
coordinates. The companion :func:`~quadsv.spatial_r_test` tests two
features at a time for spatial co-expression.

.. dropdown:: Reuse the null fit across many features

   When you test many features against the same kernel, precompute
   the null distribution once with
   :func:`~quadsv.compute_null_params` and pass the result back into
   the test:

   .. code-block:: python

      from quadsv import compute_null_params, spatial_q_test

      null = compute_null_params(kernel, method="liu")  # one-time cost
      for gene in gene_matrix.T:
          Q, pval = spatial_q_test(gene, kernel, null_params=null)

   :func:`~quadsv.spatial_q_test` and :func:`~quadsv.spatial_r_test`
   also accept a ``chunk_size`` keyword. The default ``"auto"``
   dispatches to :func:`~quadsv.auto_chunk_size` to size each batch
   for the kernel's cache sweet spot. See :doc:`/guides/scaling` for
   the cost model.


Test every feature in an AnnData
--------------------------------

The :func:`~quadsv.Detector` factory picks the right detector class
from the input type. An :class:`anndata.AnnData` returns a
:class:`~quadsv.DetectorIrregular`; a
:class:`spatialdata.SpatialData` returns a
:class:`~quadsv.DetectorGrid`.

Expected ``adata`` layout:

- ``adata.X`` (or a layer in ``adata.layers``) is the
  ``(n_obs, n_vars)`` count or expression matrix. Sparse formats
  are fine. The detector consumes one column at a time, so you do
  not need to densify up front.
- ``adata.obsm[obsm_key]`` is an ``(n_obs, 2)`` or ``(n_obs, 3)``
  array of spatial coordinates in some physical unit. The
  ``bandwidth`` argument should be in the same unit. Required for
  the ``"nufft"`` backend and for any distance-based kernel
  (``"gaussian"``, ``"matern"``).
- ``adata.obsp[obsp_key]`` is an ``(n_obs, n_obs)`` adjacency,
  affinity, or distance matrix. Used by the ``"matrix"`` backend
  when you want to feed a precomputed graph instead of building
  one from coordinates.

You need at least one of ``obsm_key`` or ``obsp_key``. If you pass
both, ``obsp_key`` wins.

Build the detector, attach the data with
:meth:`~quadsv.DetectorIrregular.setup_data`, then run
:meth:`~quadsv.DetectorIrregular.compute_qstat`:

.. code-block:: python

   import anndata as ad
   from quadsv import Detector

   adata = ad.read_h5ad("spatial_tissue.h5ad")
   print(f"Data: {adata.n_obs} spots × {adata.n_vars} genes")

   detector = Detector(
       adata,
       kernel_method="matern",
       backend="nufft",
       bandwidth=25.0,   # same units as adata.obsm["spatial"]
       nu=1.5,
   ).setup_data(adata, obsm_key="spatial", min_cells_frac=0.05)

   results = detector.compute_qstat(n_jobs=4, return_pval=True)
   svgs = results[results["P_adj"] < 0.05]
   print(f"Found {len(svgs)} SVGs at FDR < 5%")

The same detector handles spatial co-expression through
:meth:`~quadsv.DetectorIrregular.compute_rstat`:

.. code-block:: python

   top_genes = results.nlargest(100, "Q").index.tolist()
   coexp = detector.compute_rstat(
       features_x=top_genes,
       features_y=None,    # all pairs within ``features_x``
       n_jobs=4,
       return_pval=True,
   )

.. dropdown:: Picking a backend (matrix vs nufft)

   :class:`~quadsv.DetectorIrregular` ships two backends, selected
   with the ``backend`` keyword.

   ``backend="nufft"`` builds a :class:`~quadsv.NUFFTKernel`. It
   runs at ``O(n log n)`` per feature and never materialises an
   ``(n, n)`` matrix, so it scales to large ``n``. Use it with
   smooth kernels (Gaussian, Matérn).

   ``backend="matrix"`` builds a :class:`~quadsv.MatrixKernel`,
   which picks dense, sparse, or sparse-precision storage based on
   ``n``. Use it for graph kernels (``car``, ``moran``,
   ``graph_laplacian``) or when you have a precomputed adjacency in
   ``adata.obsp[obsp_key]``.

   Example with the matrix backend and a CAR kernel:

   .. code-block:: python

      detector = Detector(
          adata,
          kernel_method="car",
          backend="matrix",
          rho=0.9,
          k_neighbors=15,
      ).setup_data(adata, obsm_key="spatial", min_cells_frac=0.05)


Large regular grids (Visium HD)
-------------------------------

For rasterised grids in :class:`spatialdata.SpatialData` containers,
the same :func:`~quadsv.Detector` factory returns a
:class:`~quadsv.DetectorGrid`. Kernel hyper-parameters go to the
constructor. The bin / table / coordinate layout goes to
:meth:`~quadsv.DetectorGrid.setup_data`.

Expected ``sdata`` layout:

- A bin element ``sdata[bins]`` (typically a
  :class:`geopandas.GeoDataFrame` of bin polygons) that defines
  the rasterisation grid. For Visium HD this is one of the
  ``square_002um`` / ``square_008um`` / ``square_016um``
  shape collections; for imaging data, any shape collection
  whose footprint covers the rectangular grid you want to
  rasterise against.
- A table ``sdata.tables[table_name]`` whose ``X`` is the
  ``(n_bins, n_vars)`` expression matrix and whose ``obs``
  carries the integer column / row indices of each bin
  (``col_key`` and ``row_key``).
- The pair ``(col_key, row_key)`` must yield a contiguous
  rectangular layout. Missing bins are filled with zeros.

Code:

.. code-block:: python

   import spatialdata as sd
   from quadsv import Detector

   sdata = sd.read_zarr("visium_hd.zarr")
   detector = Detector(
       sdata,
       kernel_method="car",
       rho=0.9,
       neighbor_degree=1,
       topology="square",
   ).setup_data(
       sdata,
       bins="square_008um",       # name of the bin element in sdata
       table_name="square_008um", # name of the table in sdata.tables
       col_key="array_col",       # integer column index in table.obs
       row_key="array_row",       # integer row index in table.obs
       min_count=10,
   )
   results = detector.compute_qstat(n_jobs=4, return_pval=True)


Compare Patterns Across Samples
-------------------------------

The :func:`~quadsv.Comparator` factory picks
:class:`~quadsv.ComparatorIrregular` for a list of
:class:`anndata.AnnData` samples and :class:`~quadsv.ComparatorGrid`
for a list of :class:`spatialdata.SpatialData` samples. The
pattern-comparison path is alignment-free: each sample is converted
to per-gene spatial power spectra, spectra are reduced to common
radial frequency bins, and per-gene group differences are tested
with :meth:`~quadsv.ComparatorIrregular.test_diff_freq`.

For AnnData samples:

.. code-block:: python

   import anndata as ad
   import numpy as np
   from quadsv import Comparator

   paths = [
       "control_1.h5ad",
       "control_2.h5ad",
       "control_3.h5ad",
       "case_1.h5ad",
       "case_2.h5ad",
       "case_3.h5ad",
   ]
   samples = [ad.read_h5ad(path) for path in paths]
   design = np.array([0, 0, 0, 1, 1, 1])  # 1-D labels -> binary contrast

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

``pattern_hits`` is sorted by evidence of differential spatial
pattern and has columns ``Feature``, ``Statistic``, ``P_value``, and
``P_adj``. ``normalize_shape=True`` isolates redistribution of power
across radial frequencies from overall pattern amplitude. The
companion ``expression_hits`` table tests the DC component, so a gene
can be pattern-only, expression-only, or both.

For :class:`spatialdata.SpatialData` grids, keep the same
``Comparator`` factory and pass the grid rasterization keys:

.. code-block:: python

   import numpy as np
   import spatialdata as sd
   from quadsv import Comparator

   samples = [sd.read_zarr(path) for path in zarr_paths]
   design = np.array([0, 0, 0, 1, 1, 1])

   cmp = Comparator(
       samples,
       bins="square_008um",
       table_name="square_008um",
       col_key="array_col",
       row_key="array_row",
   ).compute_spectra().normalize_background()

   pattern_hits = cmp.test_diff_freq(design, statistic="log_l2")

See :doc:`/guides/multisample` for covariate residualisation,
permutation nulls, GLM contrasts, unit scaling, and Visium-specific
notes.


Next steps
----------

- :doc:`/guides/kernels`. Pick a kernel for your data.
- :doc:`/guides/multisample`. Compare spatial patterns across groups
  of slides.
- :doc:`/guides/scaling`. Performance and complexity reference.
- :doc:`/guides/theory`. Mathematical background.
- :doc:`/autoapi/quadsv/index`. Full API reference.
