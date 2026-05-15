FAQ
===

**What does** ``quadsv`` **stand for?**
   "Quadratic-form spatial variability." Every test in the library
   reduces to the quadratic form

   .. math::

      Q_n = \mathbf{z}^\top \tilde{\mathbf{K}} \mathbf{z},

   where :math:`\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}`
   is the double-centred kernel matrix. See :doc:`/guides/theory`.

**Why is Moran's I problematic for SVG detection?**
   Moran's I uses an indefinite adjacency matrix as its kernel, so
   its eigenvalues span both signs. Patterns aligned with positive
   eigenspaces cancel patterns aligned with negative ones, which
   produces false negatives. Use the CAR kernel
   :math:`\mathbf{K} = (\mathbf{I} - \rho \tilde{\mathbf{W}})^{-1}`
   instead. It is strictly positive definite for any
   :math:`0 < \rho < 1`. See :doc:`/guides/theory` (Theorem 2).

   .. code-block:: python

      from quadsv import MatrixKernel

      kernel = MatrixKernel.from_coordinates(
          coords, method="car", k_neighbors=4, rho=0.9
      )

**What is the difference between Q-test and R-test?**
   :func:`~quadsv.spatial_q_test` is univariate:
   :math:`Q = \mathbf{z}^\top \mathbf{K} \mathbf{z}`. It asks
   whether *one* feature is spatially structured under the kernel.
   Use it to identify spatially variable genes.

   :func:`~quadsv.spatial_r_test` is bivariate:
   :math:`R = \mathbf{x}^\top \mathbf{K} \mathbf{y}`. It asks
   whether *two* features share a spatial pattern. Use it to find
   spatially co-expressed gene pairs.

**Which backend should I pick?**
   You can let the :func:`~quadsv.Detector` factory decide from your
   input type:

   .. code-block:: python

      from quadsv import Detector

      # AnnData → DetectorIrregular
      det = Detector(adata, kernel_method="matern", backend="nufft").setup_data(adata)

      # SpatialData → DetectorGrid
      det = Detector(sdata, kernel_method="car", rho=0.9).setup_data(sdata, ...)

   For explicit control:

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Backend
        - When to use
      * - ``backend="matrix"`` (:class:`~quadsv.MatrixKernel`)
        - Any coordinate cloud or graph. Pick this for ``car``,
          ``moran``, or ``graph_laplacian`` kernels, or when you have
          a precomputed adjacency in ``adata.obsp``. Storage
          (dense / sparse / sparse-precision) is selected from
          ``n``.
      * - ``backend="nufft"`` (:class:`~quadsv.NUFFTKernel`)
        - Irregular 2-D coordinates with around :math:`10^4` spots
          or more. Runs at ``O(n log n)`` per feature. Pairs with
          Gaussian or Matérn.
      * - :class:`~quadsv.DetectorGrid`
          (:class:`~quadsv.FFTKernel`)
        - Regular rasterised grids (Visium HD). Reads
          :class:`spatialdata.SpatialData` directly and uses an
          FFT.

**Can I use** ``quadsv`` **on non-spatial data?**
   Yes, as long as you can encode "closeness" as coordinates or as
   a graph. Common cases:

   - A k-NN graph in PCA space (single-cell trajectories).
   - A pseudo-time ordering or a lineage tree.
   - A custom adjacency in ``adata.obsp``.

   Pass coordinates to
   :meth:`quadsv.MatrixKernel.from_coordinates`, or a precomputed
   kernel or precision matrix to
   :meth:`quadsv.MatrixKernel.from_matrix`. To use an
   ``adata.obsp[key]`` directly, call
   :meth:`~quadsv.DetectorIrregular.setup_data` with
   ``obsp_key=key``. Add ``is_distance=True`` if the matrix stores
   distances rather than affinities.

**Does** ``quadsv`` **support 3-D coordinates?**
   The :class:`~quadsv.MatrixKernel` family does. Pass 3-D coords
   to :meth:`quadsv.MatrixKernel.from_coordinates` the same way you
   would for 2-D. The FFT and NUFFT backends are 2-D only for now.
   If you need 3-D Fourier acceleration, please open a feature
   request on `GitHub <https://github.com/JiayuSuPKU/quadsv/issues>`_.


Further help
------------

- :doc:`/guides/quickstart` for the getting-started tour.
- :doc:`/guides/theory` for derivations.
- :doc:`/autoapi/quadsv/index` for the API reference.
- `GitHub Issues <https://github.com/JiayuSuPKU/quadsv/issues>`_
  for bug reports and feature requests.
