Welcome
=======

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   guides/installation
   guides/quickstart
   guides/theory
   guides/scaling
   guides/kernels
   guides/multisample
   guides/faq

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   autoapi/quadsv/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   changelog

`quadsv <https://github.com/JiayuSuPKU/EquivSVT>`_ is a Python library
for **detecting spatial patterns in omics data**. With it you can
score how much each gene's expression depends on space, find gene
pairs that share a spatial pattern, and compare pattern shapes across
slides, all through a single statistical framework.

The kernel you pass to the test decides what kind of spatial structure
counts. A CAR or Matérn kernel rewards smooth gradients across the
tissue. A graph-Laplacian kernel rewards sharp boundaries between
neighbouring spots. See :doc:`/guides/kernels` for how to pick one.

The library is built for spatial transcriptomics (Visium, Visium HD,
MERFISH, Slide-seq, Xenium, ...) but works with any data that has
spatial or graph structure.


Key features
------------

- **Reliable.** Uses positive-definite kernels, which avoid the
  false negatives that affect Moran's I.
- **Scalable.** Handles millions of spots through sparse solvers
  and FFT / NUFFT acceleration.
- **Flexible.** Accepts arbitrary 2-D coordinates, regular grids,
  and precomputed graphs.
- **Integrated.** Reads :class:`anndata.AnnData` and
  :class:`spatialdata.SpatialData` directly.


Quick example
-------------

.. code-block:: python

   import numpy as np
   from quadsv import NUFFTKernel, spatial_q_test

   # Spatial coordinates and one gene's expression vector
   rng = np.random.default_rng(0)
   coords = rng.uniform(0, 20, size=(500, 2))
   gene = rng.standard_normal(500)

   # Build a Matérn kernel and test the gene for spatial variability
   kernel = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
   Q, pval = spatial_q_test(gene, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

.. dropdown:: What is the Q-statistic?

   The Q-statistic
   :math:`Q = \mathbf{z}^\top \mathbf{K z}` measures how strongly
   a feature's values line up with the spatial structure encoded
   by the kernel ``K``. A large Q means the feature is spatially
   structured in the way ``K`` looks for; a small Q means the
   feature is spatially independent under ``K``. Different kernels
   look for different things: CAR and Matérn pick up smooth,
   large-scale variation, while a graph Laplacian picks up sharp,
   local variation. See :doc:`/guides/theory` for the derivation
   and :doc:`/guides/kernels` for picking a kernel.


Getting started
---------------

- :doc:`/guides/installation`
- :doc:`/guides/quickstart` (a 5-minute tour)
- :doc:`/guides/kernels` (pick a kernel for your data)
- :doc:`/guides/multisample` (compare slides across groups)
- :doc:`/guides/theory` and :doc:`/guides/scaling` (math and
  performance)


Citation
--------

Su, Jiayu, et al.
*On the consistent and scalable detection of spatial patterns.*
`arXiv:2602.02825 (2026) <https://arxiv.org/pdf/2602.02825>`_.


Reporting issues
----------------

Please open a ticket on the
`GitHub Issues page <https://github.com/JiayuSuPKU/EquivSVT/issues>`_.
