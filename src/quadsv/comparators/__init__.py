"""
``quadsv.comparators`` — cross-sample spatial-pattern comparison.

Subpackage grouping the layer-4 public classes:

- :class:`ComparatorIrregular` — wraps a sequence of
  :class:`anndata.AnnData` (irregular spots, NUFFT backend).
- :class:`ComparatorGrid` — wraps a sequence of
  :class:`spatialdata.SpatialData` (regular rasterized bins, FFT
  backend).

Both classes share the same post-``compute_spectra`` surface
(``normalize_background``, ``normalize_covariates``,
``test_diff_freq``, ``test_diff_expr``, ``effective_rank``) through
the private :class:`~quadsv.comparators.base._ComparatorBase` mixin.
Cross-sample contrasts are supplied at test time via the ``design``
argument on the test methods — the comparator itself is
design-agnostic, so one fitted comparator can serve any number of
unrelated contrasts on the same spectra. The shape-only frequency
test is available via a ``normalize_shape: bool = False`` keyword on
:meth:`test_diff_freq` (forwarded to the standalone ``compare_*``
function).

The array-level spectral feature helpers live in
:mod:`quadsv.comparators.features`; normalization primitives live in
:mod:`quadsv.comparators.normalization`; statistical comparison
primitives live in :mod:`quadsv.comparators.multisample`.
"""

from quadsv.comparators.grid import ComparatorGrid
from quadsv.comparators.irregular import ComparatorIrregular

__all__ = ["ComparatorIrregular", "ComparatorGrid"]
