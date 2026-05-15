"""
``quadsv.detectors`` — single-sample spatial-pattern detection.

Subpackage grouping the layer-3 public classes:

- :class:`Detector` — abstract base with the shared
  setup-data / compute-Q / compute-R workflow.
- :class:`DetectorIrregular` — wraps :class:`anndata.AnnData` (irregular
  coordinates; matrix and NUFFT backends).
- :class:`DetectorGrid` — wraps :class:`spatialdata.SpatialData` (regular
  rasterized grid; FFT backend).
"""

from quadsv.detectors.base import Detector
from quadsv.detectors.grid import DetectorGrid
from quadsv.detectors.irregular import DetectorIrregular

__all__ = ["Detector", "DetectorIrregular", "DetectorGrid"]
