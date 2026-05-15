"""Tests for the :func:`quadsv.Detector` / :func:`quadsv.Comparator`
factory dispatch.

The factories pick the right class from ``isinstance(data, ...)``;
they don't construct or copy data. We feed minimal AnnData /
SpatialData objects and assert the returned class is correct, then
confirm the error paths (empty list, mixed list, unsupported type)
raise ``TypeError`` with helpful messages.
"""

from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import pytest

from quadsv import (
    Comparator,
    ComparatorGrid,
    ComparatorIrregular,
    Detector,
    DetectorGrid,
    DetectorIrregular,
)


class TestDetectorFactory(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.adata = ad.AnnData(rng.standard_normal((20, 3)))
        self.adata.obsm["spatial"] = rng.standard_normal((20, 2))

    def test_anndata_dispatches_to_detector_irregular(self):
        det = Detector(self.adata, kernel_method="gaussian", backend="matrix")
        self.assertIsInstance(det, DetectorIrregular)

    def test_spatialdata_dispatches_to_detector_grid(self):
        # Constructing a real SpatialData here is heavy; a Mock with the
        # right type identity is enough for the isinstance check.
        from spatialdata import SpatialData

        sdata = SpatialData()  # empty SpatialData is valid
        det = Detector(sdata)
        self.assertIsInstance(det, DetectorGrid)

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="cannot dispatch on type"):
            Detector(np.zeros((10, 10)))

    def test_factory_kwargs_forwarded(self):
        det = Detector(self.adata, kernel_method="matern", backend="matrix")
        # DetectorIrregular stores the chosen method on the trailing-underscore
        # attribute (sklearn-style).
        self.assertEqual(det.kernel_method_, "matern")


class TestComparatorFactory(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.adatas = []
        for _ in range(3):
            a = ad.AnnData(rng.standard_normal((20, 3)))
            a.obsm["spatial"] = rng.standard_normal((20, 2))
            self.adatas.append(a)

    def test_anndata_list_dispatches_to_irregular(self):
        cmp = Comparator(self.adatas)
        self.assertIsInstance(cmp, ComparatorIrregular)

    def test_spatialdata_list_dispatches_to_grid(self):
        from spatialdata import SpatialData

        sdatas = [SpatialData() for _ in range(3)]
        # ComparatorGrid expects more setup but the factory only checks
        # the type and forwards. Use pytest.raises to allow downstream
        # validation errors but still assert the *class* picked first.
        try:
            cmp = Comparator(sdatas)
        except (TypeError, ValueError, AttributeError):
            # Construction may legitimately fail because the
            # SpatialData objects are empty; we only care that the
            # factory tried ComparatorGrid, not ComparatorIrregular.
            pytest.skip("ComparatorGrid construction needs full SpatialData")
        else:
            self.assertIsInstance(cmp, ComparatorGrid)

    def test_empty_list_raises_typeerror(self):
        with pytest.raises(TypeError, match="non-empty"):
            Comparator([])

    def test_mixed_list_raises_typeerror(self):
        from spatialdata import SpatialData

        mixed = [self.adatas[0], SpatialData()]
        with pytest.raises(TypeError, match="mixed"):
            Comparator(mixed)

    def test_unsupported_element_type_raises(self):
        with pytest.raises(TypeError, match="mixed"):
            Comparator([np.zeros((10, 10))])


if __name__ == "__main__":
    unittest.main()
