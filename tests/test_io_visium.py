"""Unit tests for quadsv.utils Visium loader + hex rasterizer."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from quadsv.utils import (
    VISIUM_V1_SPOT_SPACING_UM,
    load_visium_sample,
    visium_hex_spacing_um,
    visium_to_grid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_visium_tree(tmp: Path, ny: int = 8, nx_even: int = 4) -> Path:
    """Create a minimal Space Ranger-style directory at ``tmp``.

    Generates a Visium hex grid of ``ny`` rows × ``nx_even`` spots per row in the
    'orange-crate' layout (array_col = 2·c + row%2). Writes a filtered_feature
    H5 (Space Ranger v2 layout — single group at the root) plus the
    tissue_positions_list.csv and scalefactors_json.json.
    """
    root = tmp / "sample"
    (root / "spatial").mkdir(parents=True)

    # Build spot list in the orange-crate layout.
    spots = []
    for r in range(ny):
        for c in range(nx_even):
            spots.append((f"bc{r:02d}{c:02d}-1", 1, r, 2 * c + (r % 2)))
    barcodes = [s[0] for s in spots]

    # tissue_positions_list.csv (no header, pxl cols 4 and 5 can be dummy).
    rows = [(bc, 1, ar, ac, 100 + 10 * ar, 100 + 10 * ac) for (bc, _in, ar, ac) in spots]
    pd.DataFrame(rows).to_csv(
        root / "spatial" / "tissue_positions_list.csv", header=False, index=False
    )
    # scalefactors.
    (root / "spatial" / "scalefactors_json.json").write_text(
        json.dumps(
            {
                "spot_diameter_fullres": 50.0,
                "tissue_hires_scalef": 0.2,
                "fiducial_diameter_fullres": 75.0,
                "tissue_lowres_scalef": 0.05,
            }
        )
    )

    # filtered matrix: 3 genes, gene expression = row + col (per spot) so we can
    # verify rasterization places each spot in the correct cell.
    n_spots = len(spots)
    genes = ["GeneA", "GeneB", "GeneC"]
    X = np.zeros((n_spots, len(genes)), dtype=np.float32)
    for i, (_bc, _in, r, ac) in enumerate(spots):
        X[i, 0] = r
        X[i, 1] = ac
        X[i, 2] = r * 100 + ac  # unique per spot
    X_csr = sp.csr_matrix(X)

    # Write minimal 10x v3 h5.
    h5_path = root / "filtered_feature_bc_matrix.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        grp.create_dataset("data", data=X_csr.data)
        grp.create_dataset("indices", data=X_csr.indices)
        grp.create_dataset("indptr", data=X_csr.indptr)
        grp.create_dataset("shape", data=np.array([len(genes), n_spots], dtype=np.int64))
        fgrp = grp.create_group("features")
        fgrp.create_dataset("id", data=np.array(genes, dtype="S"))
        fgrp.create_dataset("name", data=np.array(genes, dtype="S"))
        fgrp.create_dataset(
            "feature_type", data=np.array(["Gene Expression"] * len(genes), dtype="S")
        )
        fgrp.create_dataset("genome", data=np.array(["toy"] * len(genes), dtype="S"))

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVisiumHexSpacing:
    def test_dense_spacing_is_hex_correct(self):
        dy, dx = visium_hex_spacing_um(spot_spacing_um=100.0, grid="dense")
        assert dy == pytest.approx(100.0 * np.sqrt(3) / 2)
        assert dx == pytest.approx(50.0)

    def test_collapsed_spacing(self):
        dy, dx = visium_hex_spacing_um(grid="collapsed")
        assert dy == pytest.approx(VISIUM_V1_SPOT_SPACING_UM * np.sqrt(3) / 2)
        assert dx == pytest.approx(100.0)

    def test_invalid_grid_raises(self):
        with pytest.raises(ValueError, match="grid must be"):
            visium_hex_spacing_um(grid="bogus")


class TestLoadVisiumSample:
    def test_loads_tiny_sample(self, tmp_path):
        sample = _write_tiny_visium_tree(tmp_path, ny=6, nx_even=4)
        adata = load_visium_sample(sample)
        # 6 rows * 4 spots = 24 spots, 3 genes.
        assert adata.shape == (24, 3)
        assert {"array_row", "array_col", "in_tissue"} <= set(adata.obs.columns)
        assert "spatial" in adata.obsm
        assert "scalefactors" in adata.uns["spatial"]

    def test_loads_from_outs_layout(self, tmp_path):
        sample = _write_tiny_visium_tree(tmp_path)
        # Move files into an outs/ subdirectory to simulate canonical layout.
        outs = sample / "outs"
        outs.mkdir()
        (sample / "filtered_feature_bc_matrix.h5").rename(outs / "filtered_feature_bc_matrix.h5")
        (sample / "spatial").rename(outs / "spatial")
        adata = load_visium_sample(sample)
        assert adata.n_obs > 0
        assert adata.uns["spatial"]["path"].endswith("outs")


class TestVisiumToGrid:
    def _mk_adata(self, ny: int, nx_even: int):
        """Build a tiny in-memory AnnData without touching disk."""
        spots = [(r, 2 * c + (r % 2)) for r in range(ny) for c in range(nx_even)]
        n_spots = len(spots)
        obs = pd.DataFrame(
            {
                "array_row": [s[0] for s in spots],
                "array_col": [s[1] for s in spots],
                "in_tissue": 1,
            },
            index=[f"bc{i}" for i in range(n_spots)],
        )
        var = pd.DataFrame(index=["GeneA", "GeneB"])
        X = np.zeros((n_spots, 2), dtype=np.float64)
        for i, (r, c) in enumerate(spots):
            X[i, 0] = r
            X[i, 1] = c
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_dense_rasterization_places_every_spot(self):
        adata = self._mk_adata(ny=6, nx_even=4)
        grid, spacing = visium_to_grid(adata, grid="dense", fill="zero")
        # Output shape follows the max(array_row)+1, max(array_col)+1 defaults.
        # For odd rows, col max = 2*3 + 1 = 7 -> nx = 8 indices [0..7].
        assert grid.shape == (2, 6, 8)
        # Verify GeneA == array_row at all real-spot locations.
        for r in range(adata.obs["array_row"].max() + 1):
            for ac in range(adata.obs["array_col"].max() + 1):
                if (ac % 2) == (r % 2):  # real spot
                    assert grid[0, r, ac] == r
                    assert grid[1, r, ac] == ac
        assert spacing == pytest.approx(visium_hex_spacing_um(grid="dense"))

    def test_nearest_fill_matches_neighbors(self):
        """Empty hex cells should be the mean of their two in-row hex neighbors."""
        adata = self._mk_adata(ny=4, nx_even=4)
        grid, _ = visium_to_grid(adata, grid="dense", fill="nearest")
        # For (r=0, c=1) (empty on even row), neighbors are (0, 0) = 0 and (0, 2) = 2.
        # GeneB stores array_col => expected fill = (0 + 2) / 2 = 1.
        assert grid[1, 0, 1] == pytest.approx(1.0)
        # Symmetric: (1, 0) empty on odd row: neighbors (1,-1) invalid, (1, 1) = 1.
        # Left edge gets one-sided neighbor.
        assert grid[1, 1, 0] == pytest.approx(1.0)

    def test_collapsed_rasterization(self):
        adata = self._mk_adata(ny=4, nx_even=3)
        grid, spacing = visium_to_grid(adata, grid="collapsed")
        assert grid.shape == (2, 4, 3)
        assert spacing == pytest.approx(visium_hex_spacing_um(grid="collapsed"))

    def test_missing_obs_columns_raises(self):
        adata = ad.AnnData(X=np.zeros((3, 2)))
        with pytest.raises(KeyError, match="array_row"):
            visium_to_grid(adata)

    def test_explicit_gene_selection(self):
        adata = self._mk_adata(ny=4, nx_even=3)
        grid, _ = visium_to_grid(adata, genes=["GeneB"])
        assert grid.shape[0] == 1

    def test_invalid_grid_mode_raises(self):
        adata = self._mk_adata(ny=4, nx_even=3)
        with pytest.raises(ValueError, match="grid must"):
            visium_to_grid(adata, grid="bogus")
