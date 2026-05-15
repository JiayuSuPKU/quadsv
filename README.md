# quadsv: Consistent and scalable spatial pattern detection

Detect spatial patterns in omics data via kernel-based hypothesis tests for spatial variability (*Q-tests*) and for co-expression (*R-tests*).

**Key features:**
- **Reliable**: CAR kernel eliminates false negatives from Moran's I spectral cancellation  
- **Scalable**: Implicit sparse solvers and FFT acceleration handle millions of spots  
- **Universal**: Works with Visium, Visium HD, MERFISH, lineage trees, any spatial/graph data  
- **Integrated**: Native `AnnData` and `SpatialData` support

## Installation

```bash
pip install quadsv  # From PyPI
# OR (latest dev version)
pip install git+https://github.com/JiayuSuPKU/EquivSVT.git#egg=quadsv
# OR (for development)
git clone https://github.com/JiayuSuPKU/EquivSVT.git && cd EquivSVT && pip install -e .
```

## Usage

### Q-test: Single gene spatial variability

```python
import numpy as np
from quadsv.kernels import SpatialKernel
from quadsv.statistics import spatial_q_test

# simulate coordinates and gene expression
coords = np.random.randn(500, 2)
gene_expression = np.random.randn(500)

# build CAR kernel and run Q-test
kernel = SpatialKernel.from_coordinates(coords, method='car', k_neighbors=15, rho=0.9)
Q, pval = spatial_q_test(gene_expression, kernel)
print(f"Q-statistic: {Q:.4f}, p-value: {pval:.4e}")
```

### R-test: Spatial co-expression

```python
from quadsv.statistics import spatial_r_test

# run R-test with the same kernel
gene1, gene2 = np.random.randn(500), np.random.randn(500)
R, pval = spatial_r_test(gene1, gene2, kernel)
print(f"R-statistic: {R:.4f}, p-value: {pval:.4e}")
```

### FFT-accelerated tests (for regular grids like Visium HD)

```python
import numpy as np
from quadsv.fft import FFTKernel, spatial_q_test_fft, spatial_r_test_fft

# For grid data (e.g., 1000x1000 Visium HD)
kernel_fft = FFTKernel(shape=(1000, 1000), method='car', rho=0.9)

# simulate gene expression on grid
gene_grid = np.random.randn(1000, 1000)

# run FFT-based Q-test
Q_fft, pval_fft = spatial_q_test_fft(gene_grid, kernel_fft)
print(f"FFT Q-test: Q={Q_fft:.4f}, p-value={pval_fft:.4e}")

# run FFT-based R-test
gene1_grid = np.random.randn(1000, 1000)
gene2_grid = np.random.randn(1000, 1000)
R_fft, pval_fft = spatial_r_test_fft(gene1_grid, gene2_grid, kernel_fft)
print(f"FFT R-test: R={R_fft:.4f}, p-value={pval_fft:.4e}")
```

### Tutorials

#### Detect SVG and spatial co-expression using AnnData

```python
import anndata as ad
from quadsv.detector import PatternDetector

adata = ad.read_h5ad("spatial_data.h5ad")
detector = PatternDetector(adata, min_cells_frac=0.05)

# build kernel from spatial coordinates
detector.build_kernel_from_coordinates(adata.obsm['spatial'], method='car', rho=0.9, k_neighbors=4)

# alternatively, from a precomputed graph adjacency
adata.obsp['W'] = ...  # Precomputed graph adjacency matrix (usually sparse)
detector.build_kernel_from_obsp(key='W', is_distance=False, method='car', rho=0.9)

# Compute Q-statistics and p-values genome-wide
q_results_df = detector.compute_qstat(source='var', features = None, return_pval = True)

# Returns DataFrame with columns: [Q, P_value, P_adj, Z_score]
significant_genes = q_results_df[q_results_df['P_adj'] < 0.05]
print(f"Found {len(significant_genes)} spatially variable genes")

# Select top 1000 SVGs for pairwise R-test
top_genes = significant_genes.sort_values('Q', ascending=False).head(1000).index.tolist()
r_results_df = detector.compute_rstat(source='var', features_x=top_genes, features_y=None, return_pval=True)
# Returns DataFrame with columns: [Gene_X, Gene_Y, R, P_value, P_adj, Z_score]
significant_pairs = r_results_df[r_results_df['P_adj'] < 0.05]
print(f"Found {len(significant_pairs)} spatially co-expressed gene pairs")
```

#### FFT-based Q-test for large grids

```python
import spatialdata as sd
from quadsv.detector_fft import PatternDetectorFFT

sdata = sd.read_zarr("visium_hd.zarr/")
detector = PatternDetectorFFT(sdata, kernel_method='car', rho=0.9, topology='square')

results = detector.compute_qstat(
    bins=['Visium_HD_bin'],
    table_name=['table'],
    n_jobs=4, workers=2, return_pval=True
)
```

**Full tutorials:** [See docs/quickstart.rst](docs/quickstart.rst) and test suite examples.

## Kernel Methods

| Method | Type | Spectrum | Parameters | Use Case |
|--------|------|----------|------------|----------|
| `gaussian` | Distance | Positive Definite | `bandwidth` | Isotropic, exponential decay |
| `matern` | Distance | Positive Definite | `bandwidth`, `nu` | Tunable smoothness ✓ **Recommended** |
| **`moran`** ⚠️ | Graph | **Indefinite** | `k_neighbors` | Autocorrelation (false negatives) |
| `laplacian` | Graph | Semi-Definite | `k_neighbors` | High-frequency filter |
| **`car`** ✓ | Graph | **Strictly Positive** | `rho`, `k_neighbors` | CAR kernel ✓ **Recommended** |

**Recommendation:** Use `car` (CAR kernel) for robust, consistent detection across all functional patterns.

## Testing & Development

```bash
pytest tests/ --cov=quadsv              # Run all tests
pytest tests/test_tutorials.py -v       # Run tutorial examples
pip install -e ".[dev,docs]"            # Install dev + docs dependencies
```

**Documentation:** [ReadTheDocs](https://equivsvt.readthedocs.io/)

## References

Su, Jiayu, et al. "On the consistent and scalable detection of spatial patterns." arXiv (2026): 2602.02825. [link to preprint](https://arxiv.org/abs/2602.02825)


## License & Support

- **License:** BSD-3-Clause - see [LICENSE](LICENSE)  
- **Issues:** [GitHub Issues](https://github.com/JiayuSuPKU/EquivSVT/issues)  
- **Docs:** [ReadTheDocs](https://equivsvt.readthedocs.io/)