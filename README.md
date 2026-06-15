# quadsv: Consistent and scalable spatial pattern detection and comparison

Detect spatial patterns in omics data via kernel-based hypothesis tests for spatial variability (*Q-tests*) and co-expression (*R-tests*), then compare spatial pattern spectra across samples or conditions.

**Key features:**
- **Reliable**: CAR kernel eliminates false negatives from Moran's I spectral cancellation  
- **Scalable**: Implicit sparse solvers and FFT acceleration handle millions of spots  
- **Universal**: Works with Visium, Visium HD, MERFISH, lineage trees, any spatial/graph data  
- **Integrated**: Native `AnnData` and `SpatialData` support
- **Comparative**: Alignment-free cross-sample pattern comparison with FFT/NUFFT spectra

## Installation

```bash
pip install quadsv  # From PyPI
# OR (latest dev version)
pip install git+https://github.com/JiayuSuPKU/quadsv.git#egg=quadsv
# OR (for development)
git clone https://github.com/JiayuSuPKU/quadsv.git && cd quadsv && pip install -e .
```

## Usage

### Q-test: Single gene spatial variability

```python
import numpy as np
from quadsv import MatrixKernel, spatial_q_test

# simulate coordinates and gene expression
coords = np.random.randn(500, 2)
gene_expression = np.random.randn(500)

# build CAR kernel and run Q-test
kernel = MatrixKernel.from_coordinates(coords, method='car', k_neighbors=15, rho=0.9)
Q, pval = spatial_q_test(gene_expression, kernel)
print(f"Q-statistic: {Q:.4f}, p-value: {pval:.4e}")
```

### R-test: Spatial co-expression

```python
from quadsv import spatial_r_test

# run R-test with the same kernel
gene1, gene2 = np.random.randn(500), np.random.randn(500)
R, pval = spatial_r_test(gene1, gene2, kernel)
print(f"R-statistic: {R:.4f}, p-value: {pval:.4e}")
```

### FFT-accelerated tests (for regular grids like Visium HD)

```python
import numpy as np
from quadsv import FFTKernel, spatial_q_test, spatial_r_test

# For grid data (e.g., 1000x1000 Visium HD)
kernel_fft = FFTKernel(shape=(1000, 1000), method='car', rho=0.9)

# simulate gene expression on grid
gene_grid = np.random.randn(1000, 1000)

# run FFT-based Q-test
Q_fft, pval_fft = spatial_q_test(gene_grid, kernel_fft)
print(f"FFT Q-test: Q={Q_fft:.4f}, p-value={pval_fft:.4e}")

# run FFT-based R-test
gene1_grid = np.random.randn(1000, 1000)
gene2_grid = np.random.randn(1000, 1000)
R_fft, pval_fft = spatial_r_test(gene1_grid, gene2_grid, kernel_fft)
print(f"FFT R-test: R={R_fft:.4f}, p-value={pval_fft:.4e}")
```

### Tutorials

#### Detect SVG and spatial co-expression using AnnData

```python
import anndata as ad
from quadsv import Detector

adata = ad.read_h5ad("spatial_data.h5ad")
detector = Detector(
    adata,
    kernel_method='car',
    backend='matrix',
    rho=0.9,
    k_neighbors=4,
).setup_data(adata, obsm_key='spatial', min_cells_frac=0.05)

# Compute Q-statistics and p-values genome-wide
q_results_df = detector.compute_qstat(source='var', features=None, return_pval=True)

# Returns DataFrame with columns: [Q, P_value, P_adj, Z_score]
significant_genes = q_results_df[q_results_df['P_adj'] < 0.05]
print(f"Found {len(significant_genes)} spatially variable genes")

# Select top 1000 SVGs for pairwise R-test
top_genes = significant_genes.sort_values('Q', ascending=False).head(1000).index.tolist()
r_results_df = detector.compute_rstat(
    source='var',
    features_x=top_genes,
    features_y=None,
    return_pval=True,
)
# Returns DataFrame with columns: [Feature_1, Feature_2, R, P_value, P_adj, Z_score]
significant_pairs = r_results_df[r_results_df['P_adj'] < 0.05]
print(f"Found {len(significant_pairs)} spatially co-expressed gene pairs")
```

#### FFT-based Q-test for large grids

```python
import spatialdata as sd
from quadsv import Detector

sdata = sd.read_zarr("visium_hd.zarr/")
detector = Detector(sdata, kernel_method='car', rho=0.9, topology='square').setup_data(
    sdata,
    bins='Visium_HD_bin',
    table_name='table',
    col_key='array_col',
    row_key='array_row',
)
results = detector.compute_qstat(
    n_jobs=4, workers=2, return_pval=True
)
```

#### Cross-sample spatial pattern comparison

```python
import anndata as ad
import numpy as np
from quadsv import Comparator

sample_paths = [
    "control_1.h5ad",
    "control_2.h5ad",
    "control_3.h5ad",
    "case_1.h5ad",
    "case_2.h5ad",
    "case_3.h5ad",
]
samples = [ad.read_h5ad(path) for path in sample_paths]
groups = np.array([0, 0, 0, 1, 1, 1])  # 1-D labels: control vs case

cmp = (
    Comparator(samples)
    .compute_spectra(n_jobs=4)
    .normalize_background()
)

# Pattern-level difference: compares radial power spectra, not total expression.
pattern_hits = cmp.test_diff_freq(groups, statistic='log_l2', normalize_shape=True)
# Companion expression-level difference on the DC component.
expression_hits = cmp.test_diff_expr(groups)
```

`pattern_hits` returns per-gene columns `[Feature, Statistic, P_value, P_adj]`. Use `ComparatorGrid` or the same `Comparator(...)` factory with `SpatialData` samples for regular rasterized grids.

**Full tutorials:** [See docs/guides/quickstart.rst](docs/guides/quickstart.rst), [docs/guides/multisample.rst](docs/guides/multisample.rst), and test suite examples.

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

### Troubleshooting
Dependencies may be installed in a way that causes conflicts or unexpected behavior. 
To confirm whether a failure is environment-specific, validate from a clean
conda environment:

```bash
conda create -n quadsv-test -c conda-forge python=3.12 pip -y
conda activate quadsv-test
python -m pip install -e ".[dev]"
python -m pytest -q
```

There are several known issues that may arise due to problematic environment configurations:

#### Numba cache error during import
This happens before any tests run, often through `spatialdata -> xrspatial`, where helpers are 
decorated with `numba.njit(cache=True)`. The typical error is:

```text
RuntimeError: cannot cache function ... no locator available
```

To fix this, either point Numba at a writable cache directory via

```bash
mkdir -p /private/tmp/numba-cache
NUMBA_CACHE_DIR=/private/tmp/numba-cache python -c 'import quadsv'
```

 or disable JIT to bypass the import-time cache path via

```bash
NUMBA_DISABLE_JIT=1 python -m pytest -q
```

#### Segfault during irregular NUFFT tests
On macOS arm64, having multiple OpenMP runtimes in one environment can cause segfaults. 
For example, when running `finufft.nufft2d1 -> Plan.setpts`, conda OpenBLAS may load
 `$CONDA_PREFIX/lib/libomp.dylib` while the PyPI FINUFFT wheel may load
`finufft/.dylibs/libomp.dylib`. 

To fix this, either set the environment variable `KMP_DUPLICATE_LIB_OK=True`, which risks 
introducing multithreading instability; or alternatively, use a single OpenMP runtime via

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

Longer-term, prefer a consistent conda-forge native stack so FINUFFT, OpenBLAS,
NumPy/SciPy, and scikit-learn do not bring separate vendored OpenMP runtimes.

### Documentation
[ReadTheDocs](https://quadsv.readthedocs.io/)

## References

Su, Jiayu, et al. "On the consistent and scalable detection of spatial patterns." arXiv (2026): 2602.02825. [link to preprint](https://arxiv.org/abs/2602.02825)


## License & Support

- **License:** BSD-3-Clause - see [LICENSE](LICENSE)  
- **Issues:** [GitHub Issues](https://github.com/JiayuSuPKU/quadsv/issues)
- **Docs:** [ReadTheDocs](https://quadsv.readthedocs.io/)
