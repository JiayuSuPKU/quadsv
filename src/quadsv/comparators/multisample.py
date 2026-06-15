"""Statistical comparison primitives for comparator outputs.

Spectral feature construction lives in :mod:`quadsv.comparators.features`.
Spectrum normalization lives in :mod:`quadsv.comparators.normalization`.
This module consumes normalized or raw per-sample arrays and provides:

- ``compare_two_groups`` and ``compare_two_groups_masked`` for binary labels.
- ``compare_glm`` and ``compare_glm_masked`` for design-matrix contrasts.
- Scalar DC-expression companions ``compare_two_groups_scalar`` and
  ``compare_glm_scalar`` with analytic t-distribution tests.

The public comparator classes in :mod:`quadsv.comparators` wrap these
array-level functions for AnnData and SpatialData inputs.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401  (exposed for downstream calibration tests)
from scipy.stats import t as _t_dist

from quadsv.comparators.normalization import _normalize_shape_apply
from quadsv.statistics import apply_bh_correction, cauchy_combine, liu_sf

__all__ = [
    "compare_two_groups",
    "compare_two_groups_masked",
    "compare_two_groups_scalar",
    "compare_glm",
    "compare_glm_masked",
    "compare_glm_scalar",
]

logger = logging.getLogger(__name__)

_AVAILABLE_STATISTICS = ("log_l2", "welch_t_cauchy")
_NULL_OPTIONS = ("permutation", "analytic")


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


def _resolve_freq_weights(freq_weights: np.ndarray | None, n_bins: int) -> np.ndarray:
    """Validate / normalize frequency-bin weights; return a length-``n_bins`` array.

    Passing None yields uniform weights ``1/n_bins`` - recovering the unweighted
    statistic. Any other input is cast to ``float``, required to be
    non-negative and not all-zero, and rescaled to sum-1. Non-uniform
    weights are how users express a kernel-like frequency preference (e.g.,
    low-pass polynomial vs exponential decay) inside the spectral distance.
    """
    if freq_weights is None:
        return np.full(n_bins, 1.0 / n_bins)
    w = np.asarray(freq_weights, dtype=float).ravel()
    if w.shape != (n_bins,):
        raise ValueError(f"freq_weights must have length n_bins={n_bins}, got shape {w.shape}.")
    if np.any(w < 0):
        raise ValueError("freq_weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("freq_weights must not sum to zero.")
    return w / total


def _stat_log_l2(
    group_a: np.ndarray,
    group_b: np.ndarray,
    freq_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted L2 distance between mean log-spectra. Vectorized over genes.

    The (default) uniform-weight case reduces to the plain L2 distance on
    ``n_bins`` frequency bins - up to an overall ``1/sqrt(n_bins)`` scale that is
    irrelevant under a permutation null. Non-uniform weights (which must be
    non-negative and sum to 1) let the user emphasize low or high
    frequencies the same way a kernel spectrum does (polynomial vs
    exponential decay, etc.).
    """
    eps = 1e-12
    log_a = np.log(np.maximum(group_a, eps)).mean(axis=0)  # (n_genes, n_bins)
    log_b = np.log(np.maximum(group_b, eps)).mean(axis=0)
    diff = log_a - log_b  # (n_genes, n_bins)
    n_bins = diff.shape[-1]
    w = _resolve_freq_weights(freq_weights, n_bins)
    return np.sqrt(np.sum(w * diff**2, axis=-1))


def _welch_test(group_a: np.ndarray, group_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed Welch t-statistic and analytic two-sided p-value along axis 0.

    Works for any trailing feature shape: ``(n_samples, n_features)`` gives a
    ``(n_features,)`` result for scalar DE, while ``(n_samples, n_genes, n_bins)``
    gives ``(n_genes, n_bins)`` per-frequency-bin statistics. The p-values use the
    Welch-Satterthwaite degrees of freedom from the t-distribution tail.
    """
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se2_a = var_a / max(n_a, 1)
    se2_b = var_b / max(n_b, 1)
    se2 = se2_a + se2_b + 1e-30
    t_stat = (mean_a - mean_b) / np.sqrt(se2)
    if n_a > 1 and n_b > 1:
        df = (se2**2) / ((se2_a**2) / max(n_a - 1, 1) + (se2_b**2) / max(n_b - 1, 1) + 1e-30)
    else:
        df = np.full_like(mean_a, float(max(n_a + n_b - 2, 1)))
    df = np.maximum(df, 1.0)
    pvals = 2.0 * _t_dist.sf(np.abs(t_stat), df)
    # Clip the floor to the smallest representable positive float so
    # Cauchy's tan(pi(0.5 - p)) stays finite.
    return t_stat, np.clip(pvals, np.finfo(float).tiny, 1.0)


# ---------------------------------------------------------------------------
# Permutation engine
# ---------------------------------------------------------------------------


def _exchangeable_group_labels(
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    max_exact_permutations: int = 10000,
) -> tuple[np.ndarray, bool]:
    """Build a null-distribution set of group relabellings.

    For small samples the total number of distinct two-group label
    assignments (``C(n, n_a)``) can be tiny compared to the user's
    requested ``n_perm``, which means the permutation p-value is floored
    at ``1/(C(n, n_a) + 1)``. In that regime an **exact** enumeration
    of every possible relabelling is both cheaper and strictly more
    accurate (zero Monte-Carlo noise, sharp p-values).

    Parameters
    ----------
    groups : np.ndarray
        Observed group labels, length ``n_samples`` with exactly two
        unique values.
    n_perm : int
        Number of random shuffles to produce when exact enumeration is
        infeasible. Ignored on the exact path.
    rng : np.random.Generator
        RNG for the sampling fallback.
    max_exact_permutations : int, default 10000
        If ``C(n_samples, n_a)`` is at most this, every distinct relabelling
        is enumerated (``is_exact=True``) and ``n_perm`` is overridden to
        the enumeration count. Otherwise ``n_perm`` random shuffles of
        ``groups`` are returned (``is_exact=False``).

    Returns
    -------
    perm_labels : np.ndarray
        ``(n_used, n_samples)`` array; each row is a valid relabelling
        (same ``n_a`` / ``n_b`` marginals as ``groups``).
    is_exact : bool
        True if every row is a distinct relabelling and together they
        span every possible partition; False if the rows are independent
        random shuffles.
    """
    groups = np.asarray(groups)
    n_samples = len(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if uniq.size != 2:
        raise ValueError(f"groups must have exactly two unique values, got {uniq}.")
    n_a = int(counts[0])
    total = int(math.comb(n_samples, n_a))
    if total <= max_exact_permutations:
        perm_labels = np.empty((total, n_samples), dtype=groups.dtype)
        a_val, b_val = uniq[0], uniq[1]
        for i, subset in enumerate(itertools.combinations(range(n_samples), n_a)):
            perm_labels[i] = b_val
            perm_labels[i, list(subset)] = a_val
        return perm_labels, True
    perm_labels = np.empty((n_perm, n_samples), dtype=groups.dtype)
    base = groups.copy()
    for i in range(n_perm):
        rng.shuffle(base)
        perm_labels[i] = base
    return perm_labels, False


def _permutation_pvalue(
    observed: np.ndarray,
    null_samples: np.ndarray,
) -> np.ndarray:
    """One-sided ``Pr(null >= observed)`` with an additive ``+1`` correction."""
    n_perm = null_samples.shape[0]
    ge = (null_samples >= observed[None, :]).sum(axis=0)
    return (ge + 1.0) / (n_perm + 1.0)


def _run_statistic_with_perm(
    stat_name: str,
    spectra: np.ndarray,
    group_codes: np.ndarray,
    perm_labels: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed statistic + null distribution for one statistic. Internal.

    ``perm_labels`` is a ``(n_perm_used, n_samples)`` matrix of group
    relabellings (as produced by :func:`_exchangeable_group_labels`).

    ``freq_weights`` is forwarded only to statistics that accept it (currently
    ``log_l2``); other statistics ignore it.
    """
    if stat_name != "log_l2":
        raise ValueError(f"Permutation statistic must be 'log_l2', got {stat_name!r}.")
    uniq = np.unique(group_codes)
    a_val = uniq[0]
    a_mask = group_codes == a_val

    observed = _stat_log_l2(spectra[a_mask], spectra[~a_mask], freq_weights=freq_weights)
    n_perm = perm_labels.shape[0]
    null = np.empty((n_perm, spectra.shape[1]))
    for p in range(n_perm):
        a = perm_labels[p] == a_val
        null[p] = _stat_log_l2(spectra[a], spectra[~a], freq_weights=freq_weights)
    return observed, null


# ---------------------------------------------------------------------------
# Analytic null for welch t and log_l2 (mixture-χ² tail)
# ---------------------------------------------------------------------------


def _run_welch_t_cauchy_analytic(
    spectra: np.ndarray,
    group_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin Welch t test + Cauchy-combined gene-level p-value.

    Both the per-bin significance and the gene-level combination are
    **analytic**: per-bin p-values come from the Welch t-distribution
    and the gene-level p comes from the Cauchy combination.

    Returns
    -------
    observed_abs_t : np.ndarray
        ``(n_genes, n_bins)`` observed per-bin ``|t|`` — used as the reported
        statistic summary (the max across bins sorts the output table
        sensibly, same convention as before).
    combined_pvals : np.ndarray
        ``(n_genes,)`` Cauchy-combined gene-level p-values built from per-bin
        analytic Welch p-values.
    per_bin_pvals : np.ndarray
        ``(n_genes, n_bins)`` per-bin analytic Welch two-sided p-values.
    """
    a_mask = group_codes == 0
    t_stat, per_bin_pvals = _welch_test(spectra[a_mask], spectra[~a_mask])
    abs_t = np.abs(t_stat)
    combined = cauchy_combine(per_bin_pvals, axis=-1)
    return abs_t, combined, per_bin_pvals


# Minimum residual df below which we issue a calibration warning for the
# analytic path. At df=1 the σ̂² estimator has a 100% relative noise
# (var = 2σ⁴), at df=2 it's 50%; both can produce occasional
# anti-conservative spikes.
_ANALYTIC_MIN_DF_NO_WARN = 3


def _maybe_warn_small_df_analytic(df_resid: int) -> None:
    """Warn the user when running the analytic path at very small residual df.

    Suppress with ``warnings.filterwarnings('ignore',
    message="log_l2 + null='analytic'")`` if you accept the calibration risk.
    """
    if df_resid < _ANALYTIC_MIN_DF_NO_WARN:
        rel_noise_pct = 100.0 * (2.0 / max(df_resid, 1)) ** 0.5
        warnings.warn(
            f"log_l2 + null='analytic' at residual df={df_resid}: "
            f"Variance estimator σ̂² has ~{rel_noise_pct:.0f}% relative noise, "
            f"so the analytic null may be anti-conservative on a per-test basis. "
            f"For n_a + n_b ≤ 4, prefer statistic='welch_t_cauchy' for stricter "
            f"calibration (at the cost of some sensitivity).",
            UserWarning,
            stacklevel=3,
        )


def _log_l2_analytic_pvalues(
    statistic: np.ndarray,
    lambs: np.ndarray,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """Analytic p-values for ``log_l2`` via Liu's mixture-χ² tail.

    ``statistic`` is the ``(n_genes,)`` per-gene statistic returned by
    :func:`_stat_log_l2` (square root of the quadratic form ``D'WD``).
    Squaring it here gives the H₀ statistic distributed as ``Σ_k λ_k χ²_1``,
    which Liu's approximation handles directly.
    """
    lambs_safe = np.maximum(np.asarray(lambs, dtype=float), eps)
    statistic_sq = np.asarray(statistic, dtype=float) ** 2
    return np.asarray(liu_sf(statistic_sq, lambs_safe), dtype=float)


def _comparison_frame(
    n_genes: int,
    gene_names: Sequence[str] | None,
    observed: np.ndarray,
    pvals: np.ndarray,
    *,
    extra: dict[str, Sequence[Any] | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build the common comparison result table and apply BH correction."""
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": np.asarray(observed, dtype=float),
            "P_value": np.asarray(pvals, dtype=float),
        }
    )
    if extra:
        for key, value in extra.items():
            df[key] = value
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generalized GLM analytic path for log_l2 (design matrix + contrast)
# ---------------------------------------------------------------------------


def _build_design_matrix(
    design: pd.DataFrame | np.ndarray, n_samples: int
) -> tuple[np.ndarray, list[str]]:
    """Convert ``design`` to a ``(n_samples, p)`` numeric matrix + column labels.

    - ``np.ndarray`` of shape ``(n_samples, p)`` is accepted as-is; columns
      are labelled ``x0, x1, ...``. The caller is responsible for including
      an intercept column if desired.
    - ``pd.DataFrame``: encoded via :func:`patsy.dmatrix` with the formula
      ``~ <col1> + <col2> + ...``, which adds an intercept and one-hot
      encodes categoricals (Treatment contrast against the first level).
      If patsy is not installed, raise ``ImportError`` with an install hint.
    """
    if isinstance(design, np.ndarray):
        design_matrix = np.asarray(design, dtype=float)
        if design_matrix.ndim != 2 or design_matrix.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), "
                f"got {design_matrix.shape}."
            )
        return design_matrix, [f"x{i}" for i in range(design_matrix.shape[1])]

    if not isinstance(design, pd.DataFrame):
        raise TypeError(
            f"design must be a numpy ndarray or pandas DataFrame, " f"got {type(design).__name__}."
        )
    if len(design) != n_samples:
        raise ValueError(
            f"design DataFrame length {len(design)} does not match " f"n_samples={n_samples}."
        )
    try:
        import patsy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Building a design matrix from a pandas DataFrame requires patsy. "
            "Install via `pip install patsy` or pass a pre-built numpy "
            "design matrix instead."
        ) from e
    formula = "~ " + " + ".join(str(c) for c in design.columns)
    design_matrix = patsy.dmatrix(formula, design, return_type="dataframe")
    return design_matrix.to_numpy().astype(float), list(design_matrix.columns)


def _resolve_contrast(
    contrast: str | dict[str, float] | np.ndarray, design_columns: Sequence[str]
) -> np.ndarray:
    """Map the user-supplied contrast spec to a length-``p`` numeric vector.

    - ``str``: column name in the design matrix. Auto-resolves patsy
      treatment-coded factors (e.g., ``"genotype"`` → ``"genotype[T.TG]"``)
      when there is exactly one matching column. For multi-level factors
      with >1 matching column this raises ``ValueError`` (multi-DOF
      contrasts are out of scope; pass an explicit dict or ndarray).
    - ``dict``: maps column-name → coefficient; missing columns get 0.
    - ``ndarray`` of shape ``(p,)``: used as-is.
    """
    n_terms = len(design_columns)
    if isinstance(contrast, np.ndarray):
        contrast_vector = np.asarray(contrast, dtype=float)
        if contrast_vector.shape != (n_terms,):
            raise ValueError(
                f"contrast ndarray must have length p={n_terms}, got shape {contrast_vector.shape}."
            )
        return contrast_vector
    if isinstance(contrast, str):
        if contrast in design_columns:
            target = contrast
        else:
            matches = [col for col in design_columns if col.startswith(contrast + "[T.")]
            if not matches:
                raise ValueError(
                    f"Contrast '{contrast}' not found in design columns " f"{list(design_columns)}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Contrast '{contrast}' is ambiguous — matches "
                    f"{matches}. Pass an explicit dict or ndarray (multi-DOF "
                    f"contrasts are out of scope)."
                )
            target = matches[0]
        contrast_vector = np.zeros(n_terms, dtype=float)
        contrast_vector[list(design_columns).index(target)] = 1.0
        return contrast_vector
    if isinstance(contrast, dict):
        contrast_vector = np.zeros(n_terms, dtype=float)
        for k, v in contrast.items():
            if k not in design_columns:
                raise ValueError(
                    f"Contrast key '{k}' not in design columns " f"{list(design_columns)}."
                )
            contrast_vector[list(design_columns).index(k)] = float(v)
        return contrast_vector
    raise TypeError(f"Contrast must be a str, dict, or ndarray; got {type(contrast).__name__}.")


def _contrast_is_estimable(design_matrix: np.ndarray, contrast_vector: np.ndarray) -> bool:
    """Whether ``contrast_vector`` lies in the row space of ``design_matrix``."""
    row_projector = np.linalg.pinv(design_matrix) @ design_matrix
    return bool(
        np.allclose(row_projector @ contrast_vector, contrast_vector, rtol=1e-7, atol=1e-10)
    )


def _maybe_log_expression(
    values: np.ndarray,
    *,
    log_expression: bool,
    eps: float,
) -> np.ndarray:
    """Return scalar values, optionally transformed to ``log(values + eps)``."""
    if not log_expression:
        return values
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive when log_expression=True, got {eps}.")
    if np.any(values + eps <= 0):
        raise ValueError("log_expression=True requires values + eps to be strictly positive.")
    return np.log(values + eps)


# ---------------------------------------------------------------------------
# Two-group spectral comparison functions
# ---------------------------------------------------------------------------


def compare_two_groups(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "analytic",
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """
    Test, for every gene, whether its spatial-pattern spectrum differs between two groups.

    Parameters
    ----------
    spectra : np.ndarray
        Per-sample spectral features of shape ``(n_samples, n_genes, n_bins)``.
    groups : np.ndarray
        Group labels of length ``n_samples`` taking exactly two distinct values
        (mapped internally to 0/1 in sorted order).
    gene_names : sequence of str, optional
        Names for the gene axis. If None, integer indices are used.
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
        Test statistic:

        - ``'log_l2'`` — (optionally weighted) L2 distance between mean
          log-spectra. Global / summary statistic. Pair with
          ``null='analytic'`` for an analytic mixture-χ² null that bypasses
          the small-n permutation BH-floor; ``null='permutation'``
          falls back to label permutations with exact enumeration when
          ``C(n, n_a) ≤ max_exact_permutations``.
        - ``'welch_t_cauchy'`` — per-bin Welch two-sided t-test with
          **analytic** (t-distribution) p-values combined across bins
          via Cauchy combination. Analytic is the whole point:
          permutation p-values would floor at ``1/(n_perm + 1)`` per
          bin, which would also floor the gene-level combined p-value
          and destroy BH-FDR power across thousands of genes. Yields
          an extra ``P_value_per_bin`` column.
    null : {'analytic', 'permutation'}, default 'analytic'
        Null-distribution method. ``'analytic'`` (the default) uses Liu's
        mixture-χ² approximation for the L2 quadratic form:
        under H₀ the statistic ``T² = D'WD`` is distributed as a
        weighted sum of χ²₁ variables whose tail is integrated via Liu's
        approximation (see :func:`quadsv.statistics.liu_sf`).
        ``'permutation'`` uses the empirical sample-label permutation
        null and is the only option that respects the
        ``n_perm`` / ``random_state`` / ``max_exact_permutations`` arguments.
        ``welch_t_cauchy`` carries its own analytic t-distribution null
        and ignores this selector. For selector-controlled tests,
        ``null='analytic'`` is supported with ``statistic='log_l2'``.

        **Sample-size guidance** (residual df = ``n_a + n_b - 2``):

        - df ≥ 4 (n_a + n_b ≥ 6): ``'analytic'`` recommended — strong
          calibration + sensitivity; sweeps the top of our benchmark.
        - df ≥ 3 (n_a + n_b ≥ 5): ``'analytic'`` acceptable.
        - df < 3 (n_a + n_b ≤ 4): ``'analytic'`` emits a ``UserWarning``;
          σ̂² has ≥ 67% relative noise so per-test calibration may be
          anti-conservative. Prefer ``statistic='welch_t_cauchy'``
          (per-bin Welch t with proper df-corrected denominator) or
          stay with ``null='permutation'`` if the cohort allows
          enough exact relabellings.
    n_perm : int, default 1000
        Number of label permutations for the null distribution.
    max_exact_permutations : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count.
    random_state : int, optional
        Seed for the permutation RNG.
    freq_weights : np.ndarray, optional
        Only used by ``statistic='log_l2'``. Non-negative weights of length
        ``n_bins`` (the number of frequency bins); internally renormalized to
        sum-1. Lets the user emphasize specific frequencies — e.g., a
        polynomial low-pass shape to mirror a CAR kernel, or an exponential
        high-pass shape to mirror a Gaussian kernel. ``None`` (default)
        means uniform weights.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (i.e., apply :func:`quadsv.comparators.normalization.normalize_shape`
        to ``spectra`` first). Use to isolate shape-only /
        frequency-redistribution signals independent of overall amplitude.
        Works with every valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``
        (BH-FDR), sorted by descending statistic. When
        ``statistic='welch_t_cauchy'``, the frame also carries a
        ``P_value_per_bin`` object column — each entry is an
        ``(n_bins,)`` numpy array of per-bin analytic Welch p-values for that gene.

    Raises
    ------
    ValueError
        If ``statistic`` is unknown, ``groups`` does not contain exactly two values,
        or shapes are inconsistent.
    """
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_NULL_OPTIONS)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    group_codes = (groups == uniq[1]).astype(int)  # 0 = first label sorted, 1 = second

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    rng = np.random.default_rng(random_state)  # ignored if using analytic null

    # Run per-bin t tests and combine into a single gene-level statistic
    # ``welch_t_cauchy`` carries its own analytic null and the ``null`` argument is ignored.
    if statistic == "welch_t_cauchy":
        if freq_weights is not None:
            logger.debug("freq_weights is ignored by statistic='welch_t_cauchy'.")
        observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(spectra, group_codes)
        summary_stat = observed.max(axis=-1)  # reportable scalar per gene
        df = _comparison_frame(
            n_genes,
            gene_names,
            summary_stat,
            combined_p,
            extra={"P_value_per_bin": list(per_bin_p)},
        )
        return df

    # Test the log_l2 statistic with Liu's analytic mixture-chi-square null.
    if null == "analytic":
        group_a_mask = group_codes == 0
        group_a = spectra[group_a_mask]
        group_b = spectra[~group_a_mask]
        n_a = int(group_a_mask.sum())
        n_b = int((~group_a_mask).sum())
        n_bins = spectra.shape[-1]
        weights = _resolve_freq_weights(freq_weights, n_bins)

        # Analytic log_l2 test. The observed statistic is
        # sqrt(D.T @ W @ D); the null uses the pooled full log-spectrum
        # covariance so correlated frequency bins are not treated as independent.
        observed = _stat_log_l2(group_a, group_b, freq_weights=freq_weights)

        # Estimate the pooled within-group covariance of log-spectra across all
        # genes. Flattening sample and gene axes turns sum_g R_g.T @ R_g into
        # one matrix multiply while keeping the frequency-bin covariance full.
        log_group_a = np.log(np.maximum(group_a, 1e-12))
        log_group_b = np.log(np.maximum(group_b, 1e-12))
        residuals_a = log_group_a - log_group_a.mean(axis=0, keepdims=True)
        residuals_b = log_group_b - log_group_b.mean(axis=0, keepdims=True)
        residuals = np.concatenate(
            [residuals_a, residuals_b], axis=0
        )  # (n_samples, n_genes, n_bins)
        n_total, _, _ = residuals.shape
        df_resid = max(n_a + n_b - 2, 1)
        residual_2d = residuals.reshape(n_total * n_genes, n_bins)
        sigma_log = (residual_2d.T @ residual_2d) / (n_genes * df_resid)  # (n_bins, n_bins)
        _maybe_warn_small_df_analytic(df_resid)

        # The two-group contrast variance scales the covariance of the mean
        # log-spectrum difference before Liu integrates the chi-square mixture.
        contrast_scale = (1.0 / max(n_a, 1)) + (1.0 / max(n_b, 1))
        sqrt_weights = np.sqrt(weights)
        weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
        lambs = np.maximum(np.linalg.eigvalsh(weighted_cov * contrast_scale), 0.0)
        pvals = _log_l2_analytic_pvalues(observed, lambs)
        df = _comparison_frame(n_genes, gene_names, observed, pvals)
        return df

    # Permutation path: generate the exchangeable label set once, then evaluate
    # the same log_l2 statistic under every relabelling.
    perm_labels, is_exact = _exchangeable_group_labels(
        group_codes,
        n_perm,
        rng,
        max_exact_permutations=max_exact_permutations,
    )
    if is_exact:
        logger.info(
            "Exact permutation test: enumerated %d distinct relabellings " "(C(%d, %d)).",
            perm_labels.shape[0],
            n_samples,
            int((group_codes == 0).sum()),
        )
    observed, null_dist = _run_statistic_with_perm(
        statistic, spectra, group_codes, perm_labels, freq_weights=freq_weights
    )
    # Tail probability is Pr(null >= observed), with +1 smoothing.
    pvals = _permutation_pvalue(observed, null_dist)

    df = _comparison_frame(n_genes, gene_names, observed, pvals)
    return df


def compare_two_groups_masked(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    null: str = "analytic",
    n_perm: int = 1000,
    max_exact_permutations: int = 10000,
    random_state: int | None = None,
    min_samples_per_group: int = 2,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """
    Per-gene two-group pattern test with **incomplete data** across samples.

    For each gene, only the subset of samples with ``presence[:, g] == True``
    contributes to the observed statistic and to the label-permutation null.
    Genes that fail to reach ``min_samples_per_group`` observations in at
    least one group are reported with ``NaN`` p-values and the number of
    observed samples per group, so the user sees why they were skipped.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)``.
    groups : np.ndarray
        ``(n_samples,)``, exactly two distinct labels.
    presence : np.ndarray
        ``(n_samples, n_genes)`` boolean mask. ``True`` = gene is observed
        in that sample (contributes); ``False`` = gene is absent (ignored).
    gene_names : sequence of str, optional
    statistic : {'log_l2', 'welch_t_cauchy'}, default 'log_l2'
    null : {'analytic', 'permutation'}, default 'analytic'
        Null-distribution method. ``'analytic'`` (the default) uses a
        Liu mixture-χ² test adapted for the masked
        case via a **mask-aware pooled-Σ** estimator: a single global
        ``(n_bins, n_bins)`` covariance is accumulated across every gene's present
        (sample, gene) cells (each gene contributes ``n_g - 2``
        residual degrees of freedom), and per-gene noncentrality
        scaling ``v_{c,g} = 1/n_a_g + 1/n_b_g`` adjusts the eigenvalues
        for that gene's specific cohort. Cross-bin correlation
        structure is taken to be homogeneous across genes (the same
        A3 assumption used in :func:`compare_two_groups` with the analytic null).
        Empirical calibration on synthetic missingness up to 50%
        matches the unmasked analytic path. Currently supported only with
        ``statistic='log_l2'``.

        ``'permutation'`` runs a per-gene permutation test,
        exact-enumerated when ``C(n_g, n_a_g) <= max_exact_permutations``
        (most genes at small samples).
    n_perm : int, default 1000
        Number of label permutations for the null distribution.
    max_exact_permutations : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count.
    random_state : int, optional
        Seed for the permutation RNG.
    min_samples_per_group : int, default 2
        Minimum observed samples in each group for the gene to be tested.
    freq_weights : np.ndarray, optional
        Only consumed by ``statistic='log_l2'`` (same semantics as
        :func:`compare_two_groups`).
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the statistic is computed
        (same semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals. Works with every
        valid ``statistic=`` value.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``,
        ``n_obs_A``, ``n_obs_B``. For ``'welch_t_cauchy'`` a
        ``P_value_per_bin`` column is also included (``None`` for skipped
        genes). BH-FDR is computed only over tested genes.
    """
    if statistic not in _AVAILABLE_STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Options: {list(_AVAILABLE_STATISTICS)}."
        )
    if null not in _NULL_OPTIONS:
        raise ValueError(f"Unknown null='{null}'. Options: {list(_NULL_OPTIONS)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    group_codes = (groups == uniq[1]).astype(int)

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    rng = np.random.default_rng(random_state)  # ignored if using analytic null

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    # Analytic masked path: precompute global pooled Σ + eigvalsh; then per-gene
    # T², v_c-scaled λ, and Liu-tail p-value. ``welch_t_cauchy`` carries its
    # own analytic null and falls through to the per-gene branch below.
    if null == "analytic" and statistic == "log_l2":
        group_a_mask = group_codes == 0
        log_spectra = np.log(np.maximum(spectra, 1e-12))
        sigma_acc = np.zeros((n_bins, n_bins), dtype=np.float64)
        total_df = 0

        # First pass: accumulate the mask-aware pooled full covariance once
        # across all testable genes. Per-gene analytic statistics below reuse this
        # covariance with their own cohort-size contrast scale.
        for gene_idx in range(n_genes):
            sample_mask = presence[:, gene_idx]
            idx_a = np.where(group_a_mask & sample_mask)[0]
            idx_b = np.where(~group_a_mask & sample_mask)[0]
            if len(idx_a) < min_samples_per_group or len(idx_b) < min_samples_per_group:
                continue

            # Center each group's log-spectra separately so the covariance
            # estimates within-group noise rather than group-level signal.
            log_a = log_spectra[idx_a, gene_idx, :]
            log_b = log_spectra[idx_b, gene_idx, :]
            res_a = log_a - log_a.mean(axis=0, keepdims=True)
            res_b = log_b - log_b.mean(axis=0, keepdims=True)
            residuals = np.concatenate([res_a, res_b], axis=0)
            df_gene = max(len(idx_a) + len(idx_b) - 2, 1)
            sigma_acc += residuals.T @ residuals
            total_df += df_gene

        if total_df == 0:
            raise ValueError(
                "compare_two_groups_masked + null='analytic': no genes meet "
                f"min_samples_per_group={min_samples_per_group} per arm. "
                "Cannot estimate the pooled covariance - use null='permutation' "
                "or relax the minimum."
            )

        weights = _resolve_freq_weights(freq_weights, n_bins)
        sqrt_weights = np.sqrt(weights)
        sigma_log = sigma_acc / total_df
        weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
        base_lambs = np.maximum(np.linalg.eigvalsh(weighted_cov), 0.0)

        rows: list[dict[str, Any]] = []
        df_per_gene: list[int] = []
        # Second pass: compute the observed statistic and Liu p-value for each
        # gene using its own observed sample counts.
        for gene_idx in range(n_genes):
            sample_mask = presence[:, gene_idx]
            group_a = group_codes[sample_mask] == 0
            group_b = group_codes[sample_mask] == 1
            n_a, n_b = int(group_a.sum()), int(group_b.sum())
            row: dict[str, Any] = {
                "Feature": gene_names[gene_idx],
                "n_obs_A": n_a,
                "n_obs_B": n_b,
                "Statistic": np.nan,
                "P_value": np.nan,
            }
            if n_a < min_samples_per_group or n_b < min_samples_per_group:
                rows.append(row)
                continue
            idx_a = np.where((group_codes == 0) & sample_mask)[0]
            idx_b = np.where((group_codes == 1) & sample_mask)[0]
            # Observed effect is the weighted L2 norm of the mean log-spectrum
            # difference for this gene's available samples.
            mean_diff = log_spectra[idx_a, gene_idx, :].mean(axis=0) - log_spectra[
                idx_b, gene_idx, :
            ].mean(axis=0)
            statistic_value = float(np.sqrt(np.sum(weights * mean_diff**2)))
            statistic_sq = statistic_value * statistic_value
            # Missingness changes n_a/n_b per gene, so the eigenvalues get a
            # gene-specific contrast-variance scale even though covariance is shared.
            contrast_scale = (1.0 / n_a) + (1.0 / n_b)
            lambs = np.maximum(base_lambs * contrast_scale, 1e-30)
            row["Statistic"] = statistic_value
            row["P_value"] = float(liu_sf(np.array([statistic_sq]), lambs)[0])
            df_per_gene.append(n_a + n_b - 2)
            rows.append(row)

        if df_per_gene:
            _maybe_warn_small_df_analytic(int(np.median(df_per_gene)))

        df = pd.DataFrame(rows)
        tested = df["P_value"].notna()
        df["P_adj"] = np.nan
        if tested.any():
            df.loc[tested, "P_adj"] = apply_bh_correction(df.loc[tested, "P_value"])
        return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(
            drop=True
        )

    # Permutation / welch_t_cauchy masked path (per-gene loop).
    # Welch t-test ignores the null argument and always uses its own analytic null.
    rows: list[dict[str, Any]] = []
    for gene_idx in range(n_genes):
        sample_mask = presence[:, gene_idx]
        group_a = group_codes[sample_mask] == 0
        group_b = group_codes[sample_mask] == 1
        n_a, n_b = int(group_a.sum()), int(group_b.sum())
        row: dict[str, Any] = {
            "Feature": gene_names[gene_idx],
            "n_obs_A": n_a,
            "n_obs_B": n_b,
            "Statistic": np.nan,
            "P_value": np.nan,
        }
        if statistic == "welch_t_cauchy":
            row["P_value_per_bin"] = None

        if n_a < min_samples_per_group or n_b < min_samples_per_group:
            rows.append(row)
            continue

        sub = spectra[sample_mask, gene_idx : gene_idx + 1, :]  # (n_obs, 1, n_bins)
        sub_groups = group_codes[sample_mask]

        if statistic == "welch_t_cauchy":
            # Compute the analytic Welch t-test in the present subset.
            observed, combined_p, per_bin_p = _run_welch_t_cauchy_analytic(sub, sub_groups)
            row["Statistic"] = float(observed.max())
            row["P_value"] = float(combined_p[0])
            row["P_value_per_bin"] = per_bin_p[0]
        else:
            # Per-gene exchange set — enumerate exactly when C(n_obs, n_a_obs)
            # is small, otherwise sample. Subsets are typically smaller than
            # the global one so the exact path kicks in more often here.
            perm_labels, _ = _exchangeable_group_labels(
                sub_groups,
                n_perm,
                rng,
                max_exact_permutations=max_exact_permutations,
            )
            observed, null = _run_statistic_with_perm(
                statistic, sub, sub_groups, perm_labels, freq_weights=freq_weights
            )
            pval = _permutation_pvalue(observed, null)
            row["Statistic"] = float(observed[0])
            row["P_value"] = float(pval[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    # BH-correction over tested (non-NaN) genes only.
    tested = df["P_value"].notna()
    df["P_adj"] = np.nan
    if tested.any():
        df.loc[tested, "P_adj"] = apply_bh_correction(df.loc[tested, "P_value"])
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# GLM-based continuous spectral comparison functions
# ---------------------------------------------------------------------------


def compare_glm(
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    """Log-L2 analytic spectral comparison via a design matrix and contrast.

    Generalises :func:`compare_two_groups` from binary group labels to an
    arbitrary GLM design matrix and a single-DOF linear contrast. The
    binary case is recovered exactly by passing
    ``design=pd.DataFrame({"group": groups})`` and ``contrast="group"``;
    p-values match ``compare_two_groups(..., statistic="log_l2",
    null="analytic")`` to machine precision.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)`` spectral features (raw, not logged).
    design : pd.DataFrame or np.ndarray
        Sample-level metadata. ``DataFrame`` columns are auto-encoded via
        :mod:`patsy` (treatment-coded categoricals + intercept);
        ``ndarray`` is passed through as the design matrix verbatim
        (caller responsible for the intercept column).
    contrast : str, dict, or np.ndarray
        Linear-contrast specification:

        - ``str`` — name of a design column. Auto-resolves treatment-coded
          categoricals (e.g., ``"genotype"`` matches ``"genotype[T.TG]"``).
          Multi-DOF (3+ level factor) contrasts must be passed as an
          explicit dict or ndarray.
        - ``dict[str, float]`` — coefficient per design column.
        - ``np.ndarray`` of length ``p`` — raw contrast vector.
    gene_names : sequence of str, optional
    freq_weights : np.ndarray, optional
        Optional non-negative weights over frequency bins, same semantics as
        :func:`compare_two_groups` with ``statistic="log_l2"``.
    normalize_shape : bool, default False
        If True, divide each per-(sample, gene) spectrum by its sum along
        the trailing (frequency) axis before the GLM is fit (same
        semantics as in :func:`compare_two_groups`). Use to isolate
        shape-only / frequency-redistribution signals along the design
        contrast independent of overall amplitude.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj`` —
        same schema as :func:`compare_two_groups`.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``contrast`` cannot be resolved.
    """
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")
    n_samples, n_genes, n_bins = spectra.shape

    # Resolve the design matrix and contrast vector.
    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    n_terms = design_matrix.shape[1]
    if contrast_vector.shape != (n_terms,):
        raise ValueError(f"contrast length {contrast_vector.shape} != design cols ({n_terms},).")
    rank = int(np.linalg.matrix_rank(design_matrix))
    df_resid = n_samples - rank
    if df_resid <= 0:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={rank}."
        )
    if not _contrast_is_estimable(design_matrix, contrast_vector):
        raise ValueError(
            "contrast is not estimable from the supplied design matrix "
            f"(rank={rank}, n_terms={n_terms})."
        )

    # Fit one OLS model per (gene, bin) by flattening those response columns
    # into a single 2D matrix. This keeps the GLM path vectorized.
    log_spectra = np.log(np.maximum(spectra, 1e-12))
    response = log_spectra.reshape(n_samples, n_genes * n_bins)
    xtx_inv = np.linalg.pinv(design_matrix.T @ design_matrix)
    beta_flat = xtx_inv @ (design_matrix.T @ response)
    residual_flat = response - design_matrix @ beta_flat
    beta = beta_flat.reshape(n_terms, n_genes, n_bins)
    residuals = residual_flat.reshape(n_samples, n_genes, n_bins)
    _maybe_warn_small_df_analytic(df_resid)

    # The contrast effect theta is one signed log-spectrum difference per gene.
    theta = np.tensordot(contrast_vector, beta, axes=([0], [0]))
    # Pool residual covariance across genes, preserving frequency-bin
    # correlations for the analytic null.
    residual_2d = residuals.reshape(n_samples * n_genes, n_bins)
    sigma_log = (residual_2d.T @ residual_2d) / (n_genes * df_resid)
    contrast_scale = float(contrast_vector @ xtx_inv @ contrast_vector)

    # Convert contrast effects to the log_l2 statistic and integrate the
    # corresponding weighted chi-square mixture with Liu's approximation.
    weights = _resolve_freq_weights(freq_weights, n_bins)
    statistic_sq = (weights * theta**2).sum(axis=-1)
    observed = np.sqrt(statistic_sq)
    sqrt_weights = np.sqrt(weights)
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    lambs = np.maximum(np.linalg.eigvalsh(weighted_cov * contrast_scale), 0.0)
    pvals = _log_l2_analytic_pvalues(observed, lambs)

    return _comparison_frame(n_genes, gene_names, observed, pvals)


def compare_glm_masked(  # noqa: C901
    spectra: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    freq_weights: np.ndarray | None = None,
    normalize_shape: bool = False,
    min_resid_df: int = 1,
) -> pd.DataFrame:
    """Masked design-matrix contrast test for gene-specific missing spectra.

    This is the incomplete-data analogue of :func:`compare_glm`. For each
    gene, only samples with ``presence[:, g]`` are used to fit the OLS model
    on log-spectra. Genes whose observed design has too little residual
    degrees of freedom, or whose contrast is not estimable after masking, are
    retained in the output with ``NaN`` p-values.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, n_bins)`` spectral features.
    design : pandas.DataFrame or np.ndarray
        Sample-level design, encoded exactly as in :func:`compare_glm`.
    contrast : str, dict, or np.ndarray
        Single linear contrast, resolved exactly as in :func:`compare_glm`.
    presence : np.ndarray
        Boolean ``(n_samples, n_genes)`` mask. ``True`` means the gene is
        observed in that sample and contributes to that gene's model.
    freq_weights : np.ndarray, optional
        Optional non-negative weights over frequency bins, same semantics as
        :func:`compare_glm`.
    normalize_shape : bool, default False
        If True, apply :func:`quadsv.comparators.normalization.normalize_shape`
        before fitting each gene model.
    min_resid_df : int, default 1
        Minimum per-gene residual degrees of freedom required for testing.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``n_obs``,
        ``df_resid``, and ``P_adj``. BH-FDR is computed over finite p-values.
    """
    if int(min_resid_df) < 1:
        raise ValueError(f"min_resid_df must be >= 1, got {min_resid_df}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, n_bins), got {spectra.shape}.")

    n_samples, n_genes, n_bins = spectra.shape
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )

    # Resolve the design matrix and contrast vector.
    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)

    if normalize_shape:
        spectra = _normalize_shape_apply(spectra)

    n_terms = design_matrix.shape[1]
    if contrast_vector.shape != (n_terms,):
        raise ValueError(f"contrast length {contrast_vector.shape} != design cols ({n_terms},).")

    # Prepare input and output arrays.
    log_spectra = np.log(np.maximum(spectra, 1e-12))
    weights = _resolve_freq_weights(freq_weights, n_bins)
    sqrt_weights = np.sqrt(weights)
    observed = np.full(n_genes, np.nan, dtype=float)
    pvals = np.full(n_genes, np.nan, dtype=float)
    n_obs = np.zeros(n_genes, dtype=int)
    df_resid = np.zeros(n_genes, dtype=int)
    theta = np.full((n_genes, n_bins), np.nan, dtype=float)
    contrast_var = np.full(n_genes, np.nan, dtype=float)
    eligible = np.zeros(n_genes, dtype=bool)
    sigma_acc = np.zeros((n_bins, n_bins), dtype=float)
    total_df = 0

    # First pass: fit a masked OLS model for every gene, storing each gene's
    # contrast effect and accumulating a shared residual covariance.
    for gene_idx in range(n_genes):
        sample_mask = np.asarray(presence[:, gene_idx], dtype=bool)
        n_obs[gene_idx] = int(sample_mask.sum())
        if n_obs[gene_idx] == 0:
            continue

        gene_design = design_matrix[sample_mask]
        rank_gene = int(np.linalg.matrix_rank(gene_design))
        df_gene = n_obs[gene_idx] - rank_gene
        df_resid[gene_idx] = df_gene
        if df_gene < int(min_resid_df):
            continue

        # A contrast is estimable only if it lies in the row space of this
        # gene's observed design. Missing samples can drop factor levels.
        if not _contrast_is_estimable(gene_design, contrast_vector):
            continue

        xtx_inv = np.linalg.pinv(gene_design.T @ gene_design)
        contrast_scale = float(contrast_vector @ xtx_inv @ contrast_vector)
        if not np.isfinite(contrast_scale) or contrast_scale <= 0.0:
            continue

        # Fit this gene's log-spectrum matrix, one response column per bin.
        gene_response = log_spectra[sample_mask, gene_idx, :]
        beta = xtx_inv @ (gene_design.T @ gene_response)
        residuals = gene_response - gene_design @ beta
        theta[gene_idx] = contrast_vector @ beta
        contrast_var[gene_idx] = contrast_scale
        sigma_acc += residuals.T @ residuals
        total_df += df_gene
        eligible[gene_idx] = True

    if total_df == 0:
        raise ValueError(
            "compare_glm_masked: no genes have enough observed "
            "samples and an estimable contrast to estimate the pooled covariance."
        )

    _maybe_warn_small_df_analytic(int(np.median(df_resid[eligible])))
    sigma_log = sigma_acc / total_df
    weighted_cov = sqrt_weights[:, None] * sigma_log * sqrt_weights[None, :]
    # Second pass: reuse the shared covariance, scaling by each gene's
    # contrast variance because masking changes the observed design per gene.
    for gene_idx in np.where(eligible)[0]:
        statistic_sq = float(np.sum(weights * theta[gene_idx] ** 2))
        observed[gene_idx] = float(np.sqrt(statistic_sq))
        lambs = np.maximum(np.linalg.eigvalsh(weighted_cov * contrast_var[gene_idx]), 0.0)
        pvals[gene_idx] = float(liu_sf(np.array([statistic_sq]), np.maximum(lambs, 1e-30))[0])

    return _comparison_frame(
        n_genes,
        gene_names,
        observed,
        pvals,
        extra={"n_obs": n_obs, "df_resid": df_resid},
    )


# ---------------------------------------------------------------------------
# Pseudo-bulk expression scalar comparison functions
# ---------------------------------------------------------------------------


def compare_two_groups_scalar(
    values: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Per-gene two-sample test on scalar per-sample values (classical DE).

    The natural companion to :func:`compare_two_groups`: tested on the DC scalars
    (per-gene grid means) produced by
    :func:`quadsv.comparators.features.compute_sample_spectrum`.

    For each gene, the function reports ``Statistic = abs(t)`` where ``t`` is
    the Welch two-sample t statistic, and ``P_value`` is the analytic two-sided
    tail probability under the Welch-Satterthwaite t-distribution null.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)`` — e.g.,
        log-normalized mean expression on each slide.
    groups : np.ndarray
        Group labels of length ``n_samples`` with exactly two distinct values.
    gene_names : sequence of str, optional
        Gene names. Integer indices if None.
    log_expression : bool, default False
        If True, test ``log(values + eps)`` instead of raw scalar
        expression values. ``Mean_diff`` is then reported on the log scale.
    eps : float, default 1e-12
        Additive offset used only when ``log_expression=True``.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(Welch t)``), ``Mean_diff``
        (``mean_groupA − mean_groupB``), ``P_value``, ``P_adj`` (BH-FDR), sorted
        by descending ``Statistic``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``groups`` does not contain exactly two
        distinct values.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)
    n_samples, n_genes = values.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    group_codes = (groups == uniq[1]).astype(int)

    # Compute the Welch t-test.
    a_vals = values[group_codes == 0]
    b_vals = values[group_codes == 1]
    mean_diff = a_vals.mean(axis=0) - b_vals.mean(axis=0)
    t_stat, pvals = _welch_test(a_vals, b_vals)
    observed = np.abs(t_stat)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Mean_diff": mean_diff,
            "P_value": pvals,
        }
    )
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)


def compare_glm_scalar(
    values: np.ndarray,
    design: pd.DataFrame | np.ndarray,
    contrast: str | dict[str, float] | np.ndarray,
    gene_names: Sequence[str] | None = None,
    *,
    log_expression: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Per-gene scalar linear-model test via a design matrix and contrast.

    This is the scalar-expression companion to :func:`compare_glm`: it fits an
    ordinary least-squares model independently for each gene's per-sample scalar
    values and tests one linear contrast with the usual OLS t statistic.
    ``Statistic`` is ``abs(t)``, and ``P_value`` is the analytic two-sided tail
    probability under a Student t null with ``n_samples - rank(design_matrix)`` residual
    degrees of freedom. Passing ``log_expression=True`` fits the model on
    ``log(values + eps)``, which tests multiplicative changes in non-negative
    expression-like means.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)``.
    design : pandas.DataFrame or np.ndarray
        Sample-level metadata. ``DataFrame`` columns are encoded with the same
        rules as :func:`compare_glm`; ``ndarray`` is used verbatim.
    contrast : str, dict, or np.ndarray
        Linear contrast specification resolved by the same rules as
        :func:`compare_glm`.
    gene_names : sequence of str, optional
    log_expression : bool, default False
        If True, test ``log(values + eps)`` instead of raw scalar expression
        values. ``Estimate`` is then reported on the log scale.
    eps : float, default 1e-12
        Additive offset used only when ``log_expression=True``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(t)``), ``Estimate`` (the
        signed contrast estimate), ``P_value``, and ``P_adj``. Rows are sorted by
        descending ``Statistic``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    n_samples, n_genes = values.shape

    values = _maybe_log_expression(values, log_expression=log_expression, eps=eps)

    design_matrix, design_columns = _build_design_matrix(design, n_samples)
    contrast_vector = _resolve_contrast(contrast, design_columns)
    rank = int(np.linalg.matrix_rank(design_matrix))
    df_resid = n_samples - rank
    if df_resid <= 0:
        raise ValueError(
            f"design has no residual degrees of freedom: n_samples={n_samples}, rank={rank}."
        )
    if not _contrast_is_estimable(design_matrix, contrast_vector):
        raise ValueError(
            "contrast is not estimable from the supplied design matrix "
            f"(rank={rank}, n_terms={design_matrix.shape[1]})."
        )

    # Fit the OLS model
    xtx_inv = np.linalg.pinv(design_matrix.T @ design_matrix)
    beta = xtx_inv @ design_matrix.T @ values  # (n_terms, n_genes)
    fitted = design_matrix @ beta
    resid = values - fitted
    sigma2 = np.sum(resid**2, axis=0) / df_resid
    contrast_var = float(contrast_vector @ xtx_inv @ contrast_vector)
    if contrast_var <= 0:
        raise ValueError("contrast has zero estimated variance under the supplied design.")

    # Compute the OLS contrast t statistic and p-value.
    estimate = np.asarray(contrast_vector @ beta, dtype=float)
    se = np.sqrt(np.maximum(sigma2, 0.0) * contrast_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = estimate / se
    zero_se = se <= np.finfo(float).tiny
    t_stat[zero_se & np.isclose(estimate, 0.0)] = 0.0
    perfect_effect = zero_se & ~np.isclose(estimate, 0.0)
    t_stat[perfect_effect] = np.sign(estimate[perfect_effect]) * np.inf
    observed = np.abs(t_stat)
    pvals = 2.0 * _t_dist.sf(observed, df_resid)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Estimate": estimate,
            "P_value": pvals,
        }
    )
    df["P_adj"] = apply_bh_correction(df["P_value"])
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)
