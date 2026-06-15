"""Spectrum normalization primitives for comparator outputs."""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_background", "normalize_covariates", "normalize_shape"]


def normalize_background(
    spectra: np.ndarray,
    *,
    axis: int = -2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cancel per-sample multiplicative gain via cross-gene geometric-mean centering.

    For each (sample, frequency-bin) pair, every gene's power is divided
    by the geometric mean of the spectrum across the genes axis. Use
    this to correct per-sample multiplicative gain (sequencing depth,
    antibody titre, dewaxing efficiency) that scales every gene's
    spectrum at every frequency by a sample-level factor.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra :math:`P` with shape
        ``(..., n_genes, n_bins)`` when using the default ``axis=-2``.
        Any leading dimensions (e.g., ``n_samples``) are broadcast over.
    axis : int, default -2
        Axis along which the cross-gene geometric mean is taken
        (the genes axis).
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added before the logarithm to keep
        zeros finite.

    Returns
    -------
    np.ndarray
        Background-normalized spectra :math:`\\tilde P`, same shape as
        ``spectra``. Never mutates the input.

    Notes
    -----
    Let :math:`P` denote the input spectrum, :math:`G` the number of
    genes (length of ``axis``), :math:`K` the number of frequency
    bins, and :math:`\\varepsilon` the ``eps`` floor. The per-bin
    geometric-mean background is

    .. math::

        b_{k} = \\exp\\!\\Bigl(
            \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr)
        \\Bigr),

    and the output is the per-bin gene-wise quotient

    .. math::

        \\tilde P_{g,k} = \\frac{P_{g,k}}{b_{k}}.

    Equivalently, in log-space this is per-bin mean centering across
    the genes axis,

    .. math::

        \\log \\tilde P_{g,k}
        = \\log\\bigl(P_{g,k} + \\varepsilon\\bigr)
          - \\tfrac{1}{G} \\sum_{g'=1}^{G}
            \\log\\bigl(P_{g',k} + \\varepsilon\\bigr),

    so after the transform :math:`\\prod_{g} \\tilde P_{g,k} = 1` at
    every bin :math:`k` - the cross-gene geometric mean at every
    frequency is unity.

    The operation is equivalent to a per-bin OLS regression of
    :math:`\\log P_{\\cdot,k}` against a constant (the cross-gene
    mean) followed by exponentiation. With a per-sample one-hot
    covariate stacked across all (sample, gene) rows, the residuals
    match :math:`\\log \\tilde P` row-for-row, so running this
    function sample-by-sample is identical to fitting a one-hot
    sample-ID covariate in log-space and residualizing.

    Companion functions:

    - :func:`normalize_covariates` removes per-bin bias linear in
      user-supplied covariate spectra (cell-type proportion maps,
      tissue domains, housekeeping templates).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalizing along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(2, 5, 8))      # (n_samples, n_genes, n_bins)
    >>> P_tilde = normalize_background(spec)
    >>> P_tilde.shape
    (2, 5, 8)
    >>> # Cross-gene geometric mean at each (sample, bin) is unity:
    >>> bool(np.allclose(np.prod(P_tilde, axis=-2), 1.0))
    True
    """
    log_spec = np.log(spectra + eps)
    bg = log_spec.mean(axis=axis, keepdims=True)
    return np.exp(log_spec - bg)


def normalize_covariates(
    spectra: np.ndarray,
    covariate_spectra: np.ndarray,
    *,
    fit_intercept: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Residualize log-spectra against the log of covariate spectra.

    Each gene's log-spectrum is regressed (per gene, OLS in log-space)
    on the log of the supplied covariate spectra plus an optional
    intercept; the function exponentiates and returns the residual
    spectrum. Use to remove the multiplicative contribution of
    structured per-bin templates (cell-type proportion maps,
    tissue-domain indicators, housekeeping composite expression) from
    every gene's per-frequency power.

    Operating in log-space matches the multiplicative noise model of
    spectral data, keeps the output strictly positive (so the result
    composes cleanly with the downstream ``log_l2`` test), and makes
    this helper commute exactly with :func:`normalize_background`
    (orthogonal projections along orthogonal axes - see Notes).

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative gene spectra :math:`P` of shape ``(n_genes, n_bins)`` to
        residualize.
    covariate_spectra : np.ndarray
        Non-negative covariate spectra :math:`C` of shape
        ``(n_covariates, n_bins)``.
    fit_intercept : bool, default True
        If True, prepend a column of ones to the design matrix
        :math:`X` so per-gene log-amplitude offsets along the
        frequency axis are absorbed.
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` added inside :math:`\\log(\\cdot)`
        on both ``spectra`` and ``covariate_spectra`` to keep zeros
        finite.

    Returns
    -------
    np.ndarray
        Residual spectra :math:`\\tilde P` of shape ``(n_genes, n_bins)``,
        strictly positive. Never mutates the input.

    Raises
    ------
    ValueError
        If ``covariate_spectra`` has a different last-axis length than
        ``spectra``.

    Notes
    -----
    Let :math:`P \\in \\mathbb{R}_{\\geq 0}^{G \\times K}` denote the
    input spectra (:math:`G` genes, :math:`K` frequency bins) and
    :math:`C \\in \\mathbb{R}_{\\geq 0}^{n_{\\mathrm{cov}} \\times K}`
    the covariate spectra. Build the log-design matrix

    .. math::

        X = \\bigl[\\, \\mathbf{1}_{K} \\;\\big|\\;
                \\log(C^{\\top} + \\varepsilon) \\,\\bigr]
            \\;\\in\\; \\mathbb{R}^{K \\times (n_{\\mathrm{cov}} + 1)},

    dropping the leading column :math:`\\mathbf{1}_{K}` when
    ``fit_intercept=False``. Fit per-gene OLS coefficients via the
    Moore-Penrose pseudoinverse :math:`X^{+}` against the log of the
    response,

    .. math::

        \\hat\\beta_{g} = X^{+}\\,
            \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top}
            \\;\\in\\; \\mathbb{R}^{n_{\\mathrm{cov}} + 1},

    and return the exponentiated residual

    .. math::

        \\tilde P_{g,k}
        = \\exp\\!\\Bigl(
            \\log( P_{g,k} + \\varepsilon )
            - X_{k,\\cdot}\\,\\hat\\beta_{g}
          \\Bigr).

    Equivalently,

    .. math::

        \\log \\tilde P_{g,\\cdot}^{\\top}
        = \\bigl( I_{K} - X X^{+} \\bigr)\\,
          \\bigl[ \\log( P_{g,\\cdot} + \\varepsilon ) \\bigr]^{\\top},

    i.e., the orthogonal projection of each gene's **log-spectrum**
    onto the orthogonal complement of the column space of :math:`X`,
    then exponentiated.

    **Commutativity with** :func:`normalize_background`. In log-space
    the two operations are left- vs right-multiplication of the
    :math:`G \\times K` log-spectrum matrix by orthogonal-projection
    matrices on disjoint axes,

    .. math::

        \\mathrm{bg}: \\;\\log P \\;\\mapsto\\;
            \\bigl( I_{G} - \\tfrac{1}{G}\\mathbf{1}_{G}\\mathbf{1}_{G}^{\\top}
            \\bigr)\\,\\log P,
        \\qquad
        \\mathrm{cov}: \\;\\log P \\;\\mapsto\\;
            \\log P \\,\\bigl( I_{K} - X X^{+} \\bigr).

    Left- and right-multiplication trivially commute, so we have the exact identity
    ``normalize_background(normalize_covariates(P)) ==
    normalize_covariates(normalize_background(P))``.

    With ``fit_intercept=True`` and **no** covariates (empty
    ``covariate_spectra``), this reduces to per-gene log-mean centering
    along the frequency axis,

    .. math::

        \\tilde P_{g,k}
        = \\frac{P_{g,k} + \\varepsilon}
                {\\exp\\!\\bigl(\\tfrac{1}{K}
                        \\sum_{k'=1}^{K}\\log(P_{g,k'} + \\varepsilon)
                  \\bigr)},

    i.e., dividing each gene's spectrum by its own cross-bin geometric
    mean - a per-gene companion to :func:`normalize_background`'s
    per-bin cross-gene operation, distinct from
    :func:`normalize_shape`'s arithmetic-mean / sum-1 normalization.

    Companion functions:

    - :func:`normalize_background` removes per-sample multiplicative
      gain via cross-gene geometric-mean centering in log-space
      (perpendicular axis to this function).
    - :func:`normalize_shape` removes per-(sample, gene) amplitude
      by L1-normalizing along the frequency axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng  = np.random.default_rng(0)
    >>> spec = rng.lognormal(size=(20, 8))      # (n_genes, n_bins)
    >>> cov  = rng.lognormal(size=(2, 8))       # (n_covariates, n_bins)
    >>> resid = normalize_covariates(spec, cov)
    >>> resid.shape
    (20, 8)
    >>> bool((resid > 0).all())     # log-space output is strictly positive
    True
    """
    if spectra.shape[-1] != covariate_spectra.shape[-1]:
        raise ValueError(
            f"Last axis must match: spectra has n_bins={spectra.shape[-1]}, "
            f"covariate_spectra has n_bins={covariate_spectra.shape[-1]}."
        )
    n_bins = spectra.shape[-1]
    log_spec = np.log(spectra + eps)
    log_cov = np.log(covariate_spectra + eps)

    X = log_cov.T
    if fit_intercept:
        X = np.hstack([np.ones((n_bins, 1)), X])
    pinv = np.linalg.pinv(X)
    fitted = (X @ pinv @ log_spec.T).T
    return np.exp(log_spec - fitted)


def normalize_shape(
    spectra: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project each spectrum onto the probability simplex along ``axis``.

    Each fibre along ``axis`` is divided by its L1 norm, so the result
    is a proper probability distribution over the entries along that
    axis. Two fibres that differ only by a positive scalar produce
    identical outputs - only the **shape** of the power-vs-frequency
    curve survives, the overall scale is removed.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative spectra :math:`P`. Any leading dimensions are
        preserved; normalization acts along ``axis`` only.
    axis : int, default -1
        Axis to L1-normalize along (typically the trailing
        frequency-bin axis).
    eps : float, default 1e-12
        Floor :math:`\\varepsilon` on the per-fibre sum to avoid
        division by zero.

    Returns
    -------
    np.ndarray
        Shape-normalized spectra :math:`\\tilde P`, same shape as
        ``spectra``, summing to 1 along ``axis``. Never mutates the
        input.

    Notes
    -----
    Let :math:`P` denote the input spectrum with :math:`K` entries
    along ``axis`` and :math:`\\varepsilon` the ``eps`` floor. The
    output is the per-fibre L1 quotient

    .. math::

        \\tilde P_{\\ldots,k}
        = \\frac{P_{\\ldots,k}}
                {\\sum_{k'=1}^{K} P_{\\ldots,k'} + \\varepsilon},

    so :math:`\\sum_{k} \\tilde P_{\\ldots,k} = 1` for every
    leading-index combination (modulo the :math:`\\varepsilon` floor;
    fibres whose total sum is below :math:`\\varepsilon` are
    effectively returned unchanged because the numerator dominates
    the floor).

    Equivalently, in log-space this is per-fibre log-sum centering,

    .. math::

        \\log \\tilde P_{\\ldots,k}
        = \\log P_{\\ldots,k}
          - \\log\\!\\Bigl( \\textstyle\\sum_{k'=1}^{K}
            P_{\\ldots,k'} + \\varepsilon \\Bigr).

    After this transform every fibre is a probability vector over
    frequency bins, so distances such as Jensen-Shannon and
    total-variation are well-defined between fibres.

    Used internally by the spectrum comparison functions
    (:func:`quadsv.comparators.multisample.compare_two_groups`,
    :func:`quadsv.comparators.multisample.compare_two_groups_masked`,
    :func:`quadsv.comparators.multisample.compare_glm`) when their
    ``normalize_shape=True`` keyword argument is set - the
    differential-frequency test then fires only on shape redistribution
    across radial bins, not on overall amplitude changes.

    Companion functions:

    - :func:`normalize_background` removes per-sample multiplicative
      gain via cross-gene geometric-mean centering in log-space.
    - :func:`normalize_covariates` removes per-bin bias linear in
      user-supplied covariate spectra.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    >>> P_tilde = normalize_shape(x, axis=-1)
    >>> bool(np.allclose(P_tilde.sum(axis=-1), 1.0))
    True
    >>> bool(np.allclose(P_tilde[0], P_tilde[1]))  # only the shape survives
    True
    """
    total = spectra.sum(axis=axis, keepdims=True)
    return spectra / (total + eps)


def _normalize_shape_apply(spectra: np.ndarray) -> np.ndarray:
    """Apply the comparison-default shape normalization along the frequency axis."""
    return normalize_shape(spectra, axis=-1)
