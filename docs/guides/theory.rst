Theoretical Results
===================

Our `accompanying paper <https://arxiv.org/pdf/2602.02825>`_ shows
that virtually every spatially-variable-gene (SVG) detection method
reduces to a single quadratic-form statistic, the Q-statistic. This
includes graph-based methods like Moran's I, parametric models, and
non-parametric dependence tests. The Q-statistic is

.. math::

   Q_n = \mathbf{z}^\top \mathbf{K} \mathbf{z},

where :math:`\mathbf{z}` is the standardised feature vector and
:math:`\mathbf{K}` is a kernel matrix that encodes spatial
structure. Under the null hypothesis of spatial independence,
:math:`Q_n` follows a weighted :math:`\chi^2` distribution whose
weights are the eigenvalues of :math:`\mathbf{K}`. Moment-matching
approximations turn that distribution into a fast p-value, giving
the Q-test. The kernel choice critically affects the consistency
and the power of the resulting test.

This page summarises the key theoretical results.


Theorem 1: Q-tests detect mean shifts only
------------------------------------------

All spatial Q-tests detect mean-shift patterns
(:math:`\mathbb{E}[\mathbf{x} \mid S = \mathbf{s}] \neq \mathbb{E}[\mathbf{x}]`).

This follows directly from using a linear kernel
:math:`l(x_i, x_j) = x_i x_j` in the quadratic form, which reduces
the conditional :math:`X \mid S = s_i` to its mean. To probe higher
moments (variance, distributional changes), swap in a non-linear
kernel. For example, apply a Gaussian or polynomial kernel to
:math:`\mathbf{z}^2` rather than :math:`\mathbf{z}`.

In spatial transcriptomics the distributional information is
typically absent. We observe only one realisation
:math:`(x_i, s_i)` per location, which blurs the line between mean
independence and statistical independence. Treating the signal as a
deterministic element of a Hilbert space
:math:`f \in L^2(\mathcal{S})` and applying spectrum theory of
kernel operators yields the consistency condition below.


Theorem 2: Consistency requires positive definiteness
-----------------------------------------------------

A spatial Q-test is universally consistent (power approaches 1 as
:math:`n \to \infty`) for every non-constant deterministic pattern
*if and only if* :math:`\mathbf{K}` is strictly positive definite.

Under :math:`H_0`, :math:`Q_n \approx \sum_i \lambda_i \chi^2_1`.
When some :math:`\lambda_i < 0` (indefinite kernel), signals aligned
with the negative eigenspace cancel signals aligned with the
positive eigenspace. We call this *spectral cancellation*, and it
costs the test power on composite patterns.

The implication is that you should pick a kernel with a non-negative
spectrum.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Kernel
     - Spectrum
     - Consistency
   * - Gaussian
     - Strictly positive
     - Guaranteed.
   * - Matérn
     - Strictly positive
     - Guaranteed.
   * - Moran's I
     - Indefinite
     - Spectral cancellation.
   * - Graph Laplacian
     - Non-negative
     - Guaranteed (high-frequency Moran).
   * - CAR (inverse Laplacian)
     - Strictly positive
     - Guaranteed (low-frequency Moran).


CAR is a scalable correction to Moran's I
-----------------------------------------

The Conditional Autoregressive (CAR) kernel is strictly positive
definite:

.. math::

   \mathbf{K} = (\mathbf{I} - \rho \tilde{\mathbf{W}})^{-1},

where :math:`\tilde{\mathbf{W}}` is the row-normalised adjacency
matrix and :math:`0 < \rho < 1` is the autoregressive parameter
(default :math:`0.9`). The matrix
:math:`\mathbf{I} - \rho \tilde{\mathbf{W}}` is the CAR *precision*
matrix. It is sparse with :math:`\mathcal{O}(nk)` non-zeros even
though :math:`\mathbf{K}` itself is dense, which is what makes CAR
scalable on large graphs.

Key properties:

- Strictly positive definite for any :math:`0 < \rho < 1`.
- Theoretically consistent (Theorem 2).
- Scales via sparse-precision LU solves, with no
  :math:`\mathcal{O}(n^2)` materialisation.
- Polynomial spectral decay that emphasises smooth, large-scale
  patterns while keeping a heavy tail for mid- and high-frequency
  components.

Use CAR as the default for graph-flavoured spatial-pattern
detection.


Null-distribution approximations
--------------------------------

Under :math:`H_0`, :math:`Q_n` follows a weighted :math:`\chi^2`
mixture

.. math::

   Q_n \;\sim\; \sum_{i=1}^{m} \lambda_i \chi^2_1,
   \qquad m = n - 1,

where :math:`\lambda_i` are the eigenvalues of the double-centred
kernel :math:`\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}`.
``quadsv`` ships three moment-matching fits, selected through the
``method`` argument of :func:`~quadsv.compute_null_params`:

- ``"clt"``: a normal fit to :math:`(c_1, c_2)`. Valid for any
  :math:`\mathbf{K}`, including indefinite kernels. This is the
  only sensible choice for Moran's I.
- ``"welch"``: a scaled central :math:`\chi^2` fit to
  :math:`(c_1, c_2)`. PSD kernels only. This is the default when it
  applies.
- ``"liu"``: a shifted non-central :math:`\chi^2` fit to
  :math:`(c_1, c_2, c_3, c_4)`. PSD kernels only. Tightest tail.

Here :math:`c_p = \operatorname{tr}(\tilde{\mathbf{K}}^p)` is the
:math:`p`-th spectral power sum. ``quadsv`` also applies a
finite-:math:`n` Dirichlet(1/2) correction to
:math:`\operatorname{Var}[Q_n]`. See :doc:`/guides/scaling` for the
formula, the cumulant-evaluation paths (FFT / NUFFT analytic,
Matrix Frobenius, Hutchinson probes), and the full complexity
table.


R-test: bivariate spatial co-expression
---------------------------------------

The R-statistic extends the Q-test to two features at a time:

.. math::

   R_{xy} = \mathbf{x}^\top \mathbf{K} \mathbf{y},

where :math:`\mathbf{x}` and :math:`\mathbf{y}` are standardised.
Under :math:`H_0`,
:math:`R_{xy} \sim \mathcal{N}\bigl(0, \operatorname{tr}(\mathbf{K}^2)\bigr)`,
which gives a fast Normal p-value.

A typical workflow:

1. Identify SVGs via the univariate Q-test.
2. Test pairwise R-statistics among the top SVGs.
3. Control FDR across comparisons.


Drop-in replacement for Moran's I
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Method
     - Test consistency
     - Use case
   * - Moran's I
     - Spectral cancellation.
     - Classical autocorrelation; backwards compatibility only.
   * - Graph Laplacian
     - Guaranteed.
     - High-frequency, local variation.
   * - CAR
     - Guaranteed.
     - Low-frequency, smooth patterns.

In practice:

- On a graph, use the CAR kernel for consistent, high-power
  detection across functional patterns.
- On 2-D physical space, use the FFT- and NUFFT-accelerated forms
  of any PSD kernel. Matérn is a common starting point.


See also
--------

- :doc:`/guides/quickstart` for practical recipes.
- :doc:`/guides/kernels` for kernel selection and design.
- :doc:`/guides/scaling` for null-distribution and operator
  complexity.
- :doc:`/autoapi/quadsv/statistics/index` for the statistical-test
  API.
