Kernel Design
=============

The Q-statistic measures the strength of spatial dependence in a
feature, but the kernel decides which spatial patterns count as
"dependent". Smooth large-scale gradients (low-frequency), sharp
boundaries between neighbouring spots (high-frequency), and
graph-defined neighbourhoods are all valid choices, and they are
captured by different kernels. Two genes can swap which one scores
higher just by swapping the kernel.

This page is the practical guide. For the underlying math, see
:doc:`/guides/theory` (Theorems 1-2 and the CAR derivation).


Default recommendations
-----------------------

The choice splits along two axes: what pattern you are looking for
(smooth gradient or sharp local variation) and how the data sits in
space (irregular coordinates, graph, or regular grid).

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Data layout
     - Pattern of interest
     - Pick
   * - Coordinate cloud
     - Smooth, large-scale gradient
     - :class:`~quadsv.NUFFTKernel` with ``method="matern"``
       (``nu=1.5``).
   * - Coordinate cloud (graph-defined)
     - Smooth, large-scale gradient
     - :class:`~quadsv.MatrixKernel` with ``method="car"``
       (``rho=0.9``, ``k_neighbors=4``).
   * - Coordinate cloud (graph-defined)
     - Sharp variation between neighbours
     - :class:`~quadsv.MatrixKernel` with
       ``method="graph_laplacian"``.
   * - Regular rasterised grid (Visium HD, imaging)
     - Smooth, large-scale gradient
     - :class:`~quadsv.FFTKernel` with ``method="car"`` or
       ``method="matern"``. Both have polynomial spectral decay,
       which keeps power on mid- and high-frequency modes.
   * - Small ``n`` (under ~5 000), sanity check
     - Any
     - :class:`~quadsv.MatrixKernel`. The dense matrix path is the
       simplest.

If you don't yet know which pattern type you want, start with CAR or
Matérn. They look for smooth gradients and still report calibrated
p-values when the true signal is sharp, just with lower power.

Code:

.. code-block:: python

   from quadsv import NUFFTKernel, MatrixKernel, FFTKernel

   # Irregular coords, smooth pattern (default)
   kernel = NUFFTKernel(coords, method="matern", bandwidth=25.0, nu=1.5)

   # Irregular coords, graph-flavoured pattern
   kernel = MatrixKernel.from_coordinates(
       coords, method="car", rho=0.9, k_neighbors=4
   )

   # Regular rasterised grid (CAR)
   kernel = FFTKernel(
       shape=(1000, 1000), method="car", rho=0.9, neighbor_degree=1
   )

   # Regular rasterised grid (Matérn)
   kernel = FFTKernel(
       shape=(1000, 1000), method="matern", bandwidth=4.0, nu=1.5
   )

The ``backend`` keyword on :class:`~quadsv.DetectorIrregular` selects
between :class:`~quadsv.NUFFTKernel` and
:class:`~quadsv.MatrixKernel`. See :doc:`/guides/quickstart`.


Picking a method
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 36 44

   * - Kernel ``method``
     - What it captures
     - Spectral decay
   * - ``"gaussian"``
     - Soft, distance-based smooth patterns.
     - **Exponential** in :math:`|\omega|^2`. Drops mid- and
       high-frequency structure under the noise floor quickly.
   * - ``"matern"``
     - Distance-based smooth patterns with tunable smoothness.
     - **Polynomial**, rate :math:`-(2\nu + d)` in :math:`|\omega|`.
       ``nu`` controls smoothness, ``bandwidth`` controls the
       cut-off frequency. ``nu=1.5`` is a common starting point.
   * - ``"car"``
     - Graph-based smooth patterns.
     - **Polynomial**, rate :math:`-2` in :math:`|\omega|` near
       the cut-off. ``rho`` close to ``1`` pushes the cut-off
       lower in frequency. Strictly positive definite for any
       ``rho < 1``. Supports the FFT path on regular grids.
   * - ``"graph_laplacian"``
     - Graph-based local variation: sharp differences between
       neighbouring spots.
     - High-pass; spectrum *grows* like :math:`|\omega|^2` at low
       frequency. Pairs with CAR. Use when you care about
       boundaries or textures rather than smooth gradients.
   * - ``"moran"``
     - Classical autocorrelation.
     - Indefinite spectrum, suffers from spectral cancellation.
       Avoid for SVG detection. Available for backwards comparison
       with legacy methods. See :doc:`/guides/theory` Theorem 2.

.. dropdown:: Why positive definiteness matters

   The Q-test power for a pattern :math:`f` is

   .. math::

      Q(f) = \sum_{\omega} |\hat{f}_\omega|^2 \, \lambda(\omega),

   where :math:`\lambda(\omega)` is the kernel's spectrum. If some
   eigenvalues :math:`\lambda(\omega)` are negative, contributions
   from different frequencies cancel inside the sum and the test
   loses power on composite patterns. A strictly positive spectrum
   guarantees no cancellation. See :doc:`/guides/theory` (Theorem 2)
   for the formal statement and proof. CAR is the smallest
   modification of Moran's I that fixes the problem:
   :math:`\mathbf{K}_{\text{CAR}} = (\mathbf{I} - \rho \tilde{\mathbf{W}})^{-1}`
   has every eigenvalue strictly positive for
   :math:`0 < \rho < 1`.


.. _spectral-decay-rate:

Spectral decay rate
-------------------

The kernel spectrum :math:`\lambda(\omega)` is a frequency filter:
:math:`Q(f) = \sum_\omega |\hat{f}_\omega|^2 \lambda(\omega)`. How
:math:`\lambda` falls off at large :math:`|\omega|` decides which
spatial frequencies the test still has power for.

Two regimes show up across the kernels in this library:

- **Exponential decay** (Gaussian).
  :math:`\lambda(\omega) \propto \exp(-\sigma^2 |\omega|^2 / 2)`,
  where :math:`\sigma` is the bandwidth. Power on a frequency
  :math:`\omega` collapses to numerical zero once
  :math:`|\omega| \gtrsim 1/\sigma`. Anything finer than that
  scale is invisible to the test.
- **Polynomial decay** (Matérn, CAR). Power falls off as a
  fixed power of :math:`|\omega|`, so even modes well past the
  cut-off keep some weight. The test loses power gradually rather
  than abruptly.

For dense rasterised data (Visium HD, imaging), interesting biology
often shows up at fine scales, so the polynomial tail of Matérn or
CAR usually wins over Gaussian even when the dominant signal is
smooth.

How decay rates depend on hyper-parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - Kernel
     - Spectrum (large :math:`|\omega|`, dimension :math:`d`)
     - Hyper-parameters
   * - Gaussian
     - :math:`\lambda(\omega) \propto e^{-\sigma^{2}|\omega|^{2}/2}`
     - ``bandwidth`` :math:`\sigma` sets the cut-off frequency
       :math:`\sim 1/\sigma`. There is no smoothness knob; decay
       is always exponential in :math:`|\omega|^{2}`.
   * - Matérn
     - :math:`\lambda(\omega) \propto |\omega|^{-(2\nu + d)}`
     - ``nu`` :math:`\nu` controls the polynomial decay rate
       (larger :math:`\nu` = faster decay = smoother).
       ``bandwidth`` :math:`\sigma` sets the cut-off
       :math:`\sim 1/\sigma`. ``nu=1.5`` in 2-D gives rate
       :math:`-5`; ``nu=0.5`` (exponential covariance) gives rate
       :math:`-3`; ``nu`` :math:`\to \infty` recovers Gaussian.
   * - CAR (regular grid)
     - :math:`\lambda(\omega) \propto |\omega|^{-2}` past the
       cut-off
     - ``rho`` :math:`\rho` controls the cut-off frequency
       :math:`\sim \sqrt{(1-\rho)/\rho}`. ``rho`` close to ``1``
       places the cut-off near DC, which puts most weight on
       large-scale structure but still keeps the
       :math:`|\omega|^{-2}` tail. ``neighbor_degree`` shifts the
       very-high-frequency behaviour through the adjacency
       symbol, but the asymptotic rate stays
       :math:`|\omega|^{-2}`.
   * - Graph Laplacian (regular grid)
     - :math:`\lambda(\omega) \propto |\omega|^{2}` at low
       :math:`|\omega|`
     - High-pass with no smoothing knob. ``neighbor_degree``
       controls how local "local" is.

.. dropdown:: Where these decay rates come from

   On a regular 2-D grid in :math:`d` dimensions, all four kernels
   are diagonalised by the Fourier basis. The spectra come out as:

   .. math::

      \lambda_{\text{Gaussian}}(\omega)
      &\propto \exp\!\bigl(-\tfrac{1}{2}\sigma^{2}|\omega|^{2}\bigr),\\
      \lambda_{\text{Matérn}}(\omega)
      &\propto \bigl(2\nu / \sigma^{2} + |\omega|^{2}\bigr)^{-(\nu + d/2)},\\
      \lambda_{\text{CAR}}(\omega)
      &= \bigl(1 - \rho\,\mu(\omega)\bigr)^{-1},\\
      \lambda_{\text{Lap}}(\omega)
      &= 1 - \mu(\omega),

   where :math:`\mu(\omega)` is the row-normalised adjacency
   symbol (e.g. :math:`(\cos\omega_x + \cos\omega_y)/2` for a
   4-NN square grid). Expanding around :math:`|\omega| = 0`,
   :math:`1 - \mu(\omega) \approx \tfrac{1}{4}|\omega|^{2}`, which
   gives CAR its :math:`|\omega|^{-2}` tail and the graph
   Laplacian its :math:`|\omega|^{2}` rise. Matérn's
   :math:`|\omega|^{-(2\nu+d)}` follows directly from its
   Bochner-theorem spectral density.

   On irregular coordinates the same shapes hold for
   :class:`~quadsv.NUFFTKernel`, with an oversampled auxiliary
   grid in place of the data points. See :doc:`/guides/scaling`
   for the NUFFT operator definition and its analytic moments.


Tuning the hyper-parameters
---------------------------

**Bandwidth** (Gaussian, Matérn). A reasonable starting value is
the median pairwise distance:

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import pdist

   bandwidth0 = float(np.median(pdist(coords)))
   kernel = NUFFTKernel(coords, method="matern", bandwidth=bandwidth0, nu=1.5)

**ρ** (CAR). Larger ``rho`` means stronger smoothing.

- ``rho = 0.5``: moderate smoothing.
- ``rho = 0.9``: strong smoothing (the default).
- ``rho`` close to ``1``: maximum smoothing. The spectrum becomes
  very heavy-tailed.

**k_neighbors** (graph kernels). Larger ``k`` means a denser graph
and more global patterns.

- ``k`` between 5 and 10: sparse, local.
- ``k = 30`` or more: dense, global.

If you sweep a small grid of values, look at how :math:`Q` and the
trace move with the parameter. Neither metric replaces a held-out
validation set, but together they highlight runs where the kernel
collapses onto a single mode.


Custom kernels
--------------

Two extension points cover almost everything: a custom matrix and a
custom subclass.

**1. Custom adjacency or distance matrix**

.. code-block:: python

   import numpy as np
   from quadsv.kernels import MatrixKernel

   # Build a CAR precision from a custom adjacency
   W = ...                                        # (n, n) adjacency
   precision = np.eye(W.shape[0]) - 0.9 * W       # I - ρW
   kernel = MatrixKernel.from_matrix(
       precision, method="car", is_precision=True
   )

**2. Subclass an ABC from** ``quadsv.kernels``

Backend authors get two extension points, both in
:mod:`quadsv.kernels`:

- :class:`quadsv.kernels.Kernel` is the universal ABC. Subclass it
  for any custom kernel that you can express through a single
  ``self._K`` buffer.
- :class:`quadsv.kernels.MatrixKernelBase` is the matrix-family
  base used by :class:`~quadsv.MatrixKernel`. Subclass it if you
  need the dense / sparse / sparse-precision auto-switching
  machinery for a new matrix backend.

Neither ABC is re-exported from the top-level ``quadsv``
namespace. Always import them through ``quadsv.kernels``.

The :class:`~quadsv.kernels.Kernel` ABC ships default
implementations of :meth:`~quadsv.kernels.Kernel.Kx`,
:meth:`~quadsv.kernels.Kernel.xtKx`,
:meth:`~quadsv.kernels.Kernel.xtKy`,
:meth:`~quadsv.kernels.Kernel.trace`,
:meth:`~quadsv.kernels.Kernel.square_trace`, and
:meth:`~quadsv.kernels.Kernel.eigenvalues` that all read off the
single ``self._K`` buffer. Override
:meth:`~quadsv.kernels.Kernel._build_kernel` to plug in your own
matrix:

.. code-block:: python

   import numpy as np
   from quadsv.kernels import Kernel

   class MyKernel(Kernel):
       def _build_kernel(self):
           # Return any (n, n) symmetric PSD matrix.
           return self.params["K"]

   K = np.eye(100)
   kernel = MyKernel(n=100, method="custom", K=K)

Override :meth:`~quadsv.kernels.Kernel.Kx` only if you have a
faster operator than ``K @ x``. That is what the FFT and NUFFT
backends do.

.. dropdown:: Subclassing FFTKernel for a custom spectrum

   For a regular grid, override
   :meth:`~quadsv.kernels.fft.FFTKernel._compute_eigenvalues` to
   define a custom spectral filter:

   .. code-block:: python

      import numpy as np
      import scipy.fft
      from quadsv.kernels.fft import FFTKernel

      class CustomFFTKernel(FFTKernel):
          def _compute_eigenvalues(self):
              fy = scipy.fft.fftfreq(self.ny, d=self.dy)
              fx = scipy.fft.fftfreq(self.nx, d=self.dx)
              FY, FX = np.meshgrid(fy, fx, indexing="ij")
              r2 = FX ** 2 + FY ** 2
              lam = 1.0 / (1.0 + 10.0 * r2)        # polynomial low-pass
              if self.fft_solver == "rfft2":
                  lam = lam[:, : (self.nx // 2 + 1)]
              return lam.ravel()

   Keep :math:`\lambda(\omega) > 0` for every frequency to preserve
   test consistency.


Inspecting a kernel's spectrum
------------------------------

.. dropdown:: Plot eigenvalues to verify positivity

   .. code-block:: python

      import numpy as np
      import matplotlib.pyplot as plt

      evals = kernel.eigenvalues(k=100)
      print(f"Min eigenvalue: {evals[-1]:.3e}")
      print(f"All positive  : {np.all(evals > 1e-9)}")

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
      ax1.semilogy(evals, "o-"); ax1.set_title("log scale")
      ax2.plot(evals, "o-"); ax2.axhline(0, color="red", ls="--")
      ax2.set_title("linear scale")
      plt.tight_layout(); plt.show()

   How to read the plot:

   - All ``evals > 0``: strictly positive definite, so the test is
     consistent.
   - Some ``evals <= 0``: indefinite, so the test risks spectral
     cancellation.
   - Monotone decay: the kernel emphasises low frequencies.
   - Power-law decay: a heavy tail at high frequencies (CAR-like).


See also
--------

- :doc:`/guides/quickstart` for practical recipes.
- :doc:`/guides/theory` for derivations and proofs.
- :doc:`/guides/scaling` for how kernel choice affects runtime and
  memory.
- :doc:`/autoapi/quadsv/kernels/index` for the kernel API reference.
