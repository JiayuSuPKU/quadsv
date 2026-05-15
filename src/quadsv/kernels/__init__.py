"""
``quadsv.kernels`` — spatial-kernel layer.

Subpackage grouping the public kernel classes plus the ABCs that
backend authors subclass:

- :class:`Kernel` (ABC) — universal interface.
- :class:`MatrixKernelBase` (ABC) — matrix-form base with dense /
  sparse / sparse-precision auto-switching.
- :class:`MatrixKernel` — standard concrete matrix kernel.
- :class:`FFTKernel` — regular-grid FFT-accelerated kernel.
- :class:`NUFFTKernel` — irregular-coordinate NUFFT-accelerated
  kernel.

All five are importable from this subpackage:

    from quadsv.kernels import FFTKernel, NUFFTKernel, MatrixKernel
    from quadsv.kernels import Kernel, MatrixKernelBase  # for backend authors

The three concrete classes are also re-exported at the top of the
:mod:`quadsv` namespace.
"""

from quadsv.kernels.base import Kernel, MatrixKernelBase
from quadsv.kernels.fft import FFTKernel
from quadsv.kernels.matrix import MatrixKernel
from quadsv.kernels.nufft import NUFFTKernel

__all__ = [
    "Kernel",
    "MatrixKernelBase",
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
]
