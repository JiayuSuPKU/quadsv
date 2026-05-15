"""
Top-level factory entry points: :func:`Detector` and :func:`Comparator`.

Thin one-liner discovery face on top of the four explicit classes
(:class:`DetectorIrregular`, :class:`DetectorGrid`,
:class:`ComparatorIrregular`, :class:`ComparatorGrid`). Dispatches on
the runtime type of the input data so users don't have to know the
``Irregular`` / ``Grid`` split:

>>> det = Detector(adata)             # → DetectorIrregular
>>> det = Detector(sdata)             # → DetectorGrid
>>> cmp = Comparator([adata, ...])    # → ComparatorIrregular
>>> cmp = Comparator([sdata, ...])    # → ComparatorGrid

The factories only check ``isinstance`` to pick the right class, then
forward kwargs verbatim. Asymmetry between the two:

- :func:`Detector` does **not** pass the data argument to the
  constructor — the caller chains ``.setup_data(data)`` afterwards
  (matching the explicit-class flow).
- :func:`Comparator` **does** pass the sample list as the first
  positional argument, since both comparator constructors take
  ``samples`` there. Cross-sample contrasts (``design``) are
  supplied later, at test time, on
  :meth:`~quadsv.ComparatorIrregular.test_diff_freq` /
  :meth:`~quadsv.ComparatorIrregular.test_diff_expr`.

For advanced use (custom kernel selection, sample-list inputs that
mix two backends intentionally) prefer the explicit class names —
the factories deliberately reject mixed-type lists with a
:class:`TypeError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from quadsv.comparators import ComparatorGrid, ComparatorIrregular
from quadsv.detectors.grid import DetectorGrid
from quadsv.detectors.irregular import DetectorIrregular

__all__ = ["Detector", "Comparator"]


def _is_anndata(obj: Any) -> bool:
    """Return True if ``obj`` is an :class:`anndata.AnnData`. Lazy
    import so the factories work even when ``anndata`` isn't
    available — though in practice ``anndata`` is a hard dependency
    of :mod:`quadsv`.
    """
    try:
        from anndata import AnnData
    except ImportError:  # pragma: no cover — anndata is a hard dep
        return False
    return isinstance(obj, AnnData)


def _is_spatialdata(obj: Any) -> bool:
    """Return True if ``obj`` is a :class:`spatialdata.SpatialData`.
    Lazy import so the factories raise a clear :class:`TypeError`
    rather than :class:`ImportError` when ``spatialdata`` isn't
    installed.
    """
    try:
        from spatialdata import SpatialData
    except ImportError:
        return False
    return isinstance(obj, SpatialData)


def _supported_types_msg() -> str:
    return (
        "supported types: anndata.AnnData (→ DetectorIrregular / "
        "ComparatorIrregular) or spatialdata.SpatialData (→ "
        "DetectorGrid / ComparatorGrid)"
    )


def Detector(data: Any, **kwargs: Any) -> Any:  # noqa: N802 - factory mimics class names
    """Construct the right :class:`~quadsv.Detector` for ``data``.

    Dispatches on ``type(data)``:

    - :class:`anndata.AnnData` → :class:`~quadsv.DetectorIrregular`.
    - :class:`spatialdata.SpatialData` → :class:`~quadsv.DetectorGrid`.

    The data itself is **not** passed to the constructor — the caller
    is expected to chain ``.setup_data(data, ...)`` afterwards (the
    factory only uses ``data``'s type to pick a class).

    Parameters
    ----------
    data : anndata.AnnData or spatialdata.SpatialData
        The dataset whose type drives the dispatch.
    **kwargs
        Forwarded to the chosen class's ``__init__`` verbatim.

    Returns
    -------
    DetectorIrregular or DetectorGrid
        Constructed (but not yet set up) detector instance.

    Raises
    ------
    TypeError
        If ``data`` is neither an ``AnnData`` nor a ``SpatialData``.

    Examples
    --------
    >>> from quadsv import Detector
    >>> det = Detector(adata, kernel_method="gaussian", backend="matrix")
    >>> det = det.setup_data(adata)
    >>> df = det.compute_qstat()
    """
    if _is_anndata(data):
        return DetectorIrregular(**kwargs)
    if _is_spatialdata(data):
        return DetectorGrid(**kwargs)
    raise TypeError(
        f"Detector cannot dispatch on type {type(data).__name__!r}; " f"{_supported_types_msg()}."
    )


def Comparator(  # noqa: N802 - factory mimics class names
    data_list: Sequence[Any], **kwargs: Any
) -> Any:
    """Construct the right :class:`~quadsv.Comparator` for ``data_list``.

    Dispatches on the homogeneous element type:

    - all :class:`anndata.AnnData` → :class:`~quadsv.ComparatorIrregular`.
    - all :class:`spatialdata.SpatialData` → :class:`~quadsv.ComparatorGrid`.
    - mixed types → :class:`TypeError`.

    Unlike :func:`Detector`, the data list **is** forwarded as the
    first positional arg to the chosen class (both comparator
    constructors take ``samples`` as their first positional
    parameter).

    Parameters
    ----------
    data_list : sequence of anndata.AnnData or sequence of spatialdata.SpatialData
        Per-sample inputs. Must all be of the same type.
    **kwargs
        Forwarded to the chosen class's ``__init__`` verbatim
        (e.g. ``gene_names=...``, ``feature_mode=...``, etc.). The
        cross-sample contrast (``design``) is supplied later on
        :meth:`test_diff_freq` / :meth:`test_diff_expr`, not here.

    Returns
    -------
    ComparatorIrregular or ComparatorGrid
        Constructed comparator instance.

    Raises
    ------
    TypeError
        If ``data_list`` is empty or its elements aren't all the
        same supported type.

    Examples
    --------
    >>> from quadsv import Comparator
    >>> cmp = Comparator([a1, a2, a3]).compute_spectra()
    >>> df = cmp.test_diff_freq(group_labels)
    """
    items = list(data_list)
    if len(items) == 0:
        raise TypeError(
            f"Comparator requires a non-empty list of samples; " f"{_supported_types_msg()}."
        )
    all_anndata = all(_is_anndata(x) for x in items)
    all_spatialdata = all(_is_spatialdata(x) for x in items)
    if all_anndata:
        return ComparatorIrregular(items, **kwargs)
    if all_spatialdata:
        return ComparatorGrid(items, **kwargs)
    raise TypeError(
        f"Comparator received a list of mixed / unsupported types "
        f"{[type(x).__name__ for x in items]}; {_supported_types_msg()}."
    )
