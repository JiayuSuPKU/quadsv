"""Public-API freeze test — long-term guardrail for the four-layer
``quadsv`` surface.

What this test enforces:

1. **Snapshot of ``quadsv.__all__``.** Any addition or removal to the
   top-level exports forces a deliberate edit to ``EXPECTED_ALL``
   below, which surfaces in code review.
2. **Every public name imports + has a docstring.** Catches typos in
   ``__all__`` and missing documentation on new public symbols.
3. **Canonical-path identity.** Top-level re-exports resolve to the
   *same* object as their canonical path (e.g.
   ``quadsv.spatial_q_test is quadsv.statistics.spatial_q_test``).
   Guards against accidental re-export breakage during refactors.
"""

from __future__ import annotations

import importlib

import quadsv

# ---------------------------------------------------------------------------
# Snapshot of the four-layer public surface. Any drift from this list — adds
# or removes — must be a deliberate edit reviewed alongside the change.
# Group order mirrors the package docstring (Kernels → Statistics →
# Detectors → Comparators → Factories).
# ---------------------------------------------------------------------------
EXPECTED_ALL: list[str] = [
    # Kernels
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
    # Factories
    "Detector",
    "Comparator",
]

# ABCs used only by backend authors. They live in ``quadsv.kernels`` and
# are not part of the top-level public surface, so they must not show up
# in ``quadsv.__all__`` or as attributes on the package.
_INTERNAL_BACKEND_ABCS: list[str] = ["Kernel", "MatrixKernelBase"]


def test_top_level_all_matches_snapshot():
    """``quadsv.__all__`` matches ``EXPECTED_ALL`` (set comparison).

    Order doesn't matter; the *set* is the contract. Edit
    ``EXPECTED_ALL`` deliberately when the public surface changes.
    """
    assert set(quadsv.__all__) == set(EXPECTED_ALL), (
        "quadsv.__all__ drifted from the expected snapshot.\n"
        f"  added:   {sorted(set(quadsv.__all__) - set(EXPECTED_ALL))}\n"
        f"  removed: {sorted(set(EXPECTED_ALL) - set(quadsv.__all__))}"
    )


def test_every_public_name_resolves_and_documented():
    """Every name in ``quadsv.__all__`` must resolve and carry a
    non-empty docstring."""
    for name in quadsv.__all__:
        obj = getattr(quadsv, name, None)
        assert obj is not None, f"{name} listed in __all__ but unresolved"
        doc = getattr(obj, "__doc__", None)
        assert doc and doc.strip(), f"{name} has no docstring"


# ---------------------------------------------------------------------------
# Canonical-path identity contract.
#
# Each top-level re-export must point at the same object as the
# canonical submodule path. If the re-export drifts (e.g. somebody
# accidentally rebinds the name in ``quadsv.__init__``), tests still
# import the canonical class but the user-facing shortcut becomes
# stale; this test fails loudly.
# ---------------------------------------------------------------------------
_CANONICAL_PATHS: dict[str, tuple[str, str]] = {
    # name on quadsv: (submodule, attribute on submodule)
    "MatrixKernel": ("quadsv.kernels", "MatrixKernel"),
    "FFTKernel": ("quadsv.kernels.fft", "FFTKernel"),
    "NUFFTKernel": ("quadsv.kernels.nufft", "NUFFTKernel"),
    "spatial_q_test": ("quadsv.statistics", "spatial_q_test"),
    "spatial_r_test": ("quadsv.statistics", "spatial_r_test"),
    "DetectorIrregular": ("quadsv.detectors.irregular", "DetectorIrregular"),
    "DetectorGrid": ("quadsv.detectors.grid", "DetectorGrid"),
    "ComparatorIrregular": ("quadsv.comparators", "ComparatorIrregular"),
    "ComparatorGrid": ("quadsv.comparators", "ComparatorGrid"),
    "Detector": ("quadsv.api", "Detector"),
    "Comparator": ("quadsv.api", "Comparator"),
}


def test_top_level_objects_identity_match_canonical_paths():
    """Every top-level re-export points at the same object as the
    canonical submodule path.
    """
    for name, (modpath, attr) in _CANONICAL_PATHS.items():
        top = getattr(quadsv, name)
        canonical = getattr(importlib.import_module(modpath), attr)
        assert top is canonical, f"quadsv.{name} drifted from {modpath}.{attr}"


def test_backend_abcs_are_not_top_level_public():
    """``Kernel`` and ``MatrixKernelBase`` are extension points for
    backend authors. They live at ``quadsv.kernels`` and must not be
    accessible as top-level attributes on the ``quadsv`` package.
    """
    for name in _INTERNAL_BACKEND_ABCS:
        assert name not in quadsv.__all__, f"{name} should not be in quadsv.__all__"
        assert not hasattr(quadsv, name), (
            f"quadsv.{name} should not be reachable on the top-level package; "
            f"import from quadsv.kernels instead."
        )
    # ...but the canonical path is still importable.
    from quadsv.kernels import Kernel, MatrixKernelBase  # noqa: F401
