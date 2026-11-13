"""Equilibria that come with slowcooker, for the tests and the examples.

Two GVEC equilibria, each a parameter file and its final state:

- ``"w7x"``: the W7-X standard configuration of GVEC's examples (five field
  periods, nearly zero toroidal current), solved at reduced resolution;
- ``"ellipse"``: a small rotating ellipse with two field periods.

`equilibrium_files` returns their paths; `gvec_equilibrium` loads one as a
``freeofcharge`` equilibrium (needs the ``gvec`` extra).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

__all__ = ["equilibrium_files", "gvec_equilibrium"]

_NAMES = ("w7x", "ellipse")


def equilibrium_files(name: str = "w7x") -> tuple[Path, Path]:
    """Return the GVEC parameter file and state file of a bundled equilibrium.

    Args:
        name: ``"w7x"`` or ``"ellipse"``.

    Returns:
        The paths (parameter file, state file).
    """
    if name not in _NAMES:
        raise ValueError(f"unknown equilibrium {name!r}; one of {_NAMES}")
    folder = Path(str(files("slowcooker") / "data"))
    return folder / f"{name}_parameters.ini", folder / f"{name}_state.dat"


def gvec_equilibrium(name: str = "w7x", **options: Any) -> Any:
    """Return a bundled equilibrium as a ``freeofcharge`` GVEC equilibrium.

    Args:
        name: ``"w7x"`` or ``"ellipse"``.
        **options: Passed to ``GVECEquilibrium``.

    Returns:
        A ``freeofcharge.gvec_equilibrium.GVECEquilibrium``.
    """
    from freeofcharge.gvec_equilibrium import GVECEquilibrium

    return GVECEquilibrium.from_files(*equilibrium_files(name), **options)
