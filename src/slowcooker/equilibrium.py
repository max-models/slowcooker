r"""The half dimension: re-solve the GVEC equilibrium with TORAX's pressure.

As the pressure profile evolves, the flux surfaces shift (the Shafranov shift
of a tokamak, its 3D counterpart in a stellarator) and the geometry factors
change. `EquilibriumCoupling` is a coupling for `slowcooker.run_coupled`:
before each segment it fits TORAX's total pressure $p(\rho)$ with a B-spline in
$s = \rho^2$, re-solves GVEC with it (from the previous state, the boundary and
the rotational transform unchanged), and hands TORAX the new flux-surface
averages.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr

from slowcooker._profiling import PROFILE_TAGS
from slowcooker.geometry import geometry_config

__all__ = ["EquilibriumCoupling", "pressure_parameters"]


def pressure_parameters(
    parameters: Mapping[str, Any], rho, pressure, knots: int = 6
) -> dict[str, Any]:
    r"""Return GVEC parameters with the pressure profile replaced by a fit.

    Args:
        parameters: GVEC parameters (``gvec.util.read_parameters``).
        rho: Radii of the pressure values, from the axis to the edge.
        pressure: Pressure [Pa].
        knots: Interior knots of the cubic B-spline in $s = \rho^2$, at most
            a quarter of the points so the fit stays smooth.

    Returns:
        A copy of the parameters with ``pres_type = "bspline"``.
    """
    from gvec.util.params import bspl2gvec
    from scipy.interpolate import make_lsq_spline

    s = np.asarray(rho, dtype=float) ** 2
    knots = max(1, min(knots, s.size // 4))
    interior = np.linspace(0.0, 1.0, knots + 2)[1:-1]
    t = np.concatenate([[0.0] * 4, interior, [1.0] * 4])
    spline = make_lsq_spline(s, np.maximum(np.asarray(pressure, float), 0.0), t, k=3)
    updated = copy.deepcopy(dict(parameters))
    for key in [k for k in updated if k.lower().startswith("pres_")]:
        updated.pop(key)
    updated.pop("pres", None)
    return dict(bspl2gvec("pres", bspl=spline, params=updated))


@dataclass
class EquilibriumCoupling:
    r"""Re-solve GVEC with TORAX's pressure before each segment of `slowcooker.run_coupled`.

    Args:
        parameters: GVEC parameters of the equilibrium (boundary, rotational
            transform or current, resolution).
        state: The GVEC state to start from (the equilibrium of the first
            segment's geometry).
        rho: Radii of the flux-surface averages (above zero).
        angles: $(\theta, \zeta)$ grid of each surface.
        B_0: Reference field of the geometry [T]; keep the one of the
            initial geometry (`slowcooker.geometry_config`'s ``B_0``).
        workdir: Where GVEC runs; a temporary directory if None.
        iterations: Most GVEC iterations per re-solve.
    """

    parameters: Mapping[str, Any]
    state: Any
    rho: Any = field(default_factory=lambda: np.linspace(0.02, 1.0, 40))
    angles: tuple[int, int] = (32, 32)
    B_0: float | None = None
    workdir: str | Path | None = None
    iterations: int = 5000
    geometries: dict[float, xr.Dataset] = field(default_factory=dict)
    """The flux-surface averages of every re-solve, by the segment's start time."""

    @sp.profile("slowcooker.equilibrium", tags=PROFILE_TAGS)
    def __call__(self, time: float, state: xr.Dataset) -> dict[str, Any]:
        """Re-solve with the pressure of ``state`` and return the new geometry."""
        import gvec
        from freeofcharge.gvec_equilibrium import GVECEquilibrium

        parameters = pressure_parameters(
            self.parameters, state["rho"].values, state["pressure_total"].values
        )
        parameters["totalIter"] = self.iterations
        parameters.pop("maxIter", None)
        parameters.pop("picard_current", None)
        folder = Path(self.workdir or tempfile.mkdtemp(prefix="slowcooker-gvec-"))
        folder.mkdir(parents=True, exist_ok=True)
        result = gvec.run(
            parameters,
            restartstate=self.state,
            runpath=folder / f"t{time:.6g}",
            quiet=True,
        )
        self.state = result.state
        geometry = GVECEquilibrium(result.state).transport_geometry(
            np.asarray(self.rho, dtype=float), angles=self.angles
        )
        self.geometries[time] = geometry
        section = geometry_config(geometry, B_0=self.B_0)
        section.pop("n_rho")
        return {"geometry": section}
