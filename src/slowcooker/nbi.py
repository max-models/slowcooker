r"""Neutral beam heating, fuelling and current from freeofcharge, as a TORAX source.

freeofcharge deposits the beams in the equilibrium, slows the fast ions down
on the flux surfaces and returns the heating of the electrons and ions, the
fast-ion birth rate (the fuelling) and the beam-driven current, all over the
toroidal flux radius $\rho$. `nbi_source` turns such a result into TORAX's
prescribed ``nbi`` source; `NBICoupling` recomputes it from the plasma at the
start of every segment of `slowcooker.run_coupled`.

freeofcharge is not JAX code and takes seconds per call, so it runs between
TORAX's segments, not inside its compiled time step: the beam profiles are
held over a segment and follow the plasma from one segment to the next.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr

from slowcooker._profiling import PROFILE_TAGS
from slowcooker.config import profile

__all__ = ["NBICoupling", "nbi_source", "plasma_from_state", "toroidal_radius"]

KEV = 1.0e3
"""eV per keV."""


def toroidal_radius(equilibrium: Any, rho, shells: int = 64) -> np.ndarray:
    r"""Return $\rho_\mathrm{tor} = \sqrt{\Phi/\Phi_b}$ at the radii ``rho`` of an equilibrium.

    For an equilibrium whose radius is the poloidal one
    (``radial_coordinate == "poloidal"``, freeofcharge's axisymmetric
    `Equilibrium`), $\Phi(\psi) = \int q\, d\psi$ with $\psi_n = \rho_\mathrm{pol}^2$
    and $q$ from ``flux_surfaces`` on ``shells`` shells (extended linearly to the
    axis and the edge); otherwise ``rho`` is returned as it is.
    """
    rho = np.asarray(rho, dtype=float)
    if getattr(equilibrium, "radial_coordinate", "toroidal") != "poloidal":
        return rho
    surfaces = equilibrium.flux_surfaces(np.linspace(0.0, 1.0, shells + 1))
    psi = surfaces.rho**2
    q = np.asarray(surfaces.safety_factor, dtype=float)
    q_axis = q[0] - psi[0] * (q[1] - q[0]) / (psi[1] - psi[0])
    q_edge = q[-1] + (1.0 - psi[-1]) * (q[-1] - q[-2]) / (psi[-1] - psi[-2])
    nodes = np.concatenate([[0.0], psi, [1.0]])
    values = np.concatenate([[q_axis], q, [q_edge]])
    flux = np.concatenate(
        [[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(nodes))]
    )
    return np.sqrt(np.interp(rho**2, nodes, flux / flux[-1]))


def plasma_from_state(state: xr.Dataset, radial_coordinate: str = "toroidal") -> Any:
    """Return freeofcharge's `Plasma` from slowcooker profiles at one time.

    The main ion is TORAX's (mass number from ``A_i``, charge 1); the impurities
    are one species of the average charge, rounded, with the main-ion density
    set so that the plasma is quasi-neutral.

    Args:
        state: `slowcooker.snapshot` of a TORAX output (over the toroidal
            radius), or that state interpolated to other toroidal radii with the
            radii the `Plasma` is to use in ``state.attrs["grid"]``.
        radial_coordinate: Which flux the `Plasma`'s radius is the root of.

    Returns:
        A ``freeofcharge.Plasma``.
    """
    import freeofcharge as foc

    rho = state["rho"].values
    n_e = state["n_e"].values
    ions = {}
    n_impurity = state["n_impurity_total"].values
    z_impurity = int(round(float(np.nanmean(state["Z_impurity_average"].values))))
    a_impurity = int(round(float(state["A_impurity"].mean())))
    if np.any(n_impurity > 0.0) and z_impurity > 0:
        ions["impurity"] = foc.IonSpecies(a_impurity, z_impurity, n_impurity)
    main = np.maximum(n_e - z_impurity * n_impurity, 0.0)
    ions["main"] = foc.IonSpecies(int(round(float(state["A_i"]))), 1, main)
    return foc.Plasma(
        state.attrs.get("grid", rho),
        n_e,
        state["T_e"].values * KEV,
        state["T_i"].values * KEV,
        ions=ions,
        radial_coordinate=radial_coordinate,
    )


def nbi_source(
    nbi: xr.Dataset | Mapping[float, xr.Dataset],
    *,
    current: bool = True,
    fuelling: bool = True,
) -> dict[str, Any]:
    """Return TORAX's prescribed ``nbi`` source for freeofcharge results.

    Args:
        nbi: The result of ``freeofcharge.run.run_nbi`` (over its shell centres
            ``rho``), or results by time [s]; TORAX interpolates between them.
        current: Include the beam-driven current (where it is finite).
        fuelling: Include the fast-ion births as an electron source.

    Returns:
        The ``sources`` section with the ``nbi`` entry, for `slowcooker.merge`.
        TORAX interpolates the profiles onto its cells, so the power it deposits
        equals freeofcharge's to the interpolation error (a few percent for noisy
        profiles on a coarse TORAX grid).
    """
    results = {0.0: nbi} if isinstance(nbi, xr.Dataset) else nbi
    channels: dict[str, dict] = {"power_density_e": {}, "power_density_i": {}}
    if fuelling:
        channels["particle_source_e"] = {}
    if current:
        channels["j_parallel"] = {}
    names = {
        "power_density_e": "electron_heating",
        "power_density_i": "ion_heating",
        "particle_source_e": "birth_density",
        "j_parallel": "current_density",
    }
    for time, result in sorted(results.items()):
        rho = result["rho"].values
        for channel in channels:
            values = np.nan_to_num(np.asarray(result[names[channel]].values, float))
            channels[channel].update(profile(rho, values, time))
    return {
        "sources": {
            "nbi": {
                "model_name": "prescribed",
                "mode": "prescribed",
                "prescribed_values": channels,
            }
        }
    }


@dataclass
class NBICoupling:
    """Recompute the beams with freeofcharge from the plasma before each segment.

    Args:
        equilibrium: The ``freeofcharge`` equilibrium of TORAX's geometry: a
            3D one on the toroidal flux radius (e.g. a ``GriddedEquilibrium`` of
            a GVEC one), or an axisymmetric `Equilibrium` on the poloidal one,
            whose profiles are mapped between the two radii (`toroidal_radius`).
        beams: The beams, as ``freeofcharge.run.run_nbi`` takes them.
        rho_edges: Shell edges of the deposition.
        options: Further arguments of ``run_nbi`` (``markers``, ``seed``, ...).
        current: Pass the beam-driven current to TORAX.
        fuelling: Pass the fast-ion births to TORAX as fuelling.
    """

    equilibrium: Any
    beams: Any
    rho_edges: Any = field(default_factory=lambda: np.linspace(0.0, 1.0, 26))
    options: Mapping[str, Any] = field(default_factory=dict)
    current: bool = True
    fuelling: bool = True
    results: dict[float, xr.Dataset] = field(default_factory=dict)
    """The freeofcharge result of every segment, by its start time."""

    @sp.profile("slowcooker.nbi", tags=PROFILE_TAGS)
    def __call__(self, time: float, state: xr.Dataset) -> dict[str, Any]:
        """Return the ``nbi`` source for the segment starting at ``time``."""
        from freeofcharge.run import run_nbi

        options = {"torque": False, **self.options}
        coordinate = getattr(self.equilibrium, "radial_coordinate", "toroidal")
        if coordinate == "poloidal":
            grid = np.linspace(0.0, 1.0, 51)
            mapped = state.interp(rho=toroidal_radius(self.equilibrium, grid))
            mapped.attrs["grid"] = grid
            plasma = plasma_from_state(mapped, radial_coordinate="poloidal")
        else:
            plasma = plasma_from_state(state)
        result = run_nbi(
            self.equilibrium,
            plasma,
            self.beams,
            rho_edges=np.asarray(self.rho_edges, dtype=float),
            **options,
        )
        if coordinate == "poloidal":
            result = result.assign_coords(
                rho=toroidal_radius(self.equilibrium, result["rho"].values)
            )
        self.results[time] = result
        # constant over the segment: the same profile at its start and much later
        return nbi_source(
            {time: result, time + 1.0e6: result},
            current=self.current,
            fuelling=self.fuelling,
        )
