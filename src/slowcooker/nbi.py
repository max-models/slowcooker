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

__all__ = ["NBICoupling", "nbi_source", "plasma_from_state"]

KEV = 1.0e3
"""eV per keV."""


def plasma_from_state(state: xr.Dataset) -> Any:
    """Return freeofcharge's `Plasma` from slowcooker profiles at one time.

    The main ion is TORAX's (mass number from ``A_i``, charge 1); the impurities
    are one species of the average charge, rounded, with the main-ion density
    set so that the plasma is quasi-neutral.

    Args:
        state: `slowcooker.snapshot` of a TORAX output.

    Returns:
        A ``freeofcharge.Plasma`` over the toroidal flux radius.
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
        rho,
        n_e,
        state["T_e"].values * KEV,
        state["T_i"].values * KEV,
        ions=ions,
        radial_coordinate="toroidal",
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
        equilibrium: The ``freeofcharge`` equilibrium (e.g. a
            ``GriddedEquilibrium`` of a GVEC one) on the same toroidal flux
            radius as TORAX's geometry.
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
        result = run_nbi(
            self.equilibrium,
            plasma_from_state(state),
            self.beams,
            rho_edges=np.asarray(self.rho_edges, dtype=float),
            **options,
        )
        self.results[time] = result
        # constant over the segment: the same profile at its start and much later
        return nbi_source(
            {time: result, time + 1.0e6: result},
            current=self.current,
            fuelling=self.fuelling,
        )
