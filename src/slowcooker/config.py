"""TORAX configs for slowcooker: a base case on any geometry, and merging overrides.

A TORAX config is a nested dictionary (``torax.ToraxConfig.from_dict``).
`base_config` writes one for a geometry from `slowcooker.geometry`: the
rotational transform held fixed (``psi_source = "geometry"``, current diffusion
off, the right setting for a stellarator without much plasma current), the
temperatures evolved, the density fixed, constant diffusivities, and a central
heating. Everything else is TORAX's default. Change any part with `merge`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from slowcooker.geometry import geometry_config

__all__ = ["base_config", "merge", "profile"]


def profile(rho, values, time: float = 0.0) -> dict[float, dict[float, float]]:
    """Return a profile at one time in TORAX's form ``{time: {rho: value}}``.

    Args:
        rho: Normalised radii.
        values: Values at those radii.
        time: Time [s].

    Returns:
        The nested dictionary TORAX takes for a time-varying array.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    values = np.broadcast_to(np.asarray(values, dtype=float), rho.shape)
    return {float(time): {float(r): float(v) for r, v in zip(rho, values, strict=True)}}


def merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base`` with ``overrides`` merged in.

    Nested dictionaries merge key by key; any other value replaces the one in
    ``base``, and ``None`` removes the key.

    Args:
        base: The config to start from.
        overrides: The changes.

    Returns:
        The merged config; neither input is modified.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def base_config(
    geometry: xr.Dataset | Mapping[Any, Any],
    *,
    density: float = 3.0e19,
    temperature: tuple[float, float] = (2.0, 0.1),
    heating: float = 2.0e6,
    electron_fraction: float = 0.5,
    chi: float = 1.0,
    t_final: float = 1.0,
    dt: float = 0.02,
    n_rho: int = 25,
) -> dict[str, Any]:
    """Return a TORAX config with fixed rotational transform on a geometry.

    Args:
        geometry: Flux-surface averages (`slowcooker.geometry`), one dataset or
            datasets by time; or a TORAX geometry section (with ``model_name``),
            e.g. from `slowcooker.geometry.eqdsk_geometry`.
        density: Flat electron density [m^-3].
        temperature: Initial central and edge temperatures of electrons and
            ions [keV]; the edge value is also the boundary condition. The
            initial profile is parabolic.
        heating: Total heating power [W], a Gaussian of width 0.25 on the axis.
        electron_fraction: Share of the heating that goes to the electrons.
        chi: Ion and electron heat diffusivities [m^2/s].
        t_final: End time [s].
        dt: Fixed time step [s].
        n_rho: Number of TORAX cells.

    Returns:
        The config dictionary.
    """
    centre, edge = temperature
    rho = np.linspace(0.0, 1.0, 11)
    initial = profile(rho, edge + (centre - edge) * (1.0 - rho**2))
    return {
        "profile_conditions": {
            "I_p_source": "geometry",
            "psi_source": "geometry",
            "T_e": initial,
            "T_i": copy.deepcopy(initial),
            "T_e_right_bc": edge,
            "T_i_right_bc": edge,
            "n_e": profile([0.0, 1.0], density),
            "n_e_right_bc": density,
            "normalize_n_e": False,
        },
        "plasma_composition": {},
        "numerics": {
            "evolve_current": False,
            "evolve_density": False,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
        },
        "geometry": (
            {**geometry, "n_rho": n_rho}
            if isinstance(geometry, Mapping) and "model_name" in geometry
            else geometry_config(geometry, n_rho=n_rho)
        ),
        "sources": {
            "generic_heat": {
                "P_total": heating,
                "gaussian_width": 0.25,
                "gaussian_location": 0.0,
                "electron_heat_fraction": electron_fraction,
            },
            "ei_exchange": {},
        },
        "pedestal": {},
        "transport": {
            "core_transport_models": {
                "constant": {"model_name": "prescribed", "chi_i": chi, "chi_e": chi}
            },
            "chi_min": 0.0,
        },
        "solver": {"solver_type": "linear"},
        "time_step_calculator": {
            "calculator_type": "fixed",
            "dt": dt,
            "t_final": t_final,
        },
    }
