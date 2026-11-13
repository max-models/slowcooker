r"""Geometry for TORAX from flux-surface averages, tokamak or stellarator.

TORAX reads its geometry from equilibrium codes in a few formats. The CHEASE
format is a table of flux-surface averages over the poloidal flux $\psi$, and
TORAX takes it in memory (``chease_data``), so any equilibrium whose averages we
can compute goes in without changing TORAX. `chease_columns` turns the averages
of `freeofcharge.surface_geometry` (over $\rho = \sqrt{\Phi/\Phi_b}$) into those
columns.

Two things differ from a tokamak equilibrium:

- the poloidal flux comes from the rotational transform,
  $\psi(\rho) = \int_0^\rho \iota\, d\Phi = 2\Phi_b \int_0^\rho \iota \rho'\, d\rho'$,
  so TORAX's $q = d\Phi/d\psi$ is $1/\iota$;
- TORAX builds $V' = dV/d\rho$ from the tokamak identity
  $V' = 4\pi\Phi_b\rho / (F \langle 1/R^2\rangle)$, so the column for
  $F = R B_\varphi$ holds the effective value
  $F_\mathrm{eff} = 4\pi\Phi_b\rho / (V' \langle 1/R^2\rangle)$ that makes it give
  the true $V'$; in a tokamak $F_\mathrm{eff} = F$.

With these, TORAX's metric coefficients come out as
$g_0 = V'\langle|\nabla\rho|\rangle$, $g_1 = V'^2\langle|\nabla\rho|^2\rangle$,
$g_2 = V'^2\langle|\nabla\rho|^2/R^2\rangle$ and $g_3 = \langle 1/R^2\rangle$,
whatever $\iota$ is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr

from slowcooker._profiling import PROFILE_TAGS

__all__ = [
    "chease_columns",
    "circular_geometry",
    "eqdsk_geometry",
    "geometry_config",
    "gvec_geometry",
]

MU_0 = 4.0e-7 * np.pi
MIN_CURRENT = 1.0
"""Smallest total toroidal current [A] passed to TORAX, which divides by it; a
current-free equilibrium (a stellarator) gets this much, spread as $\\rho^2$."""


def _poloidal_flux(rho, iota, toroidal_flux_edge):
    r"""Return $\psi(\rho) = 2\Phi_b \int_0^\rho \iota\rho'\,d\rho'$ by the trapezoidal rule.

    The integrand vanishes on the axis, so the first interval runs from 0.
    """
    integrand = 2.0 * toroidal_flux_edge * iota * rho
    nodes = np.concatenate([[0.0], rho])
    values = np.concatenate([[0.0], integrand])
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(nodes)
    return np.cumsum(steps)


def chease_columns(
    geometry: xr.Dataset, *, B_0: float | None = None
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    r"""Return TORAX's CHEASE columns and the normalisation of a geometry dataset.

    Args:
        geometry: Flux-surface averages over $\rho = \sqrt{\Phi/\Phi_b}$ from
            `freeofcharge.surface_geometry` (or
            `freeofcharge.gvec_equilibrium.GVECEquilibrium.transport_geometry`),
            with ``rotational_transform`` and the attribute
            ``toroidal_flux_edge`` (GVEC's $\Phi_b / 2\pi$, in Wb). All radii
            above zero; TORAX adds the axis.
        B_0: Reference field [T]; by default $\langle B^2\rangle^{1/2}$ of the
            innermost surface.

    Returns:
        The columns (``psi_chease``, ``ip_profile``, ... as in
        ``torax._src.geometry.chease.CheaseData``) and the scalars ``R_major``,
        ``a_minor`` and ``B_0``.
    """
    if "rotational_transform" not in geometry:
        raise ValueError("the geometry needs the rotational transform")
    rho = geometry["rho"].values
    phi_b = 2.0 * np.pi * float(geometry.attrs["toroidal_flux_edge"])
    iota = np.abs(geometry["rotational_transform"].values)
    vpr = geometry["volume_derivative"].values
    inv_r2 = geometry["inv_R2"].values
    r_in = geometry["R_in"].values
    r_out = geometry["R_out"].values
    r_major = 0.5 * float(r_in[-1] + r_out[-1])
    a_minor = 0.5 * float(r_out[-1] - r_in[-1])
    b_0 = float(np.sqrt(geometry["B2"].values[0])) if B_0 is None else float(B_0)

    psi = _poloidal_flux(rho, iota, phi_b)
    dpsi = 2.0 * phi_b * iota * rho
    toroidal_flux = phi_b * rho**2
    if "toroidal_current" in geometry:
        current = np.abs(geometry["toroidal_current"].values)
    else:
        current = np.zeros_like(rho)
    if current[-1] < MIN_CURRENT:
        # TORAX scales psi by the total current; 0/0 for a current-free plasma
        current = current + MIN_CURRENT * rho**2
    f_eff = 4.0 * np.pi * phi_b * rho / (vpr * inv_r2)
    psi_unit = r_major**2 * b_0 * 2.0 * np.pi
    columns = {
        "psi_chease": psi / psi_unit,
        "ip_profile": current * MU_0 / (r_major * b_0),
        "rho_tor": np.sqrt(toroidal_flux / (np.pi * b_0)) / r_major,
        "r_inboard": r_in / r_major,
        "r_outboard": r_out / r_major,
        "t_rbphi": f_eff / (r_major * b_0),
        "int_dl_over_bp": vpr / dpsi / (r_major / b_0),
        "inv_r": geometry["inv_R"].values * r_major,
        "inv_r2": inv_r2 * r_major**2,
        "bp2": dpsi**2
        * geometry["grad_rho2_over_R2"].values
        / (4.0 * np.pi**2 * b_0**2),
        "grad_psi": dpsi * geometry["grad_rho"].values / (psi_unit / r_major),
        "grad_psi2": dpsi**2 * geometry["grad_rho2"].values / (psi_unit / r_major) ** 2,
        "b2": geometry["B2"].values / b_0**2,
        "inv_b2": geometry["inv_B2"].values * b_0**2,
        "delta_upper": geometry["triangularity_upper"].values,
        "delta_bottom": geometry["triangularity_lower"].values,
        "elongation": geometry["elongation"].values,
        "ftrap": geometry["trapped_fraction"].values,
    }
    scalars = {"R_major": r_major, "a_minor": a_minor, "B_0": b_0}
    return columns, scalars


@sp.profile("slowcooker.geometry", tags=PROFILE_TAGS)
def geometry_config(
    geometry: xr.Dataset | Mapping[float, xr.Dataset],
    *,
    n_rho: int = 25,
    B_0: float | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Return the ``geometry`` section of a TORAX config for flux-surface averages.

    Args:
        geometry: One dataset (fixed geometry) or datasets by time [s] (TORAX
            interpolates between them), see `chease_columns`.
        n_rho: Number of TORAX cells.
        B_0: Reference field [T], by default that of the first dataset.
        **options: Further fields of TORAX's CHEASE geometry config, such as
            ``hires_factor``.

    Returns:
        The config section: TORAX's CHEASE model with the columns in memory and
        the trapped fraction taken from the columns.
    """
    if isinstance(geometry, xr.Dataset):
        columns, scalars = chease_columns(geometry, B_0=B_0)
        data: Any = columns
    else:
        times = sorted(geometry)
        _, scalars = chease_columns(geometry[times[0]], B_0=B_0)
        data = {
            float(t): chease_columns(geometry[t], B_0=scalars["B_0"])[0] for t in times
        }
    return {
        "model_name": "chease",
        "chease_data": data,
        "n_rho": n_rho,
        "trapped_fraction_source": "file",
        **scalars,
        **options,
    }


def circular_geometry(
    major_radius: float = 3.0,
    minor_radius: float = 1.0,
    field: float = 2.5,
    rotational_transform: Any = 0.5,
    *,
    rho: Any = None,
    angles: tuple[int, int] = (64, 4),
    toroidal_current: Any = None,
) -> xr.Dataset:
    r"""Return the flux-surface averages of a circular, concentric tokamak.

    Surfaces $R = R_0 + a\rho\cos\theta$, $Z = a\rho\sin\theta$, field
    $|B| = B_0 R_0 / R$ and toroidal flux $\Phi_b = \pi a^2 B_0$. The analytic test
    case of the geometry chain (no equilibrium solver involved).

    Args:
        major_radius: $R_0$ [m].
        minor_radius: $a$ [m].
        field: $B_0$ on the axis [T].
        rotational_transform: $\iota$, a number or one value per radius.
        rho: Radii, above zero; by default 50 from 0.02 to 1.
        angles: $(\theta, \zeta)$ grid of each surface.
        toroidal_current: Enclosed toroidal current [A], optional.

    Returns:
        The dataset of `freeofcharge.surface_geometry`.
    """
    from freeofcharge import surface_geometry

    rho = np.linspace(0.02, 1.0, 50) if rho is None else np.asarray(rho, float)
    n_theta, n_zeta = angles
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    rr, tt, _ = np.meshgrid(rho, theta, zeta, indexing="ij")
    r = major_radius + minor_radius * rr * np.cos(tt)
    z = minor_radius * rr * np.sin(tt)
    return surface_geometry(
        rho,
        minor_radius**2 * rr * r,
        np.full_like(r, 1.0 / minor_radius**2),
        r,
        z,
        field * major_radius / r,
        rotational_transform=rotational_transform,
        toroidal_current=toroidal_current,
        toroidal_flux_edge=np.pi * minor_radius**2 * field / (2.0 * np.pi),
    )


def gvec_geometry(
    equilibrium: Any,
    rho: Any = None,
    angles: tuple[int, int] = (48, 48),
) -> xr.Dataset:
    r"""Return the flux-surface averages of a GVEC equilibrium.

    Args:
        equilibrium: A `freeofcharge.gvec_equilibrium.GVECEquilibrium`, a
            ``gvec.State``, or a pair (parameter file, state file).
        rho: Radii above zero; by default 50 from 0.02 to 1.
        angles: $(\theta, \zeta)$ grid of each surface over one field period.

    Returns:
        The dataset of `freeofcharge.surface_geometry`.
    """
    from freeofcharge.gvec_equilibrium import GVECEquilibrium

    if isinstance(equilibrium, tuple):
        equilibrium = GVECEquilibrium.from_files(*equilibrium)
    elif not isinstance(equilibrium, GVECEquilibrium):
        equilibrium = GVECEquilibrium(equilibrium)
    rho = np.linspace(0.02, 1.0, 50) if rho is None else np.asarray(rho, float)
    return equilibrium.transport_geometry(rho, angles=angles)


def eqdsk_geometry(
    equilibrium: Any, directory: str | Any, *, name: str = "equilibrium.geqdsk"
) -> dict[str, Any]:
    r"""Return the TORAX geometry section of an axisymmetric freeofcharge equilibrium.

    The equilibrium is written as a G-EQDSK file into ``directory`` (with $q$,
    the axis field and the plasma current, which TORAX needs to identify the
    sign conventions), and TORAX reads it with COCOS 1: freeofcharge's $\psi$ is
    per radian and grows outwards.

    Args:
        equilibrium: A ``freeofcharge.Equilibrium``.
        directory: Folder for the file.
        name: File name.

    Returns:
        The ``geometry`` section (``model_name = "eqdsk"``) for `slowcooker.base_config`.
    """
    from pathlib import Path

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    equilibrium.to_geqdsk(folder / name)
    return {
        "model_name": "eqdsk",
        "geometry_file": name,
        "geometry_directory": str(folder),
        "cocos": 1,
    }
