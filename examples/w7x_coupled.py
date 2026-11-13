"""W7-X, everything coupled.

TORAX evolves the temperatures of a W7-X plasma heated by two neutral beams. Before
each segment, freeofcharge recomputes the beams from the plasma and GVEC re-solves
the equilibrium with TORAX's pressure; the transport is the stellarator
neoclassical model on a small turbulent background. The beam geometry is
freeofcharge's stand-in for the W7-X injectors, and the neoclassical model's
prefactors are still to be verified, so the numbers are indicative.
"""

import freeofcharge as foc
import matplotlib.pyplot as plt
import numpy as np
import plasma_plots  # noqa: F401
from freeofcharge.injectors import w7x_injectors
from gvec.util.params import read_parameters

from slowcooker import base_config, gvec_geometry, merge, profiles, run_coupled
from slowcooker.data import equilibrium_files, gvec_equilibrium
from slowcooker.equilibrium import EquilibriumCoupling
from slowcooker.nbi import NBICoupling
from slowcooker.neoclassical import register

register()
equilibrium = gvec_equilibrium("w7x")
rho = np.linspace(0.02, 1.0, 30)
geometry = gvec_geometry(equilibrium, rho=rho, angles=(32, 32))
config = merge(
    base_config(geometry, density=5e19, chi=0.1, t_final=0.6, dt=0.05, n_rho=20),
    {
        "sources": {"generic_heat": None},
        "transport": {
            "core_transport_models": {
                "neo": {"model_name": "stellarator_neoclassical", "epsilon_eff": 0.01}
            }
        },
    },
)
beams = NBICoupling(
    foc.GriddedEquilibrium(equilibrium, (48, 32, 48)),
    w7x_injectors(sources={"NI21": [1, 2]}),
    rho_edges=np.linspace(0.0, 1.0, 16),
    options={"markers": 500, "seed": 1},
)
pressure = EquilibriumCoupling(
    read_parameters(str(equilibrium_files("w7x")[0])),
    equilibrium.state,
    rho=rho,
    B_0=config["geometry"]["B_0"],
)
data = profiles(run_coupled(config, [beams, pressure], [0.0, 0.2, 0.4, 0.6]))

for start, result in beams.results.items():
    print(
        f"t = {start:.1f} s: beams absorbed {float(result.absorbed_power) / 1e6:.2f} MW,"
        f" shine-through {100 * float(result.shine_through_fraction):.1f} %"
    )

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
data.T_e.plasma.plot.profiles(over="t", ax=axes[0], title="$T_e$ [keV]")
data.T_i.plasma.plot.profiles(over="t", ax=axes[1], title="$T_i$ [keV]")
for start, surfaces in pressure.geometries.items():
    surfaces.grad_rho2.plasma.plot.lineout(
        x="rho",
        ax=axes[2],
        label=f"{start:.1f} s",
        title=r"$\langle|\nabla\rho|^2\rangle$",
    )
axes[2].legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(6, 4))
data.W_thermal_total.plasma.plot.timeseries(
    logy=False, ax=ax, title="stored energy [J]"
)
