import numpy as np
import pytest
import xarray as xr

from slowcooker.nbi import nbi_source


def _result(scale=1.0):
    rho = np.linspace(0.05, 0.95, 10)
    values = scale * (1.0 - rho**2)
    return xr.Dataset(
        {
            "electron_heating": ("rho", 2e5 * values),
            "ion_heating": ("rho", 1e5 * values),
            "birth_density": ("rho", 1e19 * values),
            "current_density": ("rho", np.full_like(rho, np.nan)),
        },
        coords={"rho": rho},
    )


def test_nbi_source_channels_and_times():
    section = nbi_source({0.0: _result(), 1.0: _result(2.0)})["sources"]["nbi"]
    assert section["model_name"] == "prescribed" and section["mode"] == "prescribed"
    values = section["prescribed_values"]
    assert set(values) == {
        "power_density_e",
        "power_density_i",
        "particle_source_e",
        "j_parallel",
    }
    assert sorted(values["power_density_e"]) == [0.0, 1.0]
    assert values["power_density_e"][1.0][0.05] == pytest.approx(
        2 * 2e5 * (1 - 0.05**2)
    )
    # a current that freeofcharge leaves undefined (3D) goes in as zero
    assert all(v == 0.0 for v in values["j_parallel"][0.0].values())
    only_heat = nbi_source(_result(), current=False, fuelling=False)
    assert set(only_heat["sources"]["nbi"]["prescribed_values"]) == {
        "power_density_e",
        "power_density_i",
    }


def test_nbi_coupling_on_w7x():
    """SC4: freeofcharge's beams heat a W7-X plasma in TORAX, recomputed per segment."""
    pytest.importorskip("gvec")
    import freeofcharge as foc
    from freeofcharge.injectors import w7x_injectors

    from slowcooker import base_config, gvec_geometry, merge, profiles, run_coupled
    from slowcooker.data import gvec_equilibrium
    from slowcooker.nbi import NBICoupling

    equilibrium = gvec_equilibrium("w7x")
    geometry = gvec_geometry(
        equilibrium, rho=np.linspace(0.02, 1.0, 30), angles=(32, 32)
    )
    config = merge(
        base_config(geometry, heating=0.0, chi=0.5, t_final=0.2, dt=0.05, n_rho=15),
        {"sources": {"generic_heat": None}},
    )
    coupling = NBICoupling(
        foc.GriddedEquilibrium(equilibrium, (24, 24, 24)),
        w7x_injectors(sources={"NI21": [1]}),
        options={"markers": 100, "seed": 1},
    )
    data = profiles(run_coupled(config, [coupling], [0.0, 0.1, 0.2]))
    assert sorted(coupling.results) == [0.0, 0.1]
    absorbed = float(coupling.results[0.1]["absorbed_power"])
    assert 1.0e6 < absorbed < 2.0e6  # one source of about 1.8 MW, little shine-through
    # TORAX gets the absorbed power of the segment's freeofcharge result, to the error of
    # interpolating 25 Monte Carlo shells onto its 15 cells (2.2 % on CI, 1-2 % locally)
    np.testing.assert_allclose(float(data.P_nbi_total.isel(t=-1)), absorbed, rtol=5e-2)
    assert float(data.T_e.isel(t=-1, rho=0)) > 0.1
