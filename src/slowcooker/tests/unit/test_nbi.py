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


def test_toroidal_radius_against_torax(tmp_path):
    """The poloidal-to-toroidal map of a tokamak against TORAX's psi on its toroidal grid."""
    import freeofcharge as foc
    from torax._src.geometry import eqdsk

    from slowcooker.geometry import eqdsk_geometry
    from slowcooker.nbi import toroidal_radius

    equilibrium = foc.Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, q_axis=1.2)
    section = eqdsk_geometry(equilibrium, tmp_path)
    section.pop("model_name")
    geo = eqdsk.EQDSKConfig(
        **section, n_rho=40, last_surface_factor=1.0, n_surfaces=200
    )._build_geometry()
    # TORAX's psi [Wb] is 2 pi times freeofcharge's (per radian)
    flux = 2 * np.pi * (equilibrium.psi_boundary - equilibrium.psi_axis)
    psi_n = np.clip(np.asarray(geo.psi) / flux, 0.0, 1.0)
    mapped = toroidal_radius(equilibrium, np.sqrt(psi_n))
    # outside rho = 0.12, where TORAX smooths psi towards its axis form
    rho = np.asarray(geo.rho_norm)
    away = rho > 0.12
    np.testing.assert_allclose(mapped[away], rho[away], atol=1e-4)
    ends = toroidal_radius(equilibrium, [0.0, 1.0])
    np.testing.assert_allclose(ends, [0.0, 1.0])
    # a 3D (toroidal-flux) equilibrium passes through unchanged
    assert toroidal_radius(object(), [0.3])[0] == 0.3


def test_nbi_coupling_in_a_tokamak(tmp_path):
    """SC4, tokamak: freeofcharge's Solov'ev equilibrium, its EQDSK in TORAX, a tangential beam."""
    import freeofcharge as foc

    from slowcooker import base_config, merge, profiles, run_coupled
    from slowcooker.geometry import eqdsk_geometry
    from slowcooker.nbi import NBICoupling

    equilibrium = foc.Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, q_axis=1.2)
    origin, direction = foc.tangential_injection(1.3, 3.5)
    beam = foc.NeutralBeam.gaussian(
        origin, direction, 60e3, 2.5e6, mass_number=2, width=0.1,
        divergence=0.01, markers=400, seed=1,
    )  # fmt: skip
    config = merge(
        base_config(
            eqdsk_geometry(equilibrium, tmp_path),
            density=4e19,
            heating=0.0,
            chi=0.5,
            t_final=0.2,
            dt=0.05,
            n_rho=15,
        ),  # fmt: skip
        {"sources": {"generic_heat": None}},
    )
    coupling = NBICoupling(equilibrium, beam, rho_edges=np.linspace(0, 1, 16))
    data = profiles(run_coupled(config, [coupling], [0.0, 0.1, 0.2]))
    result = coupling.results[0.1]
    absorbed = float(result["absorbed_power"])
    assert 1.5e6 < absorbed <= 2.5e6
    np.testing.assert_allclose(float(data.P_nbi_total.isel(t=-1)), absorbed, rtol=5e-2)
    # the shells came back on the toroidal radius
    assert result.rho.values[-1] < 1.0 and np.all(np.diff(result.rho.values) > 0)
