"""TORAX on slowcooker geometries: the stellarator go/no-go and the run helpers."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from slowcooker.config import base_config, merge, profile
from slowcooker.geometry import circular_geometry
from slowcooker.run import profiles, run, run_coupled, snapshot

KEV = 1.602176634e-16


def _steady_state(geometry, data, density, chi, edge):
    r"""T_e from the heat balance P(rho) = -n chi (g1/V') dT/drho with TORAX's source.

    Uses slowcooker's averages, not TORAX's geometry: $g_1/V' = V'\langle|\nabla\rho|^2\rangle$.
    """
    rho = np.linspace(0.0, 1.0, 4001)
    vpr = np.interp(rho, geometry.rho, geometry.volume_derivative)
    grad2 = np.interp(rho, geometry.rho, geometry.grad_rho2)
    source = np.interp(rho, data.rho, data.p_generic_heat_e)
    power = cumulative_trapezoid(source * vpr, rho, initial=0.0)
    gradient = np.zeros_like(rho)
    gradient[1:] = -power[1:] / (density * chi * vpr[1:] * grad2[1:]) / KEV
    rise = cumulative_trapezoid(gradient[::-1], rho[::-1], initial=0.0)[::-1]
    return rho, edge + rise


def _electron_heat_only(geometry, density, chi, t_final, dt=0.05):
    config = base_config(
        geometry,
        density=density,
        temperature=(1.0, 0.1),
        heating=0.5e6,
        electron_fraction=1.0,
        chi=chi,
        t_final=t_final,
        dt=dt,
        n_rho=40,
    )
    return merge(
        config,
        {"numerics": {"evolve_ion_heat": False}, "sources": {"ei_exchange": None}},
    )


@pytest.fixture(scope="module")
def ellipse_geometry():
    pytest.importorskip("gvec")
    from slowcooker.data import gvec_equilibrium
    from slowcooker.geometry import gvec_geometry

    return gvec_geometry(gvec_equilibrium("ellipse"), rho=np.linspace(0.02, 1.0, 40))


def test_stellarator_steady_state_matches_the_heat_balance(ellipse_geometry):
    """SC3, the go/no-go: TORAX with a GVEC stellarator geometry, unchanged."""
    density, chi = 3.0e19, 0.5
    # about ten confinement times a^2/chi, to the steady state
    output = run(_electron_heat_only(ellipse_geometry, density, chi, 8.0, dt=0.2))
    data = profiles(output).isel(t=-1)
    rho, expected = _steady_state(ellipse_geometry, data, density, chi, 0.1)
    # 1.5 % of the peak: the residual (1.2 %) is in the innermost cells, where TORAX
    # smooths g1 towards its asymptotic form and the source is extrapolated to the axis
    np.testing.assert_allclose(
        data.T_e,
        np.interp(data.rho, rho, expected),
        atol=1.5e-2 * float(data.T_e.max()),
    )
    # TORAX's q is GVEC's 1/iota
    iota = np.abs(
        np.interp(data.rho, ellipse_geometry.rho, ellipse_geometry.rotational_transform)
    )
    np.testing.assert_allclose(data.q, 1.0 / iota, rtol=5e-3)


def test_tokamak_steady_state_and_the_output_layout():
    geometry = circular_geometry(3.0, 1.0, 2.5, 0.5, rho=np.linspace(0.02, 1.0, 60))
    output = run(_electron_heat_only(geometry, 3.0e19, 1.0, 8.0, dt=0.2))
    data = profiles(output)
    assert set(data.T_e.dims) == {"t", "rho"}
    assert data.t.attrs["units"] == "s" and data.T_e.attrs["units"] == "keV"
    assert data.p_generic_heat_e.dims == ("t", "rho")
    assert "P_generic_heat_total" in data and data.P_generic_heat_total.dims == ("t",)
    final = snapshot(output)
    rho, expected = _steady_state(geometry, final, 3.0e19, 1.0, 0.1)
    np.testing.assert_allclose(
        final.T_e,
        np.interp(final.rho, rho, expected),
        atol=1.5e-2 * float(final.T_e.max()),
    )
    assert float(snapshot(output, 0.0).T_e[0]) == pytest.approx(1.0, rel=1e-2)


def test_coupled_segments_continue_and_apply_their_changes():
    geometry = circular_geometry(rho=np.linspace(0.02, 1.0, 30))
    config = base_config(
        geometry, heating=1.0e6, chi=1.0, t_final=0.4, dt=0.05, n_rho=15
    )
    seen = []

    def double_the_heating(time, state):
        seen.append((time, float(state.T_e[0])))
        return {"sources": {"generic_heat": {"P_total": 1.0e6 * (1 + (time > 0))}}}

    output = run_coupled(config, [double_the_heating], [0.0, 0.2, 0.4])
    data = profiles(output)
    np.testing.assert_allclose(data.t[[0, -1]], [0.0, 0.4])
    assert [t for t, _ in seen] == [0.0, 0.2]
    power = data.P_generic_heat_total
    assert float(power.sel(t=0.1, method="nearest")) == pytest.approx(1.0e6, rel=1e-3)
    assert float(power.sel(t=0.3, method="nearest")) == pytest.approx(2.0e6, rel=1e-3)
    with pytest.raises(ValueError, match="increasing"):
        run_coupled(config, [], [0.0])


def test_merge_and_profile():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = merge(base, {"a": {"b": 5}, "d": None, "e": {"f": 1}})
    assert merged == {"a": {"b": 5, "c": 2}, "e": {"f": 1}}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}
    assert profile([0.0, 1.0], 2.0, time=1.0) == {1.0: {0.0: 2.0, 1.0: 2.0}}


def test_command_line(tmp_path):
    from slowcooker.main import main

    script = tmp_path / "case.py"
    script.write_text(
        "import numpy as np\n"
        "from slowcooker import base_config, circular_geometry\n"
        "CONFIG = base_config(circular_geometry(rho=np.linspace(0.05, 1, 20)),"
        " t_final=0.1, dt=0.05, n_rho=10)\n"
    )
    out = tmp_path / "out.nc"
    assert main([str(script), "-o", str(out)]) == 0
    import xarray as xr

    assert xr.open_dataset(out).sizes["t"] == 3


def test_scan_of_the_heating_power():
    from slowcooker import scan

    geometry = circular_geometry(rho=np.linspace(0.02, 1.0, 30))
    config = base_config(geometry, chi=1.0, t_final=0.2, dt=0.1, n_rho=10)
    data = scan(config, "sources.generic_heat.P_total", [1e6, 2e6])
    assert data.P_total.values.tolist() == [1e6, 2e6]
    stored = data.W_thermal_total.isel(t=-1)
    assert float(stored.sel(P_total=2e6)) > float(stored.sel(P_total=1e6))
