import numpy as np
import pytest

gvec = pytest.importorskip("gvec")

from gvec.util.params import read_parameters  # noqa: E402

from slowcooker import base_config, gvec_geometry, profiles, run_coupled  # noqa: E402
from slowcooker.data import equilibrium_files, gvec_equilibrium  # noqa: E402
from slowcooker.equilibrium import EquilibriumCoupling, pressure_parameters  # noqa: E402


def test_pressure_fit_reproduces_the_profile():
    parameters = read_parameters(str(equilibrium_files("w7x")[0]))
    rho = np.linspace(0.0, 1.0, 41)
    pressure = 5e4 * (1 - rho**2) ** 2
    fitted = pressure_parameters(parameters, rho, pressure)
    assert fitted["pres_type"] == "bspline"
    assert "pres" not in fitted and "pres" in parameters  # the input is not changed
    from scipy.interpolate import BSpline

    spline = BSpline(
        np.asarray(fitted["pres_knots"]), np.asarray(fitted["pres_coefs"]), 3
    )
    np.testing.assert_allclose(spline(rho**2), pressure, atol=5e-3 * pressure.max())


def test_equilibrium_follows_the_pressure(tmp_path):
    """SC6: the geometry of each segment comes from a GVEC re-solve with TORAX's pressure."""
    parameters = read_parameters(str(equilibrium_files("w7x")[0]))
    equilibrium = gvec_equilibrium("w7x")
    rho = np.linspace(0.02, 1.0, 20)
    geometry = gvec_geometry(equilibrium, rho=rho, angles=(24, 24))
    config = base_config(
        geometry, density=1e20, heating=5e6, chi=0.5, t_final=0.2, dt=0.05, n_rho=12
    )
    coupling = EquilibriumCoupling(
        parameters,
        equilibrium.state,
        rho=rho,
        angles=(24, 24),
        B_0=config["geometry"]["B_0"],
        workdir=tmp_path,
    )
    data = profiles(run_coupled(config, [coupling], [0.0, 0.1, 0.2]))
    assert sorted(coupling.geometries) == [0.0, 0.1]
    first = coupling.geometries[0.0]
    # the bundled state has <beta> = 4.8 %; the transport pressure is lower, so the
    # surfaces relax and the geometry changes, while iota stays that of the parameters
    assert float(data.pressure_total.isel(t=0, rho=0)) < 2e5
    assert not np.allclose(first.grad_rho2, geometry.grad_rho2, rtol=1e-3)
    np.testing.assert_allclose(
        first.rotational_transform, geometry.rotational_transform, rtol=1e-3
    )
    np.testing.assert_allclose(data.t[-1], 0.2)
