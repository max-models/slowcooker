import numpy as np
import pytest
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry as torax_circular

from slowcooker.geometry import chease_columns, circular_geometry, geometry_config


def _torax_geometry(config):
    """The StandardGeometry TORAX builds from a CHEASE config section."""
    config = dict(config)
    config.pop("model_name")
    return chease.CheaseConfig(**config)._build_geometry()


@pytest.fixture(scope="module")
def circular():
    return circular_geometry(3.0, 1.0, 2.5, 0.5, rho=np.linspace(0.02, 1.0, 60))


def test_circular_matches_torax_circular_geometry(circular):
    ours = _torax_geometry(geometry_config(circular, n_rho=25))
    theirs = torax_circular.CircularConfig(
        R_major=3.0, a_minor=1.0, B_0=2.5, n_rho=25, elongation_LCFS=1.0
    )._build_geometry()
    for name in ("vpr", "vpr_face", "g0", "spr"):
        np.testing.assert_allclose(
            getattr(ours, name), getattr(theirs, name), rtol=1e-10, atol=1e-12
        )
    # the volume integrates V' on each code's own grid near the axis
    np.testing.assert_allclose(ours.volume_face[-1], theirs.volume_face[-1], rtol=1e-3)
    # <|grad V|^2> agrees up to TORAX's near-axis smoothing
    g1 = np.asarray(theirs.g1)
    np.testing.assert_allclose(ours.g1, g1, atol=1e-3 * g1.max())
    # TORAX's circular model sets <1/R^2> = 1/R0^2; ours is exact
    rho = np.asarray(ours.rho_norm)
    exact = 1.0 / (3.0 * np.sqrt(9.0 - rho**2))
    np.testing.assert_allclose(ours.g3, exact, rtol=1e-3)


def test_columns_give_the_metric_coefficients(circular):
    """g1, g2 from TORAX equal V'^2 times the averages (to the interpolation error)."""
    geo = _torax_geometry(geometry_config(circular, n_rho=40))
    # TORAX smooths g1 and g2 towards their asymptotic form near the axis
    away = np.asarray(geo.rho_face_norm) > 0.15
    rho = np.asarray(geo.rho_face_norm)[away]
    vpr = np.interp(rho, circular.rho, circular.volume_derivative)
    g1 = vpr**2 * np.interp(rho, circular.rho, circular.grad_rho2)
    g2 = vpr**2 * np.interp(rho, circular.rho, circular.grad_rho2_over_R2)
    np.testing.assert_allclose(np.asarray(geo.g1_face)[away], g1, rtol=5e-3)
    np.testing.assert_allclose(np.asarray(geo.g2_face)[away], g2, rtol=5e-3)


def test_poloidal_flux_follows_iota():
    rho = np.linspace(0.02, 1.0, 80)
    geo = circular_geometry(rotational_transform=0.3 + 0.4 * rho**2, rho=rho)
    columns, scalars = chease_columns(geo)
    psi = columns["psi_chease"] * scalars["R_major"] ** 2 * scalars["B_0"] * 2 * np.pi
    phi_b = 2 * np.pi * geo.attrs["toroidal_flux_edge"]
    # psi = int iota dPhi = Phi_b (0.3 rho^2 + 0.2 rho^4)
    np.testing.assert_allclose(
        psi, phi_b * (0.3 * rho**2 + 0.2 * rho**4), rtol=2e-3, atol=1e-6
    )


def test_time_dependent_geometry_and_errors(circular):
    config = geometry_config({0.0: circular, 1.0: circular}, n_rho=10)
    assert sorted(config["chease_data"]) == [0.0, 1.0]
    assert config["trapped_fraction_source"] == "file"
    with pytest.raises(ValueError, match="rotational transform"):
        chease_columns(circular.drop_vars("rotational_transform"))


def test_gvec_geometry_of_the_bundled_equilibria():
    pytest.importorskip("gvec")
    from slowcooker.data import equilibrium_files, gvec_equilibrium
    from slowcooker.geometry import gvec_geometry

    rho = np.linspace(0.1, 1.0, 5)
    ellipse = gvec_geometry(equilibrium_files("ellipse"), rho=rho, angles=(16, 16))
    assert ellipse.attrs["nfp"] == 2
    w7x = gvec_geometry(gvec_equilibrium("w7x"), rho=rho, angles=(16, 16))
    assert w7x.attrs["nfp"] == 5
    # W7-X: iota near 0.85 to 1 (GVEC's sign is negative), little toroidal current
    assert np.all(np.abs(w7x.rotational_transform) > 0.8)
    assert np.all(np.abs(w7x.toroidal_current) < 1e3)
    with pytest.raises(ValueError, match="unknown equilibrium"):
        equilibrium_files("iter")
