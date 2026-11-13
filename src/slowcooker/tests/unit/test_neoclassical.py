import jax
import numpy as np
import pytest

from slowcooker.neoclassical import (
    KEV,
    M_E,
    M_P,
    ambipolar_field,
    fluxes,
    register,
    thermal_coefficients,
)

jax.config.update("jax_enable_x64", True)

RADIUS = np.array([0.2, 0.3])
DENSITY = np.array([5e19, 5e19])


def _species(t_e=1.0, t_i=1.0):
    electrons = (M_E, -1.0, DENSITY, t_e * KEV * np.ones(2))
    ions = (2 * M_P, 1.0, DENSITY, t_i * KEV * np.ones(2))
    return electrons, ions


def _gradients(species, length=0.5):
    return (-species[2] / length, -species[3] / length)


def test_one_over_nu_scaling_without_field():
    """At E_r = 0 only the 1/nu regime is left: D_1 ~ T^(7/2) / n at fixed T_e/T_i."""
    coefficients = []
    for scale in (1.0, 2.0):
        electrons, ions = _species(scale, scale)
        background = (
            (M_E, 1.0, DENSITY, electrons[3], 17.0),
            (ions[0], 1.0, DENSITY, ions[3], 17.0),
        )
        d1, d2, d3 = thermal_coefficients(ions, background, 0.01, 2.5, 5.5, RADIUS, 0.0)
        assert np.all(d2 > d1) and np.all(d3 > d2)
        coefficients.append(np.asarray(d1))
    np.testing.assert_allclose(coefficients[1] / coefficients[0], 2**3.5, rtol=1e-6)


def test_electric_field_reduces_transport():
    electrons, ions = _species()
    background = (
        (M_E, 1.0, DENSITY, electrons[3], 17.0),
        (ions[0], 1.0, DENSITY, ions[3], 17.0),
    )
    free = thermal_coefficients(ions, background, 0.01, 2.5, 5.5, RADIUS, 0.0)[0]
    reduced = thermal_coefficients(ions, background, 0.01, 2.5, 5.5, RADIUS, 1e4)[0]
    assert np.all(np.asarray(reduced) < np.asarray(free))


@pytest.mark.parametrize(("t_e", "sign"), [(1.0, -1.0), (4.0, 1.0)])
def test_ambipolar_roots(t_e, sign):
    """Ion root (E_r < 0) at T_e = T_i, electron root (E_r > 0) at T_e >> T_i."""
    electrons, ions = _species(t_e, 1.0)
    ge, gi = _gradients(electrons), _gradients(ions)
    e_r = np.asarray(ambipolar_field(electrons, ions, ge, gi, 0.01, 2.5, 5.5, RADIUS))
    assert np.all(np.sign(e_r) == sign)
    background = (
        (M_E, 1.0, DENSITY, electrons[3], 17.0),
        (ions[0], 1.0, DENSITY, ions[3], 17.0),
    )
    ce = thermal_coefficients(electrons, background, 0.01, 2.5, 5.5, RADIUS, e_r)
    ci = thermal_coefficients(ions, background, 0.01, 2.5, 5.5, RADIUS, e_r)
    gamma_e, _ = fluxes(electrons, ge, ce, e_r)
    gamma_i, _ = fluxes(ions, gi, ci, e_r)
    np.testing.assert_allclose(gamma_i, gamma_e, rtol=1e-6)


def test_model_runs_in_torax_on_w7x():
    """SC5: the model registered with TORAX, on the GVEC W7-X geometry."""
    pytest.importorskip("gvec")
    from slowcooker import base_config, gvec_geometry, merge, profiles, run
    from slowcooker.data import gvec_equilibrium

    config_class = register()
    assert register() is config_class
    geometry = gvec_geometry(
        gvec_equilibrium("w7x"), rho=np.linspace(0.02, 1.0, 20), angles=(24, 24)
    )
    config = merge(
        base_config(geometry, heating=2e6, t_final=0.2, dt=0.05, n_rho=15),
        {
            "transport": {
                "core_transport_models": {
                    "constant": {"chi_i": 0.1, "chi_e": 0.1},
                    "neo": {
                        "model_name": "stellarator_neoclassical",
                        "epsilon_eff": 0.01,
                    },
                }
            }
        },
    )
    data = profiles(run(config)).isel(t=-1)
    chi_i = data.chi_turb_i.values
    assert np.all(chi_i >= 0.1 - 1e-12) and chi_i.max() > 0.15
