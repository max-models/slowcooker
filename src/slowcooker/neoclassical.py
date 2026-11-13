r"""Stellarator neoclassical transport as a TORAX transport model (JAX).

A reduced model of the low-collisionality regimes of a stellarator, enough to
give TORAX the neoclassical heat and particle transport and the ambipolar
radial electric field $E_r$. **Its prefactors are written from the standard
forms and still have to be checked against the literature (Nemov et al. 1999
for the $1/\nu$ regime with the effective ripple; Beidler et al. 2011 for the
monoenergetic coefficients) and against a DKES or MONKES database for one
configuration.** Until then, use it for the shape of the dependencies, not for
quantitative predictions.

Per species of mass $m$, charge $Ze$ and temperature $T$, at the normalised
energy $K = mv^2/2T$, the radial drift is $v_d = KT/(ZeBR)$ and the monoenergetic
diffusion coefficient interpolates between the $1/\nu$ and $\sqrt\nu$ regimes,

$$
D_{1/\nu} = \frac{4}{9\pi}(2\epsilon_\mathrm{eff})^{3/2}\frac{v_d^2}{\nu_D},
\qquad
D_{\sqrt\nu} = c_{\sqrt\nu}\,\epsilon_\mathrm{eff}^{3/2}\frac{v_d^2\sqrt{\nu_D}}{|\Omega_E|^{3/2}},
\qquad
D = \frac{D_{1/\nu} D_{\sqrt\nu}}{D_{1/\nu} + D_{\sqrt\nu}},
$$

with the deflection frequency $\nu_D(v)$ on the electrons and the main ions and
the $E\times B$ precession $\Omega_E = E_r/(rB)$. The thermal coefficients are
$D_j = \frac{2}{\sqrt\pi}\int_0^\infty \sqrt K e^{-K} K^{j-1} D\, dK$ (Gauss–Laguerre),
and the fluxes

$$
\Gamma = -nD_1\left(\frac{n'}{n} - \frac{ZeE_r}{T} - \frac32\frac{T'}{T}\right) - nD_2\frac{T'}{T},
\qquad
\frac{Q}{T} = -nD_2\left(\frac{n'}{n} - \frac{ZeE_r}{T} - \frac32\frac{T'}{T}\right) - nD_3\frac{T'}{T}.
$$

$E_r$ solves $\Gamma_e = Z_i\Gamma_i$ (the impurities left out); where there are
several roots, the ion root (the most negative) is taken. TORAX gets the
effective diffusivities $\chi = -Q/(nT')$, $D_e = D_{1,e}$ and the convection
that makes up the rest of $\Gamma_e$. The radius is $r = \rho\,\rho_b$, TORAX's
effective minor radius.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ambipolar_field",
    "fluxes",
    "register",
    "thermal_coefficients",
]

E = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
KEV = 1.0e3 * E

_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(24)


def _chandrasekhar(x):
    """Return erf(x) - G(x), with G the Chandrasekhar function."""
    erf = jax.scipy.special.erf(x)
    g = (erf - 2.0 * x * jnp.exp(-(x**2)) / jnp.sqrt(jnp.pi)) / (2.0 * x**2)
    return erf - g


def _deflection(species, background, x):
    """Return nu_D of ``species`` at x = v/v_th on the ``background`` species."""
    mass, charge, density, temperature = species
    v_th = jnp.sqrt(2.0 * temperature / mass)
    total = 0.0
    for b_mass, b_charge, b_density, b_temperature, log in background:
        nu = (
            b_density
            * charge**2
            * b_charge**2
            * E**4
            * log
            / (4.0 * jnp.pi * EPS0**2 * mass**2 * v_th**3)
        )
        x_b = x * v_th / jnp.sqrt(2.0 * b_temperature / b_mass)
        total = total + nu * _chandrasekhar(x_b) / x**3
    return total


def thermal_coefficients(
    species,
    background,
    epsilon_eff,
    field,
    major_radius,
    radius,
    radial_field,
    sqrt_nu_coefficient=1.0,
):
    r"""Return the thermal coefficients $(D_1, D_2, D_3)$ [m^2/s] of one species.

    Args:
        species: ``(mass [kg], charge number, density [m^-3], temperature [J])``,
            arrays over the radius.
        background: Tuples ``(mass, charge, density, temperature, Coulomb log)``
            of the species it collides with (itself included).
        epsilon_eff: Effective ripple.
        field: Field strength $B$ [T].
        major_radius: $R$ [m].
        radius: Effective minor radius $r$ [m].
        radial_field: $E_r$ [V/m].
        sqrt_nu_coefficient: $c_{\sqrt\nu}$.

    Returns:
        $D_1$, $D_2$, $D_3$.
    """
    mass, charge, _, temperature = species
    k = jnp.asarray(_NODES)[:, None]
    w = jnp.asarray(_WEIGHTS)[:, None]
    nu = _deflection(species, background, jnp.sqrt(k))
    drift = k * temperature / (charge * E * field * major_radius)
    one_over_nu = 4.0 / (9.0 * jnp.pi) * (2.0 * epsilon_eff) ** 1.5 * drift**2 / nu
    omega = jnp.maximum(
        jnp.abs(radial_field) / (jnp.maximum(radius, 1e-3) * field), 1e-6
    )
    sqrt_nu = (
        sqrt_nu_coefficient * epsilon_eff**1.5 * drift**2 * jnp.sqrt(nu) / omega**1.5
    )
    d = one_over_nu * sqrt_nu / (one_over_nu + sqrt_nu)
    weight = 2.0 / jnp.sqrt(jnp.pi) * w * jnp.sqrt(k)
    return tuple(jnp.sum(weight * k ** (j - 1) * d, axis=0) for j in (1, 2, 3))


def fluxes(species, gradients, coefficients, radial_field):
    r"""Return the particle flux $\Gamma$ [1/(m^2 s)] and heat flux $Q$ [W/m^2].

    Args:
        species: ``(mass, charge, density, temperature)``.
        gradients: ``(dn/dr, dT/dr)``.
        coefficients: $(D_1, D_2, D_3)$.
        radial_field: $E_r$ [V/m].

    Returns:
        $\Gamma$ and $Q$.
    """
    _, charge, density, temperature = species
    dn, dt = gradients
    d1, d2, d3 = coefficients
    a1 = dn / density - charge * E * radial_field / temperature - 1.5 * dt / temperature
    a2 = dt / temperature
    gamma = -density * (d1 * a1 + d2 * a2)
    heat = -density * temperature * (d2 * a1 + d3 * a2)
    return gamma, heat


def ambipolar_field(
    electrons,
    ions,
    electron_gradients,
    ion_gradients,
    epsilon_eff,
    field,
    major_radius,
    radius,
    coulomb_log=17.0,
    sqrt_nu_coefficient=1.0,
    field_range=50.0e3,
    points=201,
):
    r"""Return the ambipolar $E_r$ [V/m]: $\Gamma_e = Z_i\Gamma_i$, the ion root first.

    $E_r$ is scanned on ``points`` values in $\pm$ ``field_range``; the root is
    the first sign change of $Z_i\Gamma_i - \Gamma_e$ from the negative side,
    refined by 30 bisections; without one, the $E_r$ of the smallest imbalance.
    """
    background = (
        (electrons[0], 1.0, electrons[2], electrons[3], coulomb_log),
        (ions[0], ions[1], ions[2], ions[3], coulomb_log),
    )
    scan = jnp.linspace(-field_range, field_range, points)[:, None]

    def imbalance(radial_field):
        ce = thermal_coefficients(
            electrons, background, epsilon_eff, field, major_radius, radius,
            radial_field, sqrt_nu_coefficient,
        )  # fmt: skip
        ci = thermal_coefficients(
            ions, background, epsilon_eff, field, major_radius, radius,
            radial_field, sqrt_nu_coefficient,
        )  # fmt: skip
        gamma_e, _ = fluxes(electrons, electron_gradients, ce, radial_field)
        gamma_i, _ = fluxes(ions, ion_gradients, ci, radial_field)
        return ions[1] * gamma_i - gamma_e

    values = jax.vmap(imbalance)(scan)
    sign_change = jnp.sign(values[:-1]) != jnp.sign(values[1:])
    first = jnp.argmax(sign_change, axis=0)
    found = jnp.any(sign_change, axis=0)
    columns = jnp.arange(values.shape[1])
    f_lo = values[first, columns]
    e_lo, e_hi = scan[first, 0], scan[first + 1, 0]

    def bisect(_, bracket):
        lower, upper, f_lower = bracket
        middle = 0.5 * (lower + upper)
        f_middle = imbalance(middle)
        same = jnp.sign(f_middle) == jnp.sign(f_lower)
        return (
            jnp.where(same, middle, lower),
            jnp.where(same, upper, middle),
            jnp.where(same, f_middle, f_lower),
        )

    lower, upper, _ = jax.lax.fori_loop(0, 30, bisect, (e_lo, e_hi, f_lo))
    root = 0.5 * (lower + upper)
    fallback = scan[jnp.argmin(jnp.abs(values), axis=0), 0]
    return jnp.where(found, root, fallback)


def _transport_classes():
    """Build the TORAX classes; deferred so that importing slowcooker stays light."""
    from torax import transport
    from torax._src import array_typing
    from torax._src.torax_pydantic import torax_pydantic

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class RuntimeParams(transport.ComponentRuntimeParams):
        epsilon_eff: array_typing.FloatVector
        sqrt_nu_coefficient: array_typing.FloatScalar
        coulomb_log: array_typing.FloatScalar
        field_range: array_typing.FloatScalar

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
    class StellaratorNeoclassicalModel(transport.ComponentTransportModel):
        def call_implementation(
            self,
            transport_runtime_params,
            runtime_params,
            geo,
            core_profiles,
            two_point_mask,
            component_state,
        ):
            del runtime_params, two_point_mask
            p: Any = transport_runtime_params
            kinetic = core_profiles.kinetic
            rho_b = geo.rho_b
            radius = geo.rho_face_norm * rho_b
            t_e = kinetic.T_e.face_value() * KEV
            t_i = kinetic.T_i.face_value() * KEV
            n_e = kinetic.n_e.face_value()
            n_i = kinetic.n_i.face_value()
            z_i = core_profiles.ions.Z_i_face
            m_i = core_profiles.ions.A_i * M_P
            electrons = (M_E, -1.0, n_e, t_e)
            ions = (m_i, z_i, n_i, t_i)
            ge = (
                kinetic.n_e.face_grad() / rho_b,
                kinetic.T_e.face_grad() * KEV / rho_b,
            )
            gi = (
                kinetic.n_i.face_grad() / rho_b,
                kinetic.T_i.face_grad() * KEV / rho_b,
            )
            common = dict(
                epsilon_eff=p.epsilon_eff,
                field=geo.B_0,
                major_radius=geo.R_major,
                radius=radius,
            )
            e_r = ambipolar_field(
                electrons, ions, ge, gi, coulomb_log=p.coulomb_log,
                sqrt_nu_coefficient=p.sqrt_nu_coefficient,
                field_range=p.field_range, **common,
            )  # fmt: skip
            background = (
                (M_E, 1.0, n_e, t_e, p.coulomb_log),
                (m_i, z_i, n_i, t_i, p.coulomb_log),
            )
            ce = thermal_coefficients(
                electrons, background, radial_field=e_r,
                sqrt_nu_coefficient=p.sqrt_nu_coefficient, **common,
            )  # fmt: skip
            ci = thermal_coefficients(
                ions, background, radial_field=e_r,
                sqrt_nu_coefficient=p.sqrt_nu_coefficient, **common,
            )  # fmt: skip
            gamma_e, q_e = fluxes(electrons, ge, ce, e_r)
            _, q_i = fluxes(ions, gi, ci, e_r)

            def diffusivity(heat, density, gradient):
                safe = jnp.where(jnp.abs(gradient) > 0.0, gradient, -1.0)
                chi = -heat / (density * safe)
                return jnp.where(jnp.abs(gradient) > 0.0, jnp.maximum(chi, 0.0), 0.0)

            chi_e = diffusivity(q_e, n_e, ge[1])
            chi_i = diffusivity(q_i, n_i, gi[1])
            d_e = ce[0]
            v_e = (gamma_e + d_e * ge[0]) / n_e
            return (
                transport.TransportCoeffs(
                    chi_face_ion=chi_i,
                    chi_face_el=chi_e,
                    d_face_el=d_e,
                    v_face_el=v_e,
                ),
                component_state,
            )

    class StellaratorNeoclassicalConfig(transport.ComponentTransportConfigBase):
        r"""TORAX config of the stellarator neoclassical model.

        Attributes:
            model_name: ``"stellarator_neoclassical"``.
            epsilon_eff: Effective ripple over the radius.
            sqrt_nu_coefficient: $c_{\sqrt\nu}$ (to be calibrated).
            coulomb_log: Coulomb logarithm.
            field_range: Half width of the $E_r$ scan [V/m].
        """

        model_name: Annotated[
            Literal["stellarator_neoclassical"], torax_pydantic.JAX_STATIC
        ] = "stellarator_neoclassical"
        epsilon_eff: torax_pydantic.PositiveTimeVaryingArray = (
            torax_pydantic.ValidatedDefault(0.01)
        )
        sqrt_nu_coefficient: float = 1.0
        coulomb_log: float = 17.0
        field_range: float = 50.0e3

        def build_transport_model(self):
            return StellaratorNeoclassicalModel()

        def build_runtime_params(self, t: chex.Numeric):
            base = super().build_runtime_params(t)
            return RuntimeParams(
                epsilon_eff=self.epsilon_eff.get_value(t, "face"),
                sqrt_nu_coefficient=self.sqrt_nu_coefficient,
                coulomb_log=self.coulomb_log,
                field_range=self.field_range,
                **vars(base),
            )

    return StellaratorNeoclassicalConfig


_REGISTERED = None


def register():
    """Register the model with TORAX (once) and return its config class.

    After this, ``{"model_name": "stellarator_neoclassical", "epsilon_eff": ...}``
    can go into ``transport.core_transport_models`` of a TORAX config.
    """
    global _REGISTERED
    if _REGISTERED is None:
        from torax import transport

        _REGISTERED = _transport_classes()
        transport.register_transport_model(_REGISTERED)
    return _REGISTERED


def __getattr__(name):
    if name == "StellaratorNeoclassicalConfig":
        return register()
    raise AttributeError(name)
