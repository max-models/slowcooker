# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Tokamaks from freeofcharge's axisymmetric equilibria: `geometry.eqdsk_geometry` writes the
  G-EQDSK file that TORAX reads (COCOS 1); `base_config` takes such a geometry section;
  `nbi.toroidal_radius` maps freeofcharge's poloidal radius to TORAX's toroidal one
  ($\Phi = \int q\,d\psi$), and `NBICoupling` uses it, so the beam-driven current reaches TORAX
  in a tokamak. Example notebook "Beams in a tokamak".
- The re-solve interval of `EquilibriumCoupling`, studied in "Pressure and geometry": on W7-X one
  re-solve per confinement time leaves below 1e-4 in $T_e$. GVEC runs at log level WARNING.
- slowcooker: 1.5D core transport for tokamaks and stellarators on TORAX 2.0.0 (unchanged, pinned).
- `slowcooker.geometry`: flux-surface averages into TORAX's CHEASE geometry (in memory), with the
  poloidal flux from the rotational transform and an effective $F$ that gives the true $V'$;
  `gvec_geometry` for GVEC equilibria, `circular_geometry` as the analytic case.
- `slowcooker.config` (`base_config`, `merge`, `profile`) and `slowcooker.run` (`run`, `profiles`
  in plasma-plots' layout, `snapshot`, `scan`, and `run_coupled` for codes coupled between
  segments through TORAX's restart).
- `slowcooker.nbi`: freeofcharge's neutral beams as TORAX's prescribed `nbi` source, recomputed
  from the plasma before each segment (`NBICoupling`).
- `slowcooker.equilibrium`: GVEC re-solved with TORAX's pressure (`EquilibriumCoupling`).
- `slowcooker.neoclassical`: a stellarator neoclassical transport model (1/ν and √ν regimes with
  the effective ripple, ambipolar $E_r$) registered with TORAX; prefactors still to be verified.
- `slowcooker.data`: a reduced W7-X GVEC equilibrium and a small rotating ellipse.
- The `slowcooker` command, example notebooks (geometry, W7-X heat transport, beams, pressure and
  geometry) and a fully coupled W7-X example script.

## [Template]

### Fixed

- `setup_project.sh` also renames the docs smoke paths (`api/slowcooker/...` in `docs.yml`), so the
  docs build of a new project no longer looks for the template's API pages.

### Added

- Opt-in scope-profiler regions, as in plasmacoll, wellbalanced and patchwork: `_profiling.py`
  with the package's tag (`slowcooker`, renamed by `setup_project.sh`), registered on
  import, and an example region on `main`. Off unless the application enables the tag (e.g.
  `scope_profiler.setup(enable_tags=["slowcooker"])`). Adds the `scope-profiler>=0.7.5`
  dependency.
- `PyPI dependencies` workflow and `scripts/check_pypi_dependencies.py`: on every pull request to
  `main`, fail if a dependency is a direct reference (`name @ git+https://...`), is not on PyPI, or
  has a version range that no release satisfies, instead of failing the release after the merge.

### Changed

- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (a hardcoded one went stale in releases).

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
