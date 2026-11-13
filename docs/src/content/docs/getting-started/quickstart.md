---
title: Quickstart
description: Run TORAX on the W7-X geometry from GVEC in a few lines.
---

First, ensure that `slowcooker` is [installed](/slowcooker/getting-started/installa## A first run

```python
import numpy as np

from slowcooker import base_config, gvec_geometry, profiles, run
from slowcooker.data import gvec_equilibrium

geometry = gvec_geometry(gvec_equilibrium("w7x"), rho=np.linspace(0.02, 1.0, 40))
config = base_config(geometry, density=3e19, heating=2e6, chi=0.5, t_final=1.0)
data = profiles(run(config))  # xarray over t and rho
```

`gvec_geometry` computes the flux-surface averages of the bundled W7-X equilibrium,
`base_config` writes a TORAX config for them (rotational transform held fixed, temperatures
evolved, constant diffusivities, central heating), `run` runs TORAX and `profiles` returns the
output as one dataset. Change any part of the config with `slowcooker.merge`, and see the
[examples](/slowcooker/examples/) for the beams, the equilibrium coupling and the neoclassical
model.

## From the command line

```bash
slowcooker case.py -o profiles.nc
```

runs the `CONFIG` dictionary that `case.py` defines and writes the profiles to netCDF.

nction.

## Profiling

slowcooker's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`slowcooker.run`, `slowcooker.run_coupled`, `slowcooker.nbi`, `slowcooker.equilibrium`, ...), tagged with the opt-in tag `slowcooker`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["slowcooker"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=slowcooker`,
`scope-profiler run --enable-tag slowcooker` or `enable_tags = ["slowcooker"]` in the
TOML configuration. New regions use `@sp.profile("slowcooker.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
