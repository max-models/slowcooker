---
title: Installation
description: Create a Python environment and install slowcooker.
---

slowcooker needs Python 3.12 or newer (TORAX's requirement). The `gvec` extra builds GVEC from
source, which needs a Fortran compiler and netCDF-Fortran; on Debian or Ubuntu:

```bash
sudo apt install gfortran cmake pkg-config liblapack-dev libnetcdff-dev
```

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements with pip:

```bash
pip install -e ".[gvec]"
```

Without the `gvec` extra, slowcooker works with geometries you compute otherwise (for example
`slowcooker.circular_geometry`), but not with GVEC equilibria.

## Optional extras

| Extra  | Installs                                                |
| ------ | ------------------------------------------------------- |
| `gvec` | GVEC (built from source), for GVEC equilibria         |
| `test` | `pytest` and `coverage`                                 |
| `docs` | the notebook runner, griffe, plasma-plots and GVEC      |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```
