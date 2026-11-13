"""slowcooker: 1.5D core transport for tokamaks and stellarators on TORAX.

TORAX solves the transport; slowcooker gives it the geometry of any
equilibrium as flux-surface averages (GVEC for tokamaks and stellarators),
freeofcharge's neutral beams as a source, and the profiles back as an xarray
dataset.
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from slowcooker._profiling import PROFILE_TAGS
from slowcooker.config import base_config, merge, profile
from slowcooker.geometry import (
    chease_columns,
    circular_geometry,
    geometry_config,
    gvec_geometry,
)
from slowcooker.run import profiles, run, run_coupled, scan, snapshot

try:
    __version__ = version("slowcooker")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "base_config",
    "chease_columns",
    "circular_geometry",
    "geometry_config",
    "gvec_geometry",
    "merge",
    "profile",
    "profiles",
    "run",
    "run_coupled",
    "scan",
    "snapshot",
]
