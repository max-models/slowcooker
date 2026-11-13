"""The ``slowcooker`` command: run a TORAX config file and save the profiles."""

from __future__ import annotations

import argparse
import runpy
from collections.abc import Sequence
from pathlib import Path

import scope_profiler as sp

from slowcooker._profiling import PROFILE_TAGS


@sp.profile("slowcooker.main", tags=PROFILE_TAGS)
def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``CONFIG`` of a Python file with TORAX and write the profiles to netCDF.

    The file defines ``CONFIG``, a TORAX config dictionary (for instance from
    `slowcooker.base_config`). The output is `slowcooker.profiles` of the run.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(
        prog="slowcooker", description="Run a TORAX config and save the profiles."
    )
    parser.add_argument("config", type=Path, help="Python file that defines CONFIG")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("profiles.nc"), help="netCDF file"
    )
    args = parser.parse_args(argv)
    from slowcooker.run import profiles, run

    config = runpy.run_path(str(args.config))["CONFIG"]
    data = profiles(run(config))
    data.to_netcdf(args.output)
    print(f"wrote {args.output}: {data.sizes['t']} times, {data.sizes['rho']} radii")
    return 0
