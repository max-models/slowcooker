"""Run TORAX, return the profiles as one dataset, and couple other codes in time segments.

`run` runs a config and returns TORAX's output tree. `profiles` turns it into
one xarray dataset over ``t`` and ``rho`` (the normalised toroidal flux radius)
in plasma-plots' layout, with the scalars of TORAX over ``t``.

`run_coupled` couples codes that TORAX cannot call inside its compiled step
(freeofcharge's NBI, a GVEC re-solve): the run is cut into segments, and before
each one every coupling looks at the profiles at its start and returns changes
to the config (sources, geometry). The segments continue from each other
through TORAX's own restart, so the whole state carries over, and their
histories are stitched into one output.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr

from slowcooker._profiling import PROFILE_TAGS
from slowcooker.config import merge

__all__ = ["Coupling", "profiles", "run", "run_coupled", "scan", "snapshot"]

Coupling = Callable[[float, xr.Dataset], Mapping[str, Any]]
"""A coupling: ``coupling(time, state)`` returns config changes for the segment
that starts at ``time`` [s]; ``state`` is `snapshot` of the profiles there."""

_RADII = ("rho_norm", "rho_cell_norm", "rho_face_norm")


def _torax():
    import torax

    return torax


@sp.profile("slowcooker.run", tags=PROFILE_TAGS)
def run(
    config: Mapping[str, Any] | Any, *, max_steps: int | None = None
) -> xr.DataTree:
    """Run TORAX on a config and return its output tree.

    Args:
        config: A config dictionary or a ``torax.ToraxConfig``.
        max_steps: Most time steps, TORAX's default if None; stopping there
            before the final time is not an error.

    Returns:
        TORAX's output (``StateHistory.simulation_output_to_xr``).
    """
    torax = _torax()
    if isinstance(config, Mapping):
        config = torax.ToraxConfig.from_dict(dict(config))
    history = torax.run_simulation(config, progress_bar=False, max_steps=max_steps)
    error = history.sim_error
    stopped_early = max_steps is not None and error.name == "DID_NOT_REACH_T_FINAL"
    if error.value != 0 and not stopped_early:
        raise RuntimeError(f"TORAX stopped with {error.name}")
    return history.simulation_output_to_xr()


def profiles(output: xr.DataTree) -> xr.Dataset:
    """Return TORAX's profiles and scalars as one dataset over ``t`` and ``rho``.

    TORAX keeps its profiles on three grids (cell centres with the axis and the
    edge, the cells, the faces); here every profile is interpolated linearly
    onto the first, named ``rho``, holding the outermost values of the cell
    grid out to the axis and the edge (the normalised toroidal flux radius). The
    scalars come along over ``t``. Units are kept in the ``units`` attributes.

    Args:
        output: The tree that `run` returns.

    Returns:
        The dataset.
    """
    group = output["profiles"].to_dataset()
    rho = group["rho_norm"].values
    variables: dict[str, xr.DataArray] = {}
    for name, array in group.data_vars.items():
        radial = [d for d in array.dims if d in _RADII]
        if not radial:
            continue
        dim = radial[0]
        if dim != "rho_norm":
            # hold the first and last values out to the axis and the edge:
            # extrapolating noisy cell values (e.g. Monte Carlo sources) can flip signs
            grid = array[dim].values
            ends = []
            if grid[0] > rho[0]:
                ends.append(array.isel({dim: [0]}).assign_coords({dim: [rho[0]]}))
            ends.append(array)
            if grid[-1] < rho[-1]:
                ends.append(array.isel({dim: [-1]}).assign_coords({dim: [rho[-1]]}))
            array = xr.concat(ends, dim=dim).interp({dim: rho})
            array = array.drop_vars(dim, errors="ignore").rename({dim: "rho_norm"})
        variables[str(name)] = array
    scalars = output["scalars"].to_dataset()
    for name, array in scalars.data_vars.items():
        if set(array.dims) <= {"time"}:
            variables[str(name)] = array
    dataset = xr.Dataset(variables).rename({"time": "t", "rho_norm": "rho"})
    dataset["t"].attrs.update(long_name="time", units="s")
    dataset["rho"].attrs.update(
        long_name=r"$\rho = \sqrt{\Phi/\Phi_b}$", units="", radial_coordinate="toroidal"
    )
    return dataset


def snapshot(output: xr.DataTree, time: float | None = None) -> xr.Dataset:
    """Return the profiles of an output at one time (the last by default).

    Args:
        output: The tree that `run` returns.
        time: Time [s]; the nearest output time is taken.

    Returns:
        `profiles` at that time, without the ``t`` dimension.
    """
    data = profiles(output)
    if time is None:
        return data.isel(t=-1)
    return data.sel(t=time, method="nearest")


def _segment_config(
    config: Mapping[str, Any],
    start: float,
    end: float,
    restart: Path | None,
    changes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    segment = merge(config, {"time_step_calculator": {"t_final": end}})
    if restart is not None:
        segment["restart"] = {
            "filename": str(restart),
            "time": start,
            "do_restart": True,
            "stitch": True,
        }
    for change in changes:
        segment = merge(segment, change)
    return segment


@sp.profile("slowcooker.run_coupled", tags=PROFILE_TAGS)
def run_coupled(
    config: Mapping[str, Any],
    couplings: Sequence[Coupling],
    times: Sequence[float],
    *,
    workdir: str | Path | None = None,
) -> xr.DataTree:
    """Run TORAX in segments, updating the config from couplings before each one.

    Args:
        config: The config of the whole run; its ``t_final`` is replaced by
            the segment ends.
        couplings: Callables ``coupling(time, state)`` that return config
            changes (`merge`) for the segment starting at ``time``; ``state``
            is the `snapshot` of the profiles there.
        times: Segment boundaries [s], increasing, the first the initial time;
            the run ends at the last.
        workdir: Where the segments' outputs go (for the restarts); a
            temporary directory if None.

    Returns:
        TORAX's output of the whole run, the segments stitched together.
    """
    times = [float(t) for t in times]
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times needs at least two increasing values")
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(workdir) if workdir is not None else Path(scratch)
        folder.mkdir(parents=True, exist_ok=True)
        # the initial state: one step of the uncoupled config, at its first time
        first = run(_segment_config(config, times[0], times[1], None, ()), max_steps=1)
        state = snapshot(first, times[0])
        restart: Path | None = None
        output = first
        for k, (start, end) in enumerate(zip(times[:-1], times[1:], strict=True)):
            changes = [dict(coupling(start, state)) for coupling in couplings]
            output = run(_segment_config(config, start, end, restart, changes))
            restart = folder / f"segment_{k}.nc"
            output.to_netcdf(restart)
            output = xr.open_datatree(restart).load()
            state = snapshot(output)
        return output


def _nested(path: str, value: Any) -> dict[str, Any]:
    """Return ``{"a": {"b": value}}`` for the path ``"a.b"``."""
    change: Any = value
    for key in reversed(path.split(".")):
        change = {key: change}
    return change


@sp.profile("slowcooker.scan", tags=PROFILE_TAGS)
def scan(config: Mapping[str, Any], path: str, values: Sequence[Any]) -> xr.Dataset:
    """Run a config for each value of one parameter and stack the profiles.

    Args:
        config: The config of every run.
        path: The parameter, keys joined by dots, e.g.
            ``"sources.generic_heat.P_total"``.
        values: Its values; numbers become the coordinate of the new dimension.

    Returns:
        `profiles` of the runs along a new dimension named after the last key of
        ``path`` (they share the time grid of a fixed time step).
    """
    name = path.split(".")[-1]
    runs = [profiles(run(merge(config, _nested(path, value)))) for value in values]
    return xr.concat(runs, dim=xr.DataArray(list(values), dims=name, name=name))
