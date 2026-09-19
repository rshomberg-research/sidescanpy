"""
Memory-bounded, ping-chunked application of correction steps to an
existing ``.ss.h5`` file, in place.

Motivated by a real field OOM (signal 9) report: applying tvg +
geometric + georef to a large real survey in one eager pass holds
several full-``(ping, channel, sample)``-shaped float arrays in memory
simultaneously — ``sidescanpy.cli.correct``'s own disk-multiplier
accounting puts that combination at ~7x the raw file size. This module
processes each frequency channel's ping range in caller-controlled
chunks instead: each chunk is loaded, corrected via the exact same
:func:`sidescanpy.processing.pipeline.apply_correction_steps` the
non-chunked path uses (not a separate numerical implementation, so
results are identical either way — see this module's own tests), and
written to a temporary per-chunk file. The chunk files are then
lazily reassembled (dask-backed, never fully materialized) and written
to the real output path via the existing
:func:`sidescanpy.io.hdf5.write_survey`, whose underlying
``Dataset.to_netcdf()`` call streams a dask-backed dataset chunk-by-chunk.

**Real, empirically-confirmed gotcha, not a guess:** dask's *default*
threaded scheduler does not stream this write safely. Measured writing
a dask-backed dataset chunked at 200 pings with the default scheduler:
peak memory reached ~3x the *total* array size (9.3GB for a 3.2GB
array) — almost certainly concurrent worker threads racing for the
underlying HDF5 file's write lock, each holding its own already-computed
chunk in memory while waiting. Forcing
``dask.config.set(scheduler="single-threaded")`` around the write fixed
this completely: measured peak memory then scales with *per-chunk*
size (confirmed across 100/500/2000-ping chunks, consistently ~2-3x the
raw chunk size, not the whole file). This module always uses that
scheduler for its own write step.

**Only steps that are genuinely per-ping-independent are supported
here**: ``tvg``, ``geometric``, ``range_correction``, ``georef`` (see
``sidescanpy.processing.pipeline.VALID_STEPS``). ``crop_farrange`` is
NOT chunkable this way — :func:`sidescanpy.processing.farrange.crop_far_range_noise`
needs to see a channel's *entire* ping range to pick one good far-range
cutoff sample per channel; request it (if needed at all) via the
regular eager path in ``sidescan-correct`` first.

2026 rshomberg@uri.edu
"""

import logging
import shutil
import tempfile
from pathlib import Path

import dask
import xarray as xr

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing.pipeline import VALID_STEPS, apply_correction_steps, step_already_done

logger = logging.getLogger(__name__)

_HDF5_ENGINE = "h5netcdf"


def correct_file_chunked(
    path: Path,
    steps: frozenset[str] | set[str] | tuple[str, ...],
    *,
    step_params: dict[str, dict] | None = None,
    ping_chunk_size: int = 500,
) -> bool:
    """Apply ``steps`` to the ``.ss.h5`` file at ``path``, in place,
    processing each frequency channel in ``ping_chunk_size``-ping chunks
    so peak memory stays proportional to that chunk size, not the whole
    file.

    Parameters
    ----------
    path : Path
        The ``.ss.h5`` file to correct, in place (same atomic-replace
        contract as the non-chunked path — a failure partway through
        never corrupts or removes the original).
    steps : set of str
        Any subset of ``sidescanpy.processing.pipeline.VALID_STEPS``
        (``tvg``, ``geometric``, ``range_correction``, ``georef``).
        Already-applied steps (per :func:`step_already_done`, checked
        per chunk) are skipped, same as the non-chunked path — a step
        already done on every ping is a fast no-op, not re-applied.
    step_params : dict, optional
        Forwarded to :func:`apply_correction_steps` for every chunk —
        see that function's own docstring.
    ping_chunk_size : int, default 500
        Pings per chunk. Smaller = tighter memory bound, more chunks
        (more per-chunk overhead, more temp files). Tune this to your
        machine's actual available memory, not a survey's actual ping
        count — see this module's own docstring for measured memory-
        vs-chunk-size numbers.

    Returns
    -------
    bool
        True if every channel had at least one step actually applied
        (i.e. the file was rewritten), False if every requested step
        was already done on every channel (no-op, file untouched).

    Raises
    ------
    ValueError
        If ``steps`` contains anything outside ``VALID_STEPS`` (e.g.
        ``"crop_farrange"`` — see module docstring for why that one
        isn't chunkable).
    """
    invalid = set(steps) - set(VALID_STEPS)
    if invalid:
        raise ValueError(
            f"correct_file_chunked doesn't support {sorted(invalid)} -- only "
            f"{VALID_STEPS} are per-ping-independent enough to chunk safely."
        )

    survey = read_survey(path, chunks={"ping": ping_chunk_size})
    tmp_dir = Path(tempfile.mkdtemp(prefix=".sidescanpy_chunked_", dir=path.parent))
    any_applied = False
    try:
        for freq_key in survey.frequencies:
            ds = survey.get_frequency(freq_key).dataset
            n_pings = ds.sizes["ping"]
            chunk_paths = []
            for start in range(0, n_pings, ping_chunk_size):
                stop = min(start + ping_chunk_size, n_pings)
                chunk_ds = ds.isel(ping=slice(start, stop)).load()

                pending = {s for s in VALID_STEPS if s in steps and not step_already_done(chunk_ds, s)}
                if pending:
                    chunk_ds = apply_correction_steps(chunk_ds, pending, step_params=step_params)
                    any_applied = True

                chunk_path = tmp_dir / f"{freq_key}_{start:012d}.nc"
                chunk_ds.to_netcdf(chunk_path, engine=_HDF5_ENGINE)
                chunk_paths.append(chunk_path)

            # Lazily reassembled (dask-backed, chunk files read back on
            # demand) -- concat_dim="ping"/combine="nested" preserves
            # the chunk order above (each chunk kept its own true
            # global `ping` coordinate values from the `isel` slice, so
            # this reconstructs the original ordering exactly, not by
            # coordinate sorting).
            combined = xr.open_mfdataset(
                chunk_paths, engine=_HDF5_ENGINE, concat_dim="ping", combine="nested",
                data_vars="all", coords="minimal", compat="override",
                chunks={"ping": ping_chunk_size},
            )
            survey.add_frequency(freq_key, combined)

        if any_applied:
            tmp_output = path.with_suffix(".tmp" + path.suffix)
            with dask.config.set(scheduler="single-threaded"):
                write_survey(survey, tmp_output, overwrite=True)
            tmp_output.replace(path)
        else:
            logger.info(f"{path.name}: every requested step already done on every channel, nothing to write")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return any_applied
