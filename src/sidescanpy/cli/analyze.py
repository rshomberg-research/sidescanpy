"""
``sidescan-analyze``: data-investigation and TVG decay-fit diagnostics.

- ``segment`` — cut gaps/hardware-setting-changes/garbage-dropout-runs/
  attitude-instability out of a mission *before* analyzing it, via
  ``sidescanpy.processing.segments``. Every other subcommand here is
  only as trustworthy as the data it's pointed at; run this first on an
  unfamiliar or long mission, then point every subcommand below at one
  of its clean ``sectionN.ss.h5`` outputs rather than the raw file.

- ``nadir`` / ``beam-geometry`` / ``usable-range`` / ``units`` — one
  analysis each, against one frequency channel, each writing only the
  plot that is the direct evidence for its own result (not the full set
  ``investigate`` produces) -- for adjusting one ``--config`` value,
  re-running just that tool, and looking again. Each subcommand's flags
  are its analysis parameters, ``--config``-seedable like every other
  subcommand here.

- ``nav-attitude`` — plots heading/pitch/roll/altitude/depth/heave and
  the lon/lat position track for one ping range
  (``sidescanpy.analysis.navigation``), independent of backscatter
  entirely. Flags a heading/yaw value that falls outside both the
  [0,360) compass-bearing and [-180,180) signed conventions (this
  library's own convention is compass bearing, CW from N) so it can be
  visually distinguished from an ordinary survey-line turn. Meant for
  the same targeted-ping-range workflow as ``nadir``/``beam-geometry``/
  etc. above, not the full investigation pass.

- ``recommend-config`` — runs the range-decay fit (TVG term selection)
  and a ping-geometry measurement (``resolution_m``) against a clean
  section, plus ``nadir``/``beam-geometry``/``usable-range``/``units``
  for evidence, and writes one heavily-commented recommended config YAML
  (``sidescanpy.cli._recommend``) -- only the two analyses with a real
  downstream config consumer today emit a settable key; every other
  finding is reported as a comment, not invented as a key nothing
  reads. Meant to run against a clean ``sectionN.ss.h5`` from
  ``segment``, and to be reviewed/edited before use, not trusted as-is.

- ``investigate`` — runs every analysis in one pass
  (``sidescanpy.cli._investigate``) and writes the full plot set plus
  ``metrics.json`` per frequency, and ``summary.json`` -- the occasional
  full-report use case; not the day-to-day way to iterate on one value
  (see the four subcommands above). ``examples/investigate_dataset.py``
  is the same flow as a readable script.
- ``tvg-fit`` — the TVG decay-curve fit recipe: bin raw
  backscatter by slant range (matching what ``apply_tvg()`` itself
  actually uses internally, not ``ground_range_m``), auto-place the fit
  window past the near-field peak and the reference range inside it
  (``fit_range_decay(..., r_ref_m="auto", r_fit_min_m="auto")``), and
  render the decay-fit + residuals figure with
  ``set_fit_window_limits`` (robust y-limits from the fit
  window's own values; the full observed x-range always shown, so a
  misconfigured reference/window is visible rather than silently
  cropped out of frame).

Usage:
    sidescan-analyze segment --input ss_h5/mission.ss.h5 --output-dir exports/segments/

    sidescan-analyze investigate --input ss_h5/mission.ss.h5 \\
        --all-frequencies --pings 0:8000 --output-dir exports/investigation/

    sidescan-analyze tvg-fit --ss-h5 ss_h5/mission.ss.h5 \\
        --frequency 1495000 --pings 0:5000 --reconstruct-raw --output-dir exports/tvg_fit/

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import xarray as xr

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.characterize import characterize_values, infer_units
from sidescanpy.analysis.geometry import measure_ping_geometry
from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.analysis.navigation import summarize_navigation
from sidescanpy.analysis.tvg import fit_range_decay
from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.cli._common import (
    add_deprecated_alias,
    add_logging_args,
    require_matplotlib,
    resolve_freq_keys,
    setup_logging_from_args,
    uncorrected_alias,
)
from sidescanpy.cli._config import apply_config_to_subparsers, find_config_path, load_config
from sidescanpy.cli._recommend import render_recommended_config
from sidescanpy.cli._segments import write_segments, write_segments_report
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.core.history import STEP_TVG, has_step
from sidescanpy.core.survey import SideScanFrequencyChannel, SideScanSurvey
from sidescanpy.io.hdf5 import available_frequencies, load_channel_dataset, read_survey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.inversion import UNCORRECTED_VAR, invert, uncorrected_backscatter
from sidescanpy.processing.segments import find_segments
from sidescanpy.processing.slicing import slice_by_pings

logger = logging.getLogger(__name__)

# Deprecated flag / --config key spellings -> current dest names.
_RENAMED_OPTIONS = {
    "stage3_bin_width_m": "usable_range_bin_width_m",
    "stage4_bin_width_m": "decay_bin_width_m",
    "stage4_db_reference_percentile": "decay_db_reference_percentile",
    "stage4_fit_value_col": "decay_fit_value_col",
}

# Real bug, found from a real user report (2026-09-28): every lazy read
# below used to hardcode chunks={"ping": 500} -- not a multiple of
# HDF5_CHUNK_PING (the on-disk chunk size write_survey() actually
# stores, 256), so xr.open_dataset had to split real HDF5 chunks to
# match the requested virtual chunking, both degrading read performance
# and (xarray's own diagnostic) printing a UserWarning on every call.
# An exact multiple of HDF5_CHUNK_PING never splits a real chunk --
# 512 = 2x256, matching cli/export.py's own long-standing (if
# coincidental) convention for the same read pattern.
_PING_CHUNKS = {"ping": HDF5_CHUNK_PING * 2}


def _parse_ping_range(spec: str) -> tuple[int, int | None]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"must be 'start:stop', got {spec!r}")
    start_s, stop_s = spec.split(":", 1)
    return (int(start_s) if start_s else 0), (int(stop_s) if stop_s else None)


def _resolve_and_load(input_path: Path, requested_frequency: str | None, chunks: dict) -> tuple[str, xr.Dataset]:
    """``freq_key`` + its dataset from ``input_path``, which may be a
    single ``.ss.h5`` file (original behavior, byte-identical) or a
    directory of them (lazily dask-concatenated in acquisition/filename
    order, added 2026-09-27 -- see ``io.hdf5.load_channel_dataset``'s
    own docstring for why: a real mission too large to convert into one
    file at once on a memory-constrained container can be analyzed as
    several files without losing this pipeline's cross-file gap/
    instability detection).
    """
    available = available_frequencies(input_path, chunks)
    freq_key = resolve_freq_keys(available, requested_frequency)[0] if requested_frequency else sorted(available)[0]
    return freq_key, load_channel_dataset(input_path, freq_key, chunks)


def _ensure_value_vars(ds: xr.Dataset, args: argparse.Namespace) -> xr.Dataset:
    """Provide ``uncorrected`` (the uncorrected signal, inverted from
    ``processing_history``) when a ``--value-var`` asks for it (files store no
    raw copy)."""
    wanted = {getattr(args, "value_var", None), getattr(args, "garbage_value_var", None)}
    if UNCORRECTED_VAR in wanted and UNCORRECTED_VAR not in ds:
        ds = ds.assign({UNCORRECTED_VAR: uncorrected_backscatter(ds)})
    return ds


def _parse_numeric_or_literal(value: str | None) -> float | str | None:
    """--r-ref-m/--r-fit-min-m accept a literal string ('auto'/'mid_swath',
    passed through to fit_range_decay's own resolution logic) or a plain
    number."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


# ============================================================
# investigate
# ============================================================

def run_investigate(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _investigate

    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if args.frequency:
        freq_keys = [args.frequency]
    else:
        freq_keys = _investigate.list_frequencies(args.input)
        if not args.all_frequencies and len(freq_keys) > 1:
            logger.error(f"Multiple frequencies available ({freq_keys}) -- pass --frequency or --all-frequencies.")
            return 1

    channel = getattr(args, "channel", None)
    if channel is not None and channel.isdigit():
        channel = int(channel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for freq_key in freq_keys:
        logger.info(f"Investigating {args.input.name} / {freq_key} ({args.value_var})")
        row = _investigate.run(
            args.input, freq_key, args.output_dir / freq_key,
            value_var=args.value_var,
            ping_range=_parse_ping_range(args.pings),
            channel=channel,
            nadir_n_samples=args.nadir_n_samples,
            usable_range_bin_width_m=args.usable_range_bin_width_m,
            snr_threshold=args.snr_threshold,
            decay_bin_width_m=args.decay_bin_width_m,
            decay_db_reference_percentile=args.decay_db_reference_percentile,
            decay_fit_value_col=args.decay_fit_value_col,
            reject_outliers=args.reject_outliers,
            outlier_mad_k=args.outlier_mad_k,
        )
        logger.info(
            f"  {freq_key}: units={row['units']} nadir={row['nadir']} "
            f"usable_range_m={row['usable_range_m']} tvg_status={row['tvg_status']}"
        )
        summary_rows.append(row)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    logger.info(f"Done. Artifacts written to {args.output_dir}")
    return 0


# ============================================================
# tvg-fit
# ============================================================

def run_tvg_fit(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freq_key, ds = _resolve_and_load(args.ss_h5, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)

    if args.reconstruct_raw:
        ds_raw = invert(ds)
    else:
        if has_step(ds, STEP_TVG):
            logger.error(
                f"{args.ss_h5} is already TVG-corrected -- pass --reconstruct-raw to invert "
                "the recorded corrections first."
            )
            return 1
        ds_raw = ds

    ds_geo = ds_raw if "ground_range_m" in ds_raw else compute_geometric_ranges(ds_raw)

    if args.usable_range_m is not None:
        # Usable-range auto-detection fails on some Klein SDF data: hardware
        # AGC flattens backscatter so SNR never crosses the threshold.
        # --usable-range-m bypasses it for exactly this case.
        usable_range_m = args.usable_range_m
        logger.info(f"Using manually-supplied usable_range_m={usable_range_m:.2f} (auto-detection skipped)")
    else:
        usable_range = estimate_usable_range(ds_geo, range_var=args.range_var, bin_width_m=args.bin_width_m)
        usable_range_m = usable_range.usable_range_m
        if usable_range_m is None:
            logger.error(
                "Could not determine usable_range_m for this survey/ping range -- a known failure mode "
                "on AGC-flattened data, not necessarily bad data. Pass --usable-range-m explicitly."
            )
            return 1

    decay = fit_range_decay(
        ds_geo, range_var=args.range_var, bin_width_m=args.bin_width_m,
        usable_range_m=usable_range_m, r_ref_m=args.r_ref_m, r_fit_min_m=args.r_fit_min_m,
    )

    if decay.fit is None:
        logger.error(
            f"{args.ss_h5.name} {freq_key}: no decay model could be fit "
            f"(classification={decay.tvg_status}): {'; '.join(decay.notes) or 'no detail'}"
        )
        return 1

    logger.info(
        f"{args.ss_h5.name} {freq_key}: tvg-fit classification={decay.tvg_status} "
        f"best_candidate={decay.best_candidate} rmse={decay.best_rmse_db:.2f}dB",
    )
    logger.debug("=" * 70)
    logger.debug(f"{args.ss_h5.name}  {freq_key}  pings {start}:{stop if stop is not None else ''}")
    logger.debug(f"  range_var:        {args.range_var}")
    logger.debug(f"  usable_range_m:   {usable_range_m:.2f}")
    logger.debug(f"  r_ref_m:          {decay.r_ref_m:.2f}")
    logger.debug(f"  fit window:       [{decay.r_fit_min_m:.2f}, {decay.r_fit_max_m:.2f}]")
    logger.debug(f"  classification:   {decay.tvg_status}")
    logger.debug(f"  best_candidate:   {decay.best_candidate}")
    logger.debug(f"  best_rmse_db:     {decay.best_rmse_db:.3f}  (fixed-scale)")
    logger.debug(f"  n_bins_used:      {decay.n_bins_used}")
    for note in decay.notes:
        logger.debug(f"  note: {note}")
    logger.debug("=" * 70)

    from sidescanpy.cli import _figures

    out_path = args.output_dir / f"{args.ss_h5.stem}_{freq_key}_tvg_fit.png"
    _figures.save_tvg_fit_png(
        decay, out_path,
        suptitle=f"{args.ss_h5.name}  {freq_key}  pings {start}:{stop if stop is not None else ''}",
        residual_xlabel="Slant range (m)" if args.range_var == "slant_range_m" else f"{args.range_var} (m)",
        dpi=140,
    )
    logger.info(f"wrote {out_path}")

    return 0


# ============================================================
# segment
# ============================================================

def run_segment(args: argparse.Namespace) -> int:
    """Every detector in ``find_segments`` needs the full ping axis's
    arrays materialized at once (there's no per-chunk scanning logic --
    a real, separate piece of work if a survey turns out to need it,
    matching ``sidescan-correct --ping-chunk-size``'s own precedent).
    ``--pings`` bounds *which* slice gets materialized, the same
    memory-safety escape hatch every other narrow subcommand here
    already has -- default (0, 8000) matches ``investigate``'s own
    default, not "the whole file," since a real production `.ss.h5` can
    be many GB.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_segments_overview

    if not (args.input.is_file() or args.input.is_dir()):
        logger.error(f"Input not found: {args.input}")
        return 1

    all_freq_keys = available_frequencies(args.input, _PING_CHUNKS)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)

    segments = find_segments(
        ds,
        gap_factor=args.gap_factor,
        time_gap_factor=args.time_gap_factor,
        sound_vel_rel_tol=args.sound_vel_rel_tol,
        garbage_value_var=args.garbage_value_var,
        garbage_min_run_pings=args.garbage_min_run_pings,
        garbage_degenerate_frac=args.garbage_degenerate_frac,
        stability_window_pings=args.stability_window_pings,
        stability_threshold=args.stability_threshold,
        stability_min_run_pings=args.stability_min_run_pings,
        min_segment_pings=args.min_segment_pings,
    )

    # write_segments() needs every frequency channel sliced to the exact
    # same [start:stop] window find_segments' own (0-indexed-from-start)
    # ping numbers are relative to -- passing the original, full-range
    # dataset here would silently write the wrong pings whenever
    # `start != 0`. Works identically whether args.input is a single file
    # (source_files/source_format/survey_name taken from it directly) or
    # a directory (every *.ss.h5 in it, "ss_h5" as a reasonable
    # source_format label -- see io.hdf5.load_channel_dataset).
    if args.input.is_dir():
        source_files = sorted(args.input.glob("*.ss.h5"))
        source_format, survey_name = "ss_h5", args.input.name
    else:
        _meta_survey = read_survey(args.input, chunks=_PING_CHUNKS)
        source_files = _meta_survey.source_files
        source_format, survey_name = _meta_survey.source_format, _meta_survey.survey_name
    sliced_survey = SideScanSurvey(source_files=source_files, source_format=source_format, survey_name=survey_name)
    for fk in all_freq_keys:
        channel_ds = load_channel_dataset(args.input, fk, _PING_CHUNKS)
        channel = SideScanFrequencyChannel(channel_ds, freq_key=fk)
        sliced = slice_by_pings(channel_ds, start=start, stop=stop if stop is not None else channel.n_pings)
        sliced_survey.add_frequency(fk, sliced.load())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_segments_report(segments, args.output_dir / "segments.json")
    written = write_segments(sliced_survey, segments, args.output_dir)

    logger.info(f"{args.input.name} / {freq_key}: {ds.sizes['ping']} pings -> {len(segments)} segment(s)")
    for seg in segments:
        logger.debug(f"  {seg.label}: [{seg.start}:{seg.stop}] {seg.status} ({', '.join(seg.reasons)})")
    logger.info(f"Wrote {len(written)} clean segment .ss.h5 file(s) + segments.json to {args.output_dir}")

    fig, _ = plot_segments_overview(ds, segments)
    overview_path = args.output_dir / "overview.png"
    _figures.save_figure(fig, overview_path, dpi=120)
    logger.debug(f"wrote {overview_path}")

    return 0


# ============================================================
# nadir
# ============================================================

def run_nadir(args: argparse.Namespace) -> int:
    """Classify + locate nadir, writing only the
    zoomed boundary-detection plot by default (the direct evidence for
    the classification) -- ``--verbose-diagnostics`` also writes the
    full-survey waterfall boundary overlay.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_nadir_waterfall, plot_nadir_zoom

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    nadir = analyze_nadir(
        ds_geo, value_var=args.value_var,
        nadir_n_samples=args.nadir_n_samples,
        strong_return_min_sample_buffer=args.strong_return_min_sample_buffer,
        strong_return_peak_percentile=args.strong_return_peak_percentile,
        min_first_bottom_return_sample=args.min_first_bottom_return_sample,
        boundary_smoothing_window=args.boundary_smoothing_window,
    )

    logger.info(f"{args.input.name} / {freq_key}: nadir = {nadir.nadir_type}")
    for key in ("altitude_sample_median", "first_bottom_return_sample_median", "altitude_to_bottom_gap_sample_median"):
        if getattr(nadir, key) is not None:
            logger.debug(f"  {key}: {getattr(nadir, key):.2f}")
    for note in nadir.notes:
        logger.debug(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(nadir.to_dict(), f, indent=2, default=str)

    fig, _ = plot_nadir_zoom(ds_geo, args.value_var, nadir=nadir, nadir_n_samples=args.nadir_n_samples)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_nadir_zoom.png"
    _figures.save_figure(fig, out_path, dpi=140)
    logger.debug(f"wrote {out_path}")

    if args.verbose_diagnostics:
        fig, _ = plot_nadir_waterfall(ds_geo, args.value_var, nadir=nadir)
        wf_path = args.output_dir / f"{args.input.stem}_{freq_key}_nadir_waterfall.png"
        _figures.save_figure(fig, wf_path, dpi=140)
        logger.debug(f"wrote {wf_path}")

    return 0


# ============================================================
# beam-geometry
# ============================================================

def run_beam_geometry(args: argparse.Namespace) -> int:
    """Triangulate the beam's near/far edges from the nadir boundary and
    the usable range (both run here as prerequisites -- use the
    ``nadir``/``usable-range`` subcommands to tune either). Writes only
    the beam-geometry diagram, the direct evidence for the result.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_beam_geometry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    nadir = analyze_nadir(ds_geo, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples)
    usable_range = estimate_usable_range(
        ds_geo, value_var=args.value_var, bin_width_m=args.usable_range_bin_width_m, snr_threshold=args.snr_threshold,
    )
    beam = estimate_beam_geometry(
        ds_geo, nadir.first_bottom_return_sample, usable_range.usable_range_m,
        near_angle_max_std_deg=args.near_angle_max_std_deg, min_valid_fraction=args.min_valid_fraction,
    )

    logger.info(f"{args.input.name} / {freq_key}: beam_geometry = {beam.confidence}")
    for key in ("near_angle_deg_median", "far_angle_deg", "beam_center_angle_deg", "beam_width_deg"):
        if getattr(beam, key) is not None:
            logger.debug(f"  {key}: {getattr(beam, key):.2f}")
    for note in beam.notes:
        logger.debug(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(beam.to_dict(), f, indent=2, default=str)

    fig, _ = plot_beam_geometry(ds_geo, beam, value_var=args.value_var)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_beam_geometry.png"
    _figures.save_figure(fig, out_path, dpi=140)
    logger.debug(f"wrote {out_path}")

    return 0


# ============================================================
# usable-range
# ============================================================

def run_usable_range(args: argparse.Namespace) -> int:
    """Bin backscatter by range, find the usable-range boundary, and write
    only the median/std-vs-range plot -- the direct evidence for the number.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_usable_range

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)
    ds_geo = ds if args.range_var in ds else compute_geometric_ranges(ds)

    usable_range = estimate_usable_range(
        ds_geo, value_var=args.value_var, range_var=args.range_var, bin_width_m=args.bin_width_m,
        snr_threshold=args.snr_threshold, reject_outliers=args.reject_outliers, outlier_mad_k=args.outlier_mad_k,
    )

    usable_range_m = usable_range.usable_range_m
    logger.info(f"{args.input.name} / {freq_key}: usable_range_m = {usable_range_m}")
    for note in usable_range.notes:
        logger.debug(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(usable_range.to_dict(), f, indent=2, default=str)

    fig, _ = plot_usable_range(usable_range)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_usable_range.png"
    _figures.save_figure(fig, out_path, dpi=140)
    logger.debug(f"wrote {out_path}")

    return 0


# ============================================================
# units
# ============================================================

def run_units(args: argparse.Namespace) -> int:
    """Classify units from the raw value-range signature and write only
    the distribution histogram -- the direct evidence for the call.
    Doesn't cross-check against a range-decay fit (``investigate`` does)
    -- run ``tvg-fit`` separately if that matters for a source.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_units_distribution

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)

    values = characterize_values(ds, value_var=args.value_var)
    units = infer_units(values)

    logger.info(f"{args.input.name} / {freq_key}: units = {units.units}")
    for note in units.notes:
        logger.debug(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(units.to_dict(), f, indent=2, default=str)

    fig, _ = plot_units_distribution(ds, args.value_var)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_units.png"
    _figures.save_figure(fig, out_path, dpi=140)
    logger.debug(f"wrote {out_path}")

    return 0


# ============================================================
# nav-attitude
# ============================================================

def run_nav_attitude(args: argparse.Namespace) -> int:
    """Navigation/attitude diagnostic (heading/pitch/roll/altitude/
    depth/heave/position) for one ping range -- independent of
    backscatter entirely, so unlike the other single-analysis subcommands
    above it needs neither ``--value-var`` nor geometric ranges. Point
    it at the specific ping range a heading/attitude anomaly was
    reported in and look at the trace directly.
    """
    from sidescanpy.cli import _figures
    from sidescanpy.viz.analysis import plot_navigation_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)

    result = summarize_navigation(ds, heading_var=args.heading_var)

    logger.info(f"{args.input.name} / {freq_key}: {args.heading_var} classification = {result.heading_convention}")
    for note in result.notes:
        logger.debug(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    fig, _ = plot_navigation_summary(ds, ping_offset=start, result=result)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_nav_attitude.png"
    _figures.save_figure(fig, out_path, dpi=140)
    logger.debug(f"wrote {out_path}")

    return 0


# ============================================================
# recommend-config
# ============================================================

def run_recommend_config(args: argparse.Namespace) -> int:
    """Run every analysis that has a real downstream config consumer today
    (the range-decay fit's TVG term selection, plus the ping-geometry
    measurement behind ``resolution_m``) against one clean section, plus
    the informational-only analyses (nadir/beam-geometry/usable-range/units)
    for evidence -- and write one heavily-commented recommended config
    YAML (``cli._recommend.render_recommended_config``). Meant to
    run against an already-segmented, clean ``sectionN.ss.h5`` (see
    ``segment``) -- running it against a raw, unsegmented mission mixes
    whatever heterogeneity that file has into every number below.
    """
    if not (args.input.is_file() or args.input.is_dir()):
        logger.error(f"Input not found: {args.input}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    freq_key, ds = _resolve_and_load(args.input, args.frequency, _PING_CHUNKS)
    start, stop = args.pings
    ds = _ensure_value_vars(ds.isel(ping=slice(start, stop)).load(), args)
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    values = characterize_values(ds_geo, value_var=args.value_var)
    units = infer_units(values)
    nadir = analyze_nadir(ds_geo, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples)
    usable_range = estimate_usable_range(
        ds_geo, value_var=args.value_var, range_var="slant_range_m",
        bin_width_m=args.bin_width_m, snr_threshold=args.snr_threshold,
    )
    try:
        beam = estimate_beam_geometry(ds_geo, nadir.first_bottom_return_sample, usable_range.usable_range_m)
    except (ValueError, KeyError) as exc:
        logger.warning(f"beam-geometry could not be computed ({exc}) -- omitting from the recommendation")
        beam = None

    usable_range_m = usable_range.usable_range_m
    decay = None
    if usable_range_m is not None:
        decay = fit_range_decay(
            ds_geo, range_var="slant_range_m", bin_width_m=args.bin_width_m,
            usable_range_m=usable_range_m, r_ref_m="auto", r_fit_min_m="auto",
        )

    geometry = measure_ping_geometry(ds_geo)

    text = render_recommended_config(
        source_name=args.input.name, freq_key=freq_key, ping_range=(start, stop), value_var=args.value_var,
        values=values, nadir=nadir, beam=beam, usable_range=usable_range, decay=decay, units=units,
        geometry=geometry,
    )
    args.output.write_text(text)

    logger.info(
        f"{args.input.name} / {freq_key}: tvg_status="
        f"{decay.tvg_status if decay else None}  nadir={nadir.nadir_type}  "
        f"usable_range_m={usable_range_m}  ping_spacing_m={geometry.ping_spacing_m_median}"
    )
    logger.info(f"wrote {args.output}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_inv = subparsers.add_parser("investigate", help="Characterize units/TVG status/nadir/usable range for an unfamiliar data source")
    p_inv.add_argument("--input", required=True, type=Path, help="Path to a .ss.h5, .jsf, or .sdf file")
    freq_group = p_inv.add_mutually_exclusive_group()
    freq_group.add_argument("--frequency", type=str, help="e.g. hz_1495000")
    freq_group.add_argument("--all-frequencies", action="store_true")
    p_inv.add_argument("--value-var", default="uncorrected", type=uncorrected_alias("backscatter_raw"), help="Variable to analyze. uncorrected = the uncorrected signal (inverted from processing_history; no raw copy is stored). 'backscatter_raw' is a deprecated alias")
    p_inv.add_argument("--pings", type=str, default="0:8000", help="Ping range 'start:stop' (stop empty = all)")
    p_inv.add_argument(
        "--channel", default=None,
        help="Restrict the usable-range and range-decay binning to one side: port/starboard or 0/1 "
             "(default: pool both sides)",
    )
    p_inv.add_argument("--nadir-n-samples", type=int, default=50)
    p_inv.add_argument("--usable-range-bin-width-m", type=float, default=2.0)
    p_inv.add_argument("--decay-bin-width-m", type=float, default=5.0)
    p_inv.add_argument("--decay-db-reference-percentile", type=float, default=99.0)
    p_inv.add_argument("--snr-threshold", type=float, default=3.0)
    p_inv.add_argument("--reject-outliers", action="store_true")
    p_inv.add_argument("--outlier-mad-k", type=float, default=3.5)
    p_inv.add_argument("--decay-fit-value-col", choices=["mean", "median"], default="mean")
    add_deprecated_alias(p_inv, "--stage3-bin-width-m", "--usable-range-bin-width-m", type=float)
    add_deprecated_alias(p_inv, "--stage4-bin-width-m", "--decay-bin-width-m", type=float)
    add_deprecated_alias(p_inv, "--stage4-db-reference-percentile", "--decay-db-reference-percentile", type=float)
    add_deprecated_alias(p_inv, "--stage4-fit-value-col", "--decay-fit-value-col", choices=["mean", "median"])
    p_inv.add_argument("--output-dir", required=True, type=Path)
    p_inv.set_defaults(func=run_investigate)

    p_tvg = subparsers.add_parser("tvg-fit", help="TVG decay-curve fit + residuals, using the adopted auto-window recipe")
    p_tvg.add_argument(
        "--ss-h5", required=True, type=Path,
        help="Path to a .ss.h5 file, or a directory of them (lazily concatenated, same convention as "
             "sidescan-export's own --ss-h5-dir)",
    )
    p_tvg.add_argument(
        "--frequency", type=str, required=True,
        help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
             "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
    )
    p_tvg.add_argument("--pings", type=_parse_ping_range, default=(0, 5000))
    p_tvg.add_argument("--reconstruct-raw", action="store_true", help="Invert the recorded corrections first (needed for an already-corrected .ss.h5)")
    p_tvg.add_argument("--bin-width-m", type=float, default=1.0)
    p_tvg.add_argument("--r-ref-m", type=_parse_numeric_or_literal, default="auto", help="Number, 'auto' (default), or 'mid_swath'")
    p_tvg.add_argument("--r-fit-min-m", type=_parse_numeric_or_literal, default="auto", help="Number or 'auto' (default)")
    p_tvg.add_argument("--range-var", default="slant_range_m", choices=("slant_range_m", "ground_range_m"))
    p_tvg.add_argument(
        "--usable-range-m", type=float, default=None,
        help="Bypass usable_range_m auto-detection and use this value instead -- needed on some "
             "Klein SDF data where AGC flattens backscatter enough that the SNR threshold never "
             "trips; not normally needed for JSF.",
    )
    p_tvg.add_argument("--output-dir", required=True, type=Path)
    p_tvg.set_defaults(func=run_tvg_fit)

    p_seg = subparsers.add_parser(
        "segment", help="Cut gaps/setting-changes/garbage/unstable stretches out of a mission before analyzing it",
    )
    p_seg.add_argument(
        "--input", required=True, type=Path,
        help="Path to a .ss.h5 file, or a directory of them (lazily concatenated, same convention as "
             "sidescan-export's own --ss-h5-dir)",
    )
    p_seg.add_argument(
        "--frequency", type=str, default=None,
        help="Which frequency channel to compute segmentation from (default: first, sorted). "
             "The resulting ping ranges are applied to every frequency channel when writing.",
    )
    p_seg.add_argument(
        "--pings", type=_parse_ping_range, default=(0, 8000),
        help="'start:stop' (stop empty = all). Every detector needs this whole window's arrays in "
             "memory at once -- bound it explicitly on a large real .ss.h5 rather than risk loading "
             "the whole file. Default matches investigate's own default, not 'the whole file'.",
    )
    p_seg.add_argument("--gap-factor", type=float, default=8.0, help="Along-track distance jump, as a multiple of the survey's own typical ping spacing, that counts as a gap")
    p_seg.add_argument("--time-gap-factor", type=float, default=8.0, help="Raw ping-timestamp jump, as a multiple of the survey's own typical ping interval, that counts as a gap (independent of the distance check -- either can fire)")
    p_seg.add_argument("--sound-vel-rel-tol", type=float, default=0.05, help="Relative step in sound_vel_ms between consecutive pings that counts as a setting change")
    p_seg.add_argument("--garbage-value-var", type=uncorrected_alias("backscatter_raw"), default="uncorrected")
    p_seg.add_argument("--garbage-min-run-pings", type=int, default=20)
    p_seg.add_argument("--garbage-degenerate-frac", type=float, default=0.999)
    p_seg.add_argument("--stability-window-pings", type=int, default=100)
    p_seg.add_argument("--stability-threshold", type=float, default=1.0)
    p_seg.add_argument("--stability-min-run-pings", type=int, default=50)
    p_seg.add_argument("--min-segment-pings", type=int, default=200)
    p_seg.add_argument("--output-dir", required=True, type=Path)
    p_seg.set_defaults(func=run_segment)

    def _common_single_channel_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--input", required=True, type=Path,
            help="Path to a .ss.h5 file, or a directory of them (lazily concatenated, same convention "
                 "as sidescan-export's own --ss-h5-dir)",
        )
        p.add_argument(
            "--frequency", type=str, required=True,
            help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
                 "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
        )
        p.add_argument("--pings", type=_parse_ping_range, default=(0, 8000), help="'start:stop' (stop empty = all)")
        p.add_argument("--value-var", default="uncorrected", type=uncorrected_alias("backscatter_raw"), help="Variable to analyze. uncorrected = the uncorrected signal (inverted from processing_history; no raw copy is stored). 'backscatter_raw' is a deprecated alias")
        p.add_argument("--output-dir", required=True, type=Path)

    p_nadir = subparsers.add_parser("nadir", help="Classify + locate nadir")
    _common_single_channel_args(p_nadir)
    p_nadir.add_argument("--nadir-n-samples", type=int, default=50)
    p_nadir.add_argument("--strong-return-min-sample-buffer", type=float, default=0.0)
    p_nadir.add_argument("--strong-return-peak-percentile", type=float, default=100.0)
    p_nadir.add_argument("--min-first-bottom-return-sample", type=float, default=None)
    p_nadir.add_argument("--boundary-smoothing-window", type=int, default=15)
    p_nadir.add_argument(
        "--verbose-diagnostics", action="store_true",
        help="Also write the full-survey waterfall boundary overlay, not just the zoomed plot",
    )
    p_nadir.set_defaults(func=run_nadir)

    p_beam = subparsers.add_parser("beam-geometry", help="Triangulate the beam's near/far edges")
    _common_single_channel_args(p_beam)
    p_beam.add_argument("--nadir-n-samples", type=int, default=50)
    p_beam.add_argument("--usable-range-bin-width-m", type=float, default=2.0)
    add_deprecated_alias(p_beam, "--stage3-bin-width-m", "--usable-range-bin-width-m", type=float)
    p_beam.add_argument("--snr-threshold", type=float, default=3.0)
    p_beam.add_argument("--near-angle-max-std-deg", type=float, default=15.0)
    p_beam.add_argument("--min-valid-fraction", type=float, default=0.2)
    p_beam.set_defaults(func=run_beam_geometry)

    p_ur = subparsers.add_parser("usable-range", help="Find the usable-range boundary")
    _common_single_channel_args(p_ur)
    p_ur.add_argument("--range-var", default="slant_range_m", choices=("slant_range_m", "ground_range_m"))
    p_ur.add_argument("--bin-width-m", type=float, default=2.0)
    p_ur.add_argument("--snr-threshold", type=float, default=3.0)
    p_ur.add_argument("--reject-outliers", action="store_true")
    p_ur.add_argument("--outlier-mad-k", type=float, default=3.5)
    p_ur.set_defaults(func=run_usable_range)

    p_units = subparsers.add_parser("units", help="Classify raw-value units from their range signature")
    _common_single_channel_args(p_units)
    p_units.set_defaults(func=run_units)

    p_nav = subparsers.add_parser(
        "nav-attitude",
        help="Plot heading/pitch/roll/altitude/depth/heave + position track for a ping range",
    )
    p_nav.add_argument(
        "--input", required=True, type=Path,
        help="Path to a .ss.h5 file, or a directory of them (lazily concatenated, same convention "
             "as sidescan-export's own --ss-h5-dir)",
    )
    p_nav.add_argument(
        "--frequency", type=str, default=None,
        help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
             "surveys, or a literal freq_key for any other convention -- optional here since "
             "nav/attitude fields are identical across every frequency channel of the same file "
             "(default: first, sorted)",
    )
    p_nav.add_argument("--pings", type=_parse_ping_range, default=(0, 8000), help="'start:stop' (stop empty = all)")
    p_nav.add_argument(
        "--heading-var", type=str, default="heading_deg",
        help="Which field to run the compass/signed convention check against",
    )
    p_nav.add_argument("--output-dir", required=True, type=Path)
    p_nav.set_defaults(func=run_nav_attitude)

    p_rec = subparsers.add_parser(
        "recommend-config",
        help="Write one evidence-commented recommended config YAML for a clean section (TVG terms + resolution_m; "
             "nadir/beam-geometry/usable-range/units reported as evidence only, no config flag consumes them yet)",
    )
    p_rec.add_argument(
        "--input", required=True, type=Path,
        help="Path to a clean (already-segmented) .ss.h5 file, or a directory of them (lazily "
             "concatenated, same convention as sidescan-export's own --ss-h5-dir)",
    )
    p_rec.add_argument(
        "--frequency", type=str, required=True,
        help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
             "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
    )
    p_rec.add_argument("--pings", type=_parse_ping_range, default=(0, None), help="'start:stop' (stop empty = all -- default: the whole section)")
    p_rec.add_argument("--value-var", default="uncorrected", type=uncorrected_alias("backscatter_raw"), help="Variable to analyze. uncorrected = the uncorrected signal (inverted from processing_history; no raw copy is stored). 'backscatter_raw' is a deprecated alias")
    p_rec.add_argument("--nadir-n-samples", type=int, default=50)
    p_rec.add_argument("--bin-width-m", type=float, default=1.0, help="Shared by usable-range and the adopted TVG-fit recipe, matching tvg-fit's own default")
    p_rec.add_argument("--snr-threshold", type=float, default=3.0)
    p_rec.add_argument("--output", required=True, type=Path, help="Where to write the recommended config YAML")
    p_rec.set_defaults(func=run_recommend_config)

    for p in (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_nav, p_rec):
        p.add_argument("--config", type=Path, default=None, help="YAML or JSON file seeding this subcommand's defaults -- explicit flags still override")
        p.add_argument("--strict-config", action="store_true", help="Treat config-file keys this subcommand does not recognize as an error instead of a warning")
        # default=SUPPRESS: see sidescanpy/cli/convert.py's matching
        # block for why a plain default=False here would silently
        # overwrite a real -v already set before the subcommand.
        add_logging_args(p, subcommand=True)

    add_logging_args(parser)

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        config = load_config(config_path)
        apply_config_to_subparsers(
            (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_nav, p_rec), config, argv,
            renamed_keys=_RENAMED_OPTIONS,
        )

    args = parser.parse_args(argv)

    setup_logging_from_args(args)

    if args.func is not run_recommend_config and not require_matplotlib(f"sidescan-analyze {args.subcommand}"):
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
