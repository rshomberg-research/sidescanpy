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

- ``nav-attitude`` (alias ``nav-summary``) — plots heading/pitch/roll/altitude/depth/heave and
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

Every subcommand writes its result as JSON (``metrics.json`` from the
result's ``to_dict()``; ``segment`` writes ``segments.json``,
``investigate`` ``summary.json`` plus ``<freq>/metrics.json``,
``recommend-config`` its YAML) and, unless ``--no-plots``, the evidence
PNG(s) via :mod:`sidescanpy.viz.analysis` (needs the ``viz`` extra).

Usage:
    sidescan-analyze segment --input ss_h5/mission.ss.h5 --output-dir exports/segments/

    sidescan-analyze investigate --input ss_h5/mission.ss.h5 \\
        --all-frequencies --pings 0:8000 --output-dir exports/investigation/

    sidescan-analyze tvg-fit --ss-h5 ss_h5/mission.ss.h5 \\
        --frequency 1495000 --pings 0:5000 --reconstruct-raw --output-dir exports/tvg_fit/

2026 rshomberg@uri.edu
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import xarray as xr

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.characterize import characterize_values, infer_units
from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.analysis.navigation import summarize_navigation
from sidescanpy.analysis.tvg import fit_range_decay
from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.cli._common import (
    add_common_args,
    add_deprecated_alias,
    add_logging_args,
    add_plots_arg,
    require_matplotlib,
    resolve_freq_keys,
    save_plot,
    setup_logging_from_args,
    uncorrected_alias,
    write_json,
)
from sidescanpy.cli._config import CLIParser, parse_args
from sidescanpy.cli._recommend import render_recommended_config, run_analyses
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.core.history import STEP_TVG, has_step
from sidescanpy.io.hdf5 import available_frequencies, load_channel_dataset, read_survey_pings
from sidescanpy.processing import (
    UNCORRECTED_VAR,
    compute_geometric_ranges,
    find_segments,
    invert,
    with_uncorrected_var,
    write_segments,
)

logger = logging.getLogger(__name__)

# Deprecated flag / --config key spellings -> current dest names.
_RENAMED_OPTIONS = {
    "stage3_bin_width_m": "usable_range_bin_width_m",
    "stage4_bin_width_m": "decay_bin_width_m",
    "stage4_db_reference_percentile": "decay_db_reference_percentile",
    "stage4_fit_value_col": "decay_fit_value_col",
}

# A multiple of HDF5_CHUNK_PING (the on-disk chunk size) so lazy reads never
# split a real HDF5 chunk.
_PING_CHUNKS = {"ping": HDF5_CHUNK_PING * 2}


def _parse_ping_range(spec: str) -> tuple[int, int | None]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"must be 'start:stop', got {spec!r}")
    start_s, stop_s = spec.split(":", 1)
    return (int(start_s) if start_s else 0), (int(stop_s) if stop_s else None)


def _parse_numeric_or_literal(value: str | None) -> float | str | None:
    """A number, or a literal ('auto'/'mid_swath') for fit_range_decay to resolve."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _load(args: argparse.Namespace, *, path: Path | None = None, geometry_var: str | None = None) -> tuple[str, xr.Dataset]:
    """``(freq_key, ds)`` for ``--frequency`` (default: first sorted) at
    ``--input`` (a ``.ss.h5`` or a directory of them), cut to ``--pings`` and
    loaded; adds ``uncorrected`` when a ``--value-var`` asks for it, and
    geometric ranges when ``geometry_var`` is missing."""
    path = path if path is not None else args.input
    available = available_frequencies(path, _PING_CHUNKS)
    freq_key = resolve_freq_keys(available, args.frequency)[0] if args.frequency else sorted(available)[0]
    start, stop = args.pings
    ds = load_channel_dataset(path, freq_key, _PING_CHUNKS).isel(ping=slice(start, stop)).load()
    if UNCORRECTED_VAR in {getattr(args, "value_var", None), getattr(args, "garbage_value_var", None)}:
        ds = with_uncorrected_var(ds)
    if geometry_var is not None and geometry_var not in ds:
        ds = compute_geometric_ranges(ds)
    return freq_key, ds


def _finish(args: argparse.Namespace, freq_key: str, result: Any, summary: str, fields=()) -> Path:
    """Log the one-line result (INFO) and its details (DEBUG); write ``metrics.json``."""
    logger.info(f"{args.input.name} / {freq_key}: {summary}")
    for key in fields:
        if getattr(result, key) is not None:
            logger.debug(f"  {key}: {getattr(result, key):.2f}")
    for note in result.notes:
        logger.debug(f"  note: {note}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return write_json(result, args.output_dir / "metrics.json")


def _png(args: argparse.Namespace, freq_key: str, name: str) -> Path:
    return args.output_dir / f"{args.input.stem}_{freq_key}_{name}.png"


# ============================================================
# investigate
# ============================================================

def run_investigate(args: argparse.Namespace) -> int:
    """Every analysis in one pass, per frequency (see ``cli/_investigate.py``)."""
    from sidescanpy.cli import _investigate

    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        return 1
    freq_keys = [args.frequency] if args.frequency else _investigate.list_frequencies(args.input)
    if len(freq_keys) > 1 and not args.all_frequencies:
        logger.error(f"Multiple frequencies available ({freq_keys}) -- pass --frequency or --all-frequencies.")
        return 1
    channel = int(args.channel) if args.channel is not None and args.channel.isdigit() else args.channel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for freq_key in freq_keys:
        row = _investigate.run(
            args.input, freq_key, args.output_dir / freq_key, plots=getattr(args, "plots", True),
            value_var=args.value_var, ping_range=_parse_ping_range(args.pings), channel=channel,
            nadir_n_samples=args.nadir_n_samples, usable_range_bin_width_m=args.usable_range_bin_width_m,
            snr_threshold=args.snr_threshold, decay_bin_width_m=args.decay_bin_width_m,
            decay_db_reference_percentile=args.decay_db_reference_percentile,
            decay_fit_value_col=args.decay_fit_value_col,
            reject_outliers=args.reject_outliers, outlier_mad_k=args.outlier_mad_k,
        )
        logger.info(
            f"{args.input.name} / {freq_key}: units={row['units']} nadir={row['nadir']} "
            f"usable_range_m={row['usable_range_m']} tvg_status={row['tvg_status']}"
        )
        rows.append(row)
    write_json(rows, args.output_dir / "summary.json")
    return 0


# ============================================================
# tvg-fit
# ============================================================

def run_tvg_fit(args: argparse.Namespace) -> int:
    """Range-decay fit (``fit_range_decay``, auto window + reference) on the
    uncorrected signal; writes ``metrics.json`` and the fit + residuals PNG."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_key, ds = _load(args, path=args.ss_h5)
    if args.reconstruct_raw:
        ds = invert(ds)
    elif has_step(ds, STEP_TVG):
        logger.error(f"{args.ss_h5} is already TVG-corrected -- pass --reconstruct-raw to invert the recorded corrections first.")
        return 1
    ds = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    usable_range_m = args.usable_range_m
    if usable_range_m is None:
        usable_range_m = estimate_usable_range(ds, range_var=args.range_var, bin_width_m=args.bin_width_m).usable_range_m
        if usable_range_m is None:
            logger.error(
                "Could not determine usable_range_m for this survey/ping range -- a known failure mode "
                "on AGC-flattened data, not necessarily bad data. Pass --usable-range-m explicitly."
            )
            return 1
    decay = fit_range_decay(
        ds, range_var=args.range_var, bin_width_m=args.bin_width_m,
        usable_range_m=usable_range_m, r_ref_m=args.r_ref_m, r_fit_min_m=args.r_fit_min_m,
    )
    if decay.fit is None:
        logger.error(f"{args.ss_h5.name} {freq_key}: no decay model could be fit "
                     f"(classification={decay.tvg_status}): {'; '.join(decay.notes) or 'no detail'}")
        return 1

    start, stop = args.pings
    logger.info(f"{args.ss_h5.name} {freq_key}: tvg-fit classification={decay.tvg_status} "
                f"best_candidate={decay.best_candidate} rmse={decay.best_rmse_db:.2f}dB")
    logger.debug(f"  usable_range_m={usable_range_m:.2f} r_ref_m={decay.r_ref_m:.2f} fit window="
                 f"[{decay.r_fit_min_m:.2f}, {decay.r_fit_max_m:.2f}] n_bins_used={decay.n_bins_used}")
    write_json(decay, args.output_dir / "metrics.json")
    if args.plots:
        from sidescanpy.cli import _figures

        out_path = args.output_dir / f"{args.ss_h5.stem}_{freq_key}_tvg_fit.png"
        _figures.save_tvg_fit_png(
            decay, out_path,
            suptitle=f"{args.ss_h5.name}  {freq_key}  pings {start}:{stop if stop is not None else ''}",
            residual_xlabel="Slant range (m)" if args.range_var == "slant_range_m" else f"{args.range_var} (m)",
            dpi=140,
        )
        logger.debug(f"wrote {out_path}")
    return 0


# ============================================================
# segment
# ============================================================

def run_segment(args: argparse.Namespace) -> int:
    """``find_segments`` on one channel; writes each clean segment (every
    frequency) as ``<label>.ss.h5`` plus ``segments.json``. Every detector
    needs the whole ``--pings`` window in memory, so bound it on a big file."""
    if not (args.input.is_file() or args.input.is_dir()):
        logger.error(f"Input not found: {args.input}")
        return 1
    freq_key, ds = _load(args)
    segments = find_segments(
        ds, gap_factor=args.gap_factor, time_gap_factor=args.time_gap_factor,
        sound_vel_rel_tol=args.sound_vel_rel_tol, garbage_value_var=args.garbage_value_var,
        garbage_min_run_pings=args.garbage_min_run_pings, garbage_degenerate_frac=args.garbage_degenerate_frac,
        stability_window_pings=args.stability_window_pings, stability_threshold=args.stability_threshold,
        stability_min_run_pings=args.stability_min_run_pings, min_segment_pings=args.min_segment_pings,
    )
    # Segment ping numbers are relative to --pings' start: cut every channel the same way.
    survey = read_survey_pings(args.input, *args.pings, chunks=_PING_CHUNKS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"segments": [s.to_dict() for s in segments]}, args.output_dir / "segments.json")
    written = write_segments(survey, segments, args.output_dir)
    for seg in segments:
        logger.debug(f"  {seg.label}: [{seg.start}:{seg.stop}] {seg.status} ({', '.join(seg.reasons)})")
    logger.info(f"{args.input.name} / {freq_key}: {ds.sizes['ping']} pings -> {len(segments)} segment(s), "
                f"{len(written)} clean .ss.h5 written to {args.output_dir}")
    if args.plots:
        from sidescanpy.viz.analysis import plot_segments_overview

        save_plot(plot_segments_overview(ds, segments)[0], args.output_dir / "overview.png", dpi=120)
    return 0


# ============================================================
# single analyses
# ============================================================

def run_nadir(args: argparse.Namespace) -> int:
    """Classify + locate nadir; plots the zoomed boundary detection
    (``--verbose-diagnostics`` adds the full waterfall overlay)."""
    freq_key, ds = _load(args, geometry_var="ground_range_m")
    nadir = analyze_nadir(
        ds, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples,
        strong_return_min_sample_buffer=args.strong_return_min_sample_buffer,
        strong_return_peak_percentile=args.strong_return_peak_percentile,
        min_first_bottom_return_sample=args.min_first_bottom_return_sample,
        boundary_smoothing_window=args.boundary_smoothing_window,
    )
    _finish(args, freq_key, nadir, f"nadir = {nadir.nadir_type}",
            ("altitude_sample_median", "first_bottom_return_sample_median", "altitude_to_bottom_gap_sample_median"))
    if args.plots:
        from sidescanpy.viz.analysis import plot_nadir_waterfall, plot_nadir_zoom

        fig, _ = plot_nadir_zoom(ds, args.value_var, nadir=nadir, nadir_n_samples=args.nadir_n_samples)
        save_plot(fig, _png(args, freq_key, "nadir_zoom"))
        if args.verbose_diagnostics:
            save_plot(plot_nadir_waterfall(ds, args.value_var, nadir=nadir)[0], _png(args, freq_key, "nadir_waterfall"))
    return 0


def run_beam_geometry(args: argparse.Namespace) -> int:
    """Beam near/far edges from the nadir boundary and usable range (both
    computed here with defaults; tune them with ``nadir``/``usable-range``)."""
    freq_key, ds = _load(args, geometry_var="ground_range_m")
    nadir = analyze_nadir(ds, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples)
    usable_range = estimate_usable_range(
        ds, value_var=args.value_var, bin_width_m=args.usable_range_bin_width_m, snr_threshold=args.snr_threshold,
    )
    beam = estimate_beam_geometry(
        ds, nadir.first_bottom_return_sample, usable_range.usable_range_m,
        near_angle_max_std_deg=args.near_angle_max_std_deg, min_valid_fraction=args.min_valid_fraction,
    )
    _finish(args, freq_key, beam, f"beam_geometry = {beam.confidence}",
            ("near_angle_deg_median", "far_angle_deg", "beam_center_angle_deg", "beam_width_deg"))
    if args.plots:
        from sidescanpy.viz.analysis import plot_beam_geometry

        save_plot(plot_beam_geometry(ds, beam, value_var=args.value_var)[0], _png(args, freq_key, "beam_geometry"))
    return 0


def run_usable_range(args: argparse.Namespace) -> int:
    """Usable-range boundary from binned backscatter vs. range."""
    freq_key, ds = _load(args, geometry_var=args.range_var)
    usable_range = estimate_usable_range(
        ds, value_var=args.value_var, range_var=args.range_var, bin_width_m=args.bin_width_m,
        snr_threshold=args.snr_threshold, reject_outliers=args.reject_outliers, outlier_mad_k=args.outlier_mad_k,
    )
    _finish(args, freq_key, usable_range, f"usable_range_m = {usable_range.usable_range_m}")
    if args.plots:
        from sidescanpy.viz.analysis import plot_usable_range

        save_plot(plot_usable_range(usable_range)[0], _png(args, freq_key, "usable_range"))
    return 0


def run_units(args: argparse.Namespace) -> int:
    """Units from the value-range signature alone (``investigate`` also
    cross-checks against a range-decay fit)."""
    freq_key, ds = _load(args)
    units = infer_units(characterize_values(ds, value_var=args.value_var))
    _finish(args, freq_key, units, f"units = {units.units}")
    if args.plots:
        from sidescanpy.viz.analysis import plot_units_distribution

        save_plot(plot_units_distribution(ds, args.value_var)[0], _png(args, freq_key, "units"))
    return 0


def run_nav_attitude(args: argparse.Namespace) -> int:
    """Navigation/attitude summary (``summarize_navigation``) for one ping range."""
    freq_key, ds = _load(args)
    result = summarize_navigation(ds, heading_var=args.heading_var)
    _finish(args, freq_key, result, f"{args.heading_var} classification = {result.heading_convention}")
    if args.plots:
        from sidescanpy.viz.analysis import plot_navigation_summary

        fig, _ = plot_navigation_summary(ds, ping_offset=args.pings[0], result=result)
        save_plot(fig, _png(args, freq_key, "nav_attitude"))
    return 0


def run_recommend_config(args: argparse.Namespace) -> int:
    """Evidence-commented recommended config YAML for one clean section
    (``cli/_recommend.py``); meant for a ``segment`` output, reviewed before use."""
    if not (args.input.is_file() or args.input.is_dir()):
        logger.error(f"Input not found: {args.input}")
        return 1
    freq_key, ds = _load(args, geometry_var="ground_range_m")
    results = run_analyses(ds, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples,
                           bin_width_m=args.bin_width_m, snr_threshold=args.snr_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_recommended_config(
        source_name=args.input.name, freq_key=freq_key, ping_range=args.pings, value_var=args.value_var, **results,
    ))
    decay = results["decay"]
    logger.info(
        f"{args.input.name} / {freq_key}: tvg_status={decay.tvg_status if decay else None}  "
        f"nadir={results['nadir'].nadir_type}  usable_range_m={results['usable_range'].usable_range_m}  "
        f"ping_spacing_m={results['geometry'].ping_spacing_m_median} -> {args.output}"
    )
    return 0


def build_parser() -> tuple[CLIParser, Any]:
    """The parser and its subparsers action (``.choices``: name -> subparser)."""
    parser = CLIParser(prog="sidescan-analyze", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_inv = subparsers.add_parser("investigate", help="Characterize units/TVG status/nadir/usable range for an unfamiliar data source")
    p_inv.add_argument("--input", required=True, type=Path, help="Path to a .ss.h5, .jsf, .sdf or .xtf file")
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
        "nav-attitude", aliases=["nav-summary"],
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

    for p in (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_nav):
        add_plots_arg(p)
    for p in (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_nav, p_rec):
        add_common_args(p)
    add_logging_args(parser)
    return parser, subparsers


def main(argv: list[str] | None = None) -> int:
    parser, subparsers = build_parser()
    args = parse_args(
        parser, sys.argv[1:] if argv is None else argv, subcommands=subparsers.choices, renamed_keys=_RENAMED_OPTIONS,
    )
    setup_logging_from_args(args)
    if getattr(args, "plots", False) and not require_matplotlib(f"sidescan-analyze {args.subcommand}"):
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
