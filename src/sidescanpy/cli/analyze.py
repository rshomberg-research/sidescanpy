"""
``sidescan-analyze``: data-investigation and TVG decay-fit diagnostics.

- ``segment`` — cut gaps/hardware-setting-changes/garbage-dropout-runs/
  attitude-instability out of a mission *before* analyzing it, via
  ``sidescanpy.analysis.segmentation``. Every other subcommand here is
  only as trustworthy as the data it's pointed at; run this first on an
  unfamiliar or long mission, then point every subcommand below at one
  of its clean ``sectionN.ss.h5`` outputs rather than the raw file.

- ``nadir`` / ``beam-geometry`` / ``usable-range`` / ``units`` — each of
  ``run_investigation()``'s stages 2 / 2.5 / 3 / 1+5, runnable on its
  own against one frequency channel, each writing only the plot(s) that
  are the direct evidence for its own classification (not the full
  per-stage diagnostic set ``investigate`` below produces) -- meant for
  the iterative workflow of adjusting one ``--config`` value, re-running
  just that one tool, and looking at the result before adjusting again,
  rather than re-running the whole five-stage pass every time. Each
  subcommand's own flags are its analysis parameters (seeds/thresholds),
  already ``--config``-seedable via the same YAML loader every other
  subcommand here uses.

- ``recommend-config`` — runs Stage 4 (TVG term selection) and a new
  ping-geometry measurement (``resolution_m``) against a clean section,
  plus ``nadir``/``beam-geometry``/``usable-range``/``units`` for
  evidence, and writes one heavily-commented recommended config YAML
  (``sidescanpy.analysis.recommend``) -- only the two stages with a
  real downstream config consumer today emit a settable key; every
  other finding is reported as a comment, not invented as a key nothing
  reads. Meant to run against a clean ``sectionN.ss.h5`` from
  ``segment``, and to be reviewed/edited before use, not trusted as-is.

- ``investigate`` — promotes ``project-a/scripts/investigate_dataset.py``
  essentially unchanged: a thin driver over
  ``sidescanpy.investigation.run_investigation()`` (the
  ``sidescan-data-investigation`` skill's pipeline), running every stage
  in one pass and writing the full diagnostic set per stage -- the
  occasional full-report use case; not the default day-to-day way to
  iterate on one specific value (see the four subcommands above for
  that).
- ``tvg-fit`` — the TVG decay-curve fit recipe adopted after an extended
  real-data investigation (see ``sidescanpy``'s own ``CLAUDE.md``,
  "v0.14.0 follow-up 1" through "11" for the full history): bin raw
  backscatter by slant range (matching what ``apply_tvg()`` itself
  actually uses internally, not ``ground_range_m``), auto-place the fit
  window past the near-field peak and the reference range inside it
  (``stage4_decay(..., r_ref_m="auto", r_fit_min_m="auto")`` — the
  library-native successor to the manual recipe
  ``project-a/scripts/debug/plot_official_tvg_fit.py`` prototyped;
  that script's exploratory predecessors
  (``compare_tvg_fit_recipes.py``/``validate_tvg_fit_flatness.py``/
  ``tvg_range_residual_diagnostic.py``) are kept in that project as
  historical record of how this recipe was found, not promoted here —
  they were one-time comparison tools, not an ongoing capability), and
  render the decay-fit + residuals figure with
  ``apply_stage4_fit_window_limits`` (robust y-limits from the fit
  window's own values; the full observed x-range always shown, so a
  misconfigured reference/window is visible rather than silently
  cropped out of frame).

Usage:
    sidescan-analyze segment --input data/active/unknown/ss_h5/20241017181822.ss.h5 \\
        --output-dir data/active/unknown/exports/segments/

    sidescan-analyze investigate --input data/active/unknown/ss_h5/20241017181822.ss.h5 \\
        --all-frequencies --pings 0:8000 \\
        --output-dir data/active/unknown/exports/investigation/

    sidescan-analyze tvg-fit --ss-h5 data/active/unknown/ss_h5/20241017181822.ss.h5 \\
        --frequency 1495000 --pings 0:5000 --reconstruct-raw \\
        --output-dir data/active/unknown/exports/tvg_fit/

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy.analysis.investigation import (
    stage1_characterize,
    stage2_5_beam_geometry,
    stage2_nadir,
    stage3_usable_range,
    stage4_decay,
    stage5_units,
)
from sidescanpy.analysis.recommend import measure_ping_geometry, render_recommended_config
from sidescanpy.analysis.segmentation import find_segments, write_segments, write_segments_report
from sidescanpy.cli._common import resolve_freq_keys
from sidescanpy.cli._config import apply_config_defaults, find_config_path, load_config
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.investigation import run_investigation
from sidescanpy.io.dispatch import detect_format, read_and_convert
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.pipeline import reconstruct_raw
from sidescanpy.processing.slicing import slice_by_pings
from sidescanpy.viz.investigation import (
    apply_stage4_fit_window_limits,
    plot_segments_overview,
    plot_stage1_histograms,
    plot_stage2_5_beam_geometry,
    plot_stage2_nadir_zoom,
    plot_stage2_waterfall,
    plot_stage3_usable_range,
    plot_stage4_residuals,
)
from sidescanpy.viz.tvg_diagnostics import plot_candidate_fit

logger = logging.getLogger(__name__)

_HDF5_SUFFIXES = (".h5", ".hdf5")


def _parse_ping_range(spec: str) -> tuple[int, int | None]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"must be 'start:stop', got {spec!r}")
    start_s, stop_s = spec.split(":", 1)
    return (int(start_s) if start_s else 0), (int(stop_s) if stop_s else None)


def _parse_numeric_or_literal(value: str | None) -> float | str | None:
    """--r-ref-m/--r-fit-min-m accept a literal string ('auto'/'mid_swath',
    passed through to stage4_decay's own resolution logic) or a plain
    number."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


# ============================================================
# investigate (promoted from project-a/scripts/investigate_dataset.py)
# ============================================================

def _list_frequencies(input_path: Path) -> list[str]:
    if input_path.suffix.lower() in _HDF5_SUFFIXES or input_path.name.endswith(".ss.h5"):
        survey = read_survey(input_path)
    else:
        fmt = detect_format([input_path])
        survey = read_and_convert([input_path], fmt)
    return sorted(survey.frequencies)


def run_investigate(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        return 1

    ping_range = _parse_ping_range(args.pings)

    if args.frequency:
        freq_keys = [args.frequency]
    else:
        freq_keys = _list_frequencies(args.input)
        if not args.all_frequencies and len(freq_keys) > 1:
            logger.error(f"Multiple frequencies available ({freq_keys}) — pass --frequency or --all-frequencies.")
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for freq_key in freq_keys:
        logger.info(f"Investigating {args.input.name} / {freq_key} ({args.value_var})")
        result = run_investigation(
            args.input,
            freq_key=freq_key,
            value_var=args.value_var,
            ping_range=ping_range,
            output_dir=args.output_dir / freq_key,
            nadir_n_samples=args.nadir_n_samples,
            stage3_bin_width_m=args.stage3_bin_width_m,
            stage4_bin_width_m=args.stage4_bin_width_m,
            stage4_db_reference_percentile=args.stage4_db_reference_percentile,
            snr_threshold=args.snr_threshold,
            reject_outliers=args.reject_outliers,
            outlier_mad_k=args.outlier_mad_k,
            stage4_fit_value_col=args.stage4_fit_value_col,
        )
        summary_rows.append(result.summary_row())

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    logger.info("Summary:")
    for row in summary_rows:
        logger.info(f"  {row}")
    logger.info(f"Done. Artifacts written to {args.output_dir}")
    return 0


# ============================================================
# tvg-fit
# ============================================================

def run_tvg_fit(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    survey = read_survey(args.ss_h5, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()

    if args.reconstruct_raw:
        ds_raw = reconstruct_raw(ds)
    else:
        if ds.attrs.get("processing_level") != "raw":
            logger.error(
                f"{args.ss_h5} is processing_level={ds.attrs.get('processing_level')!r}, not 'raw' -- "
                "pass --reconstruct-raw to rebuild a raw-equivalent dataset from backscatter_raw."
            )
            return 1
        ds_raw = ds

    ds_geo = ds_raw if "ground_range_m" in ds_raw else compute_geometric_ranges(ds_raw)

    if args.usable_range_m is not None:
        # Stage 3's SNR-threshold auto-detection has a known failure
        # mode on Klein SDF data: real-time hardware AGC flattens
        # backscatter enough that SNR vs. the far-range noise floor
        # never crosses threshold at any range, so usable_range_m comes
        # back None even on perfectly good data (confirmed across every
        # Klein channel checked in sidescanpy's own investigation
        # history -- see its CLAUDE.md). --usable-range-m bypasses
        # Stage 3 entirely for exactly this case.
        usable_range_m = args.usable_range_m
        logger.info(f"Using manually-supplied usable_range_m={usable_range_m:.2f} (Stage 3 auto-detection skipped)")
    else:
        stage3 = stage3_usable_range(ds_geo, range_var=args.range_var, bin_width_m=args.bin_width_m)
        usable_range_m = stage3.metrics.get("usable_range_m")
        if usable_range_m is None:
            logger.error(
                "Stage 3 could not determine usable_range_m for this survey/ping range -- "
                "a known Klein SDF failure mode (see sidescanpy's own CLAUDE.md), not necessarily "
                "bad data. Pass --usable-range-m explicitly to bypass Stage 3's auto-detection."
            )
            return 1

    stage4 = stage4_decay(
        ds_geo, range_var=args.range_var, bin_width_m=args.bin_width_m,
        usable_range_m=usable_range_m, r_ref_m=args.r_ref_m, r_fit_min_m=args.r_fit_min_m,
    )
    m = stage4.metrics

    logger.info("=" * 70)
    logger.info(f"{args.ss_h5.name}  {freq_key}  pings {start}:{stop if stop is not None else ''}")
    logger.info(f"  range_var:        {args.range_var}")
    logger.info(f"  usable_range_m:   {usable_range_m:.2f}")
    logger.info(f"  r_ref_m:          {m['r_ref_m']:.2f}")
    logger.info(f"  fit window:       [{m['r_fit_min_m']:.2f}, {m['r_fit_max_m']:.2f}]")
    logger.info(f"  classification:   {stage4.classification}")
    logger.info(f"  best_candidate:   {m['best_candidate']}")
    logger.info(f"  best_rmse_db:     {m['best_rmse_db']:.3f}  (fixed-scale)")
    logger.info(f"  n_bins_used:      {m['n_bins_used']}")
    for note in stage4.notes:
        logger.info(f"  note: {note}")
    logger.info("=" * 70)

    fit_result = stage4.data["fit_result"]

    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for i, c in enumerate(fit_result.candidates):
        plot_candidate_fit(c, fit_result.observed_range_m, fit_result.observed_loss_db, ax=ax_fit, show_observed=(i == 0))
    ax_fit.set_title(f"Decay fit — best: {m['best_candidate']}, RMSE={m['best_rmse_db']:.2f}dB")
    ax_fit.set_xlabel("")

    plot_stage4_residuals(stage4, ax=ax_res, relative_x=False)
    ax_res.set_title("Residuals")
    ax_res.set_xlabel("Slant range (m)" if args.range_var == "slant_range_m" else f"{args.range_var} (m)")

    apply_stage4_fit_window_limits(ax_fit, stage4)

    fig.suptitle(f"{args.ss_h5.name}  {freq_key}  pings {start}:{stop if stop is not None else ''}")
    fig.tight_layout()
    out_path = args.output_dir / f"{args.ss_h5.stem}_{freq_key}_tvg_fit.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
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
    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        return 1

    survey = read_survey(args.input, chunks={"ping": 500})
    if args.frequency:
        freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    else:
        freq_key = sorted(survey.frequencies)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()

    segments = find_segments(
        ds,
        gap_factor=args.gap_factor,
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
    # `survey` here would silently write the wrong pings whenever
    # `start != 0`.
    sliced_survey = SideScanSurvey(
        source_files=survey.source_files, source_format=survey.source_format, survey_name=survey.survey_name,
    )
    for fk in survey.frequencies:
        channel = survey.get_frequency(fk)
        sliced = slice_by_pings(channel, start=start, stop=stop if stop is not None else channel.n_pings)
        sliced_survey.add_frequency(fk, sliced.dataset.load())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_segments_report(segments, args.output_dir / "segments.json")
    written = write_segments(sliced_survey, segments, args.output_dir)

    logger.info(f"{args.input.name} / {freq_key}: {ds.sizes['ping']} pings -> {len(segments)} segment(s)")
    for seg in segments:
        logger.info(f"  {seg.label}: [{seg.start}:{seg.stop}] {seg.status} ({', '.join(seg.reasons)})")
    logger.info(f"Wrote {len(written)} clean segment .ss.h5 file(s) + segments.json to {args.output_dir}")

    fig, _ = plot_segments_overview(ds, segments)
    overview_path = args.output_dir / "overview.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    logger.info(f"wrote {overview_path}")

    return 0


# ============================================================
# nadir
# ============================================================

def run_nadir(args: argparse.Namespace) -> int:
    """Stage 2 on its own: classify + locate nadir, write only the
    zoomed boundary-detection plot by default (the direct evidence for
    the classification) -- ``--verbose-diagnostics`` also writes the
    full-survey waterfall boundary overlay.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    survey = read_survey(args.input, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    stage2 = stage2_nadir(
        ds_geo, value_var=args.value_var,
        nadir_n_samples=args.nadir_n_samples,
        strong_return_min_sample_buffer=args.strong_return_min_sample_buffer,
        strong_return_peak_percentile=args.strong_return_peak_percentile,
        min_first_bottom_return_sample=args.min_first_bottom_return_sample,
        boundary_smoothing_window=args.boundary_smoothing_window,
    )

    logger.info(f"{args.input.name} / {freq_key}: nadir = {stage2.classification}")
    for key in ("altitude_sample_median", "first_bottom_return_sample_median", "altitude_to_bottom_gap_sample_median"):
        if key in stage2.metrics:
            logger.info(f"  {key}: {stage2.metrics[key]:.2f}")
    for note in stage2.notes:
        logger.info(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(stage2.to_json_dict(), f, indent=2, default=str)

    fig, _ = plot_stage2_nadir_zoom(ds_geo, args.value_var, stage2=stage2, nadir_n_samples=args.nadir_n_samples)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_nadir_zoom.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    if args.verbose_diagnostics:
        fig, _ = plot_stage2_waterfall(ds_geo, args.value_var, stage2=stage2)
        wf_path = args.output_dir / f"{args.input.stem}_{freq_key}_nadir_waterfall.png"
        fig.savefig(wf_path, dpi=140)
        plt.close(fig)
        logger.info(f"wrote {wf_path}")

    return 0


# ============================================================
# beam-geometry
# ============================================================

def run_beam_geometry(args: argparse.Namespace) -> int:
    """Stage 2.5 on its own: triangulate the beam's near/far edges from
    Stage 2's nadir boundary and Stage 3's usable range (both run here
    as prerequisites, not separately configurable beyond the flags
    below -- run the ``nadir``/``usable-range`` subcommands directly
    first if either needs closer tuning on its own). Writes only the
    beam-geometry diagram, the direct evidence for the classification.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    survey = read_survey(args.input, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    stage2 = stage2_nadir(ds_geo, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples)
    stage3 = stage3_usable_range(
        ds_geo, value_var=args.value_var, bin_width_m=args.stage3_bin_width_m, snr_threshold=args.snr_threshold,
    )
    stage2_5 = stage2_5_beam_geometry(
        ds_geo, stage2, stage3, value_var=args.value_var,
        near_angle_max_std_deg=args.near_angle_max_std_deg, min_valid_fraction=args.min_valid_fraction,
    )

    logger.info(f"{args.input.name} / {freq_key}: beam_geometry = {stage2_5.classification}")
    for key in ("near_angle_deg_median", "far_angle_deg", "beam_center_angle_deg", "beam_width_deg"):
        if stage2_5.metrics.get(key) is not None:
            logger.info(f"  {key}: {stage2_5.metrics[key]:.2f}")
    for note in stage2_5.notes:
        logger.info(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(stage2_5.to_json_dict(), f, indent=2, default=str)

    fig, _ = plot_stage2_5_beam_geometry(ds_geo, stage2_5, value_var=args.value_var)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_beam_geometry.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    return 0


# ============================================================
# usable-range
# ============================================================

def run_usable_range(args: argparse.Namespace) -> int:
    """Stage 3 on its own: bin backscatter by range, find the usable-
    range boundary, write only the median/std-vs-range plot -- the
    direct evidence for the number.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    survey = read_survey(args.input, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()
    ds_geo = ds if args.range_var in ds else compute_geometric_ranges(ds)

    stage3 = stage3_usable_range(
        ds_geo, value_var=args.value_var, range_var=args.range_var, bin_width_m=args.bin_width_m,
        snr_threshold=args.snr_threshold, reject_outliers=args.reject_outliers, outlier_mad_k=args.outlier_mad_k,
    )

    usable_range_m = stage3.metrics.get("usable_range_m")
    logger.info(f"{args.input.name} / {freq_key}: usable_range_m = {usable_range_m}")
    for note in stage3.notes:
        logger.info(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(stage3.to_json_dict(), f, indent=2, default=str)

    fig, _ = plot_stage3_usable_range(stage3)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_usable_range.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    return 0


# ============================================================
# units
# ============================================================

def run_units(args: argparse.Namespace) -> int:
    """Stage 1 + Stage 5 on their own: classify units from the raw
    value-range signature, write only the distribution histogram -- the
    direct evidence for the classification. Doesn't cross-check against
    a Stage 4 TVG-fit result (``run_investigation()``'s fuller version
    can) -- run ``tvg-fit`` separately if that cross-check matters for
    a specific source.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    survey = read_survey(args.input, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()

    stage1 = stage1_characterize(ds, value_var=args.value_var)
    stage5 = stage5_units(stage1)

    logger.info(f"{args.input.name} / {freq_key}: units = {stage5.classification}")
    for note in stage5.notes:
        logger.info(f"  note: {note}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(stage5.to_json_dict(), f, indent=2, default=str)

    fig, _ = plot_stage1_histograms(ds, args.value_var)
    out_path = args.output_dir / f"{args.input.stem}_{freq_key}_units.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    return 0


# ============================================================
# recommend-config
# ============================================================

def run_recommend_config(args: argparse.Namespace) -> int:
    """Run every stage that has a real downstream config consumer today
    (Stage 4 TVG term selection, plus the ping-geometry measurement
    behind ``resolution_m``) against one clean section, plus the
    informational-only stages (nadir/beam-geometry/usable-range/units)
    for evidence -- and write one heavily-commented recommended config
    YAML (``analysis.recommend.render_recommended_config``). Meant to
    run against an already-segmented, clean ``sectionN.ss.h5`` (see
    ``segment``) -- running it against a raw, unsegmented mission mixes
    whatever heterogeneity that file has into every number below.
    """
    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    survey = read_survey(args.input, chunks={"ping": 500})
    freq_key = resolve_freq_keys(survey.frequencies, args.frequency)[0]
    ds = survey.get_frequency(freq_key).dataset
    start, stop = args.pings
    ds = ds.isel(ping=slice(start, stop)).load()
    ds_geo = ds if "ground_range_m" in ds else compute_geometric_ranges(ds)

    stage1 = stage1_characterize(ds_geo, value_var=args.value_var)
    stage5 = stage5_units(stage1)
    stage2 = stage2_nadir(ds_geo, value_var=args.value_var, nadir_n_samples=args.nadir_n_samples)
    stage3 = stage3_usable_range(
        ds_geo, value_var=args.value_var, range_var="slant_range_m",
        bin_width_m=args.bin_width_m, snr_threshold=args.snr_threshold,
    )
    try:
        stage2_5 = stage2_5_beam_geometry(ds_geo, stage2, stage3, value_var=args.value_var)
    except (ValueError, KeyError) as exc:
        logger.warning(f"beam-geometry could not be computed ({exc}) -- omitting from the recommendation")
        stage2_5 = None

    usable_range_m = stage3.metrics.get("usable_range_m")
    stage4 = None
    if usable_range_m is not None:
        stage4 = stage4_decay(
            ds_geo, range_var="slant_range_m", bin_width_m=args.bin_width_m,
            usable_range_m=usable_range_m, r_ref_m="auto", r_fit_min_m="auto",
        )

    geometry = measure_ping_geometry(ds_geo)

    text = render_recommended_config(
        source_name=args.input.name, freq_key=freq_key, ping_range=(start, stop), value_var=args.value_var,
        stage1=stage1, stage2=stage2, stage2_5=stage2_5, stage3=stage3, stage4=stage4, stage5=stage5,
        geometry=geometry,
    )
    args.output.write_text(text)

    logger.info(
        f"{args.input.name} / {freq_key}: tvg_status="
        f"{stage4.classification if stage4 else None}  nadir={stage2.classification}  "
        f"usable_range_m={usable_range_m}  ping_spacing_m={geometry.get('ping_spacing_m_median')}"
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
    p_inv.add_argument("--value-var", default="backscatter_raw")
    p_inv.add_argument("--pings", type=str, default="0:8000", help="Ping range 'start:stop' (stop empty = all)")
    p_inv.add_argument("--nadir-n-samples", type=int, default=50)
    p_inv.add_argument("--stage3-bin-width-m", type=float, default=2.0)
    p_inv.add_argument("--stage4-bin-width-m", type=float, default=5.0)
    p_inv.add_argument("--stage4-db-reference-percentile", type=float, default=99.0)
    p_inv.add_argument("--snr-threshold", type=float, default=3.0)
    p_inv.add_argument("--reject-outliers", action="store_true")
    p_inv.add_argument("--outlier-mad-k", type=float, default=3.5)
    p_inv.add_argument("--stage4-fit-value-col", choices=["mean", "median"], default="mean")
    p_inv.add_argument("--output-dir", required=True, type=Path)
    p_inv.set_defaults(func=run_investigate)

    p_tvg = subparsers.add_parser("tvg-fit", help="TVG decay-curve fit + residuals, using the adopted auto-window recipe")
    p_tvg.add_argument("--ss-h5", required=True, type=Path)
    p_tvg.add_argument(
        "--frequency", type=str, required=True,
        help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
             "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
    )
    p_tvg.add_argument("--pings", type=_parse_ping_range, default=(0, 5000))
    p_tvg.add_argument("--reconstruct-raw", action="store_true", help="Rebuild processing_level='raw' from backscatter_raw (needed for an already-pipelined .ss.h5)")
    p_tvg.add_argument("--bin-width-m", type=float, default=1.0)
    p_tvg.add_argument("--r-ref-m", type=_parse_numeric_or_literal, default="auto", help="Number, 'auto' (default), or 'mid_swath'")
    p_tvg.add_argument("--r-fit-min-m", type=_parse_numeric_or_literal, default="auto", help="Number or 'auto' (default)")
    p_tvg.add_argument("--range-var", default="slant_range_m", choices=("slant_range_m", "ground_range_m"))
    p_tvg.add_argument(
        "--usable-range-m", type=float, default=None,
        help="Bypass Stage 3's usable_range_m auto-detection and use this value instead -- "
             "needed on some Klein SDF data where AGC flattens backscatter enough that Stage 3's "
             "SNR threshold never trips (see sidescanpy's own CLAUDE.md); not normally needed for JSF.",
    )
    p_tvg.add_argument("--output-dir", required=True, type=Path)
    p_tvg.set_defaults(func=run_tvg_fit)

    p_seg = subparsers.add_parser(
        "segment", help="Cut gaps/setting-changes/garbage/unstable stretches out of a mission before analyzing it",
    )
    p_seg.add_argument("--input", required=True, type=Path, help="Path to a .ss.h5 file")
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
    p_seg.add_argument("--sound-vel-rel-tol", type=float, default=0.05, help="Relative step in sound_vel_ms between consecutive pings that counts as a setting change")
    p_seg.add_argument("--garbage-value-var", type=str, default="backscatter_raw")
    p_seg.add_argument("--garbage-min-run-pings", type=int, default=20)
    p_seg.add_argument("--garbage-degenerate-frac", type=float, default=0.999)
    p_seg.add_argument("--stability-window-pings", type=int, default=100)
    p_seg.add_argument("--stability-threshold", type=float, default=1.0)
    p_seg.add_argument("--stability-min-run-pings", type=int, default=50)
    p_seg.add_argument("--min-segment-pings", type=int, default=200)
    p_seg.add_argument("--output-dir", required=True, type=Path)
    p_seg.set_defaults(func=run_segment)

    def _common_single_channel_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", required=True, type=Path, help="Path to a .ss.h5 file")
        p.add_argument(
            "--frequency", type=str, required=True,
            help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
                 "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
        )
        p.add_argument("--pings", type=_parse_ping_range, default=(0, 8000), help="'start:stop' (stop empty = all)")
        p.add_argument("--value-var", default="backscatter_raw")
        p.add_argument("--output-dir", required=True, type=Path)

    p_nadir = subparsers.add_parser("nadir", help="Classify + locate nadir (Stage 2 on its own)")
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

    p_beam = subparsers.add_parser("beam-geometry", help="Triangulate the beam's near/far edges (Stage 2.5 on its own)")
    _common_single_channel_args(p_beam)
    p_beam.add_argument("--nadir-n-samples", type=int, default=50)
    p_beam.add_argument("--stage3-bin-width-m", type=float, default=2.0)
    p_beam.add_argument("--snr-threshold", type=float, default=3.0)
    p_beam.add_argument("--near-angle-max-std-deg", type=float, default=15.0)
    p_beam.add_argument("--min-valid-fraction", type=float, default=0.2)
    p_beam.set_defaults(func=run_beam_geometry)

    p_ur = subparsers.add_parser("usable-range", help="Find the usable-range boundary (Stage 3 on its own)")
    _common_single_channel_args(p_ur)
    p_ur.add_argument("--range-var", default="slant_range_m", choices=("slant_range_m", "ground_range_m"))
    p_ur.add_argument("--bin-width-m", type=float, default=2.0)
    p_ur.add_argument("--snr-threshold", type=float, default=3.0)
    p_ur.add_argument("--reject-outliers", action="store_true")
    p_ur.add_argument("--outlier-mad-k", type=float, default=3.5)
    p_ur.set_defaults(func=run_usable_range)

    p_units = subparsers.add_parser("units", help="Classify raw-value units from their range signature (Stage 1+5 on their own)")
    _common_single_channel_args(p_units)
    p_units.set_defaults(func=run_units)

    p_rec = subparsers.add_parser(
        "recommend-config",
        help="Write one evidence-commented recommended config YAML for a clean section (TVG terms + resolution_m; "
             "nadir/beam-geometry/usable-range/units reported as evidence only, no config flag consumes them yet)",
    )
    p_rec.add_argument("--input", required=True, type=Path, help="Path to a clean (already-segmented) .ss.h5 file")
    p_rec.add_argument(
        "--frequency", type=str, required=True,
        help="freq_hz_start (e.g. 1495000, resolved to freq_key 'hz_1495000') for JSF/SDF-style "
             "surveys, or a literal freq_key (e.g. 'sensor1_data1') for any other convention",
    )
    p_rec.add_argument("--pings", type=_parse_ping_range, default=(0, None), help="'start:stop' (stop empty = all -- default: the whole section)")
    p_rec.add_argument("--value-var", default="backscatter_raw")
    p_rec.add_argument("--nadir-n-samples", type=int, default=50)
    p_rec.add_argument("--bin-width-m", type=float, default=1.0, help="Shared by usable-range and the adopted TVG-fit recipe, matching tvg-fit's own default")
    p_rec.add_argument("--snr-threshold", type=float, default=3.0)
    p_rec.add_argument("--output", required=True, type=Path, help="Where to write the recommended config YAML")
    p_rec.set_defaults(func=run_recommend_config)

    for p in (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_rec):
        p.add_argument("--config", type=Path, default=None, help="YAML or JSON file seeding this subcommand's defaults -- explicit flags still override")
        # default=SUPPRESS: see sidescanpy/cli/convert.py's matching
        # block for why a plain default=False here would silently
        # overwrite a real -v already set before the subcommand.
        p.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Also accepted after the subcommand")

    parser.add_argument("-v", "--verbose", action="store_true")

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        config = load_config(config_path)
        for p in (p_inv, p_tvg, p_seg, p_nadir, p_beam, p_ur, p_units, p_rec):
            apply_config_defaults(p, config)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
