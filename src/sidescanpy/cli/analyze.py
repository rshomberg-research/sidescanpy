"""
``sidescan-analyze``: data-investigation and TVG decay-fit diagnostics.

Two subcommands:

- ``investigate`` — promotes ``project-a/scripts/investigate_dataset.py``
  essentially unchanged: a thin driver over
  ``sidescanpy.investigation.run_investigation()`` (the
  ``sidescan-data-investigation`` skill's pipeline), characterizing an
  unknown/unfamiliar sidescan source's units, TVG status, nadir
  handling, and usable range.
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

from sidescanpy.analysis.investigation import stage3_usable_range, stage4_decay
from sidescanpy.cli._common import resolve_freq_keys
from sidescanpy.cli._config import apply_config_defaults, find_config_path, load_config
from sidescanpy.investigation import run_investigation
from sidescanpy.io.dispatch import detect_format, read_and_convert
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.pipeline import reconstruct_raw
from sidescanpy.viz.investigation import apply_stage4_fit_window_limits, plot_stage4_residuals
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

    for p in (p_inv, p_tvg):
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
        for p in (p_inv, p_tvg):
            apply_config_defaults(p, config)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
