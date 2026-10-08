"""
``sidescan-convert``: convert raw survey files and target catalogs to
the format-agnostic standard model.

Two subcommands:

- ``survey`` — ``.jsf``/``.sdf``/``.xtf`` -> ``.ss.h5``, one output file per input
  file. Far-range noise is detected and cropped in memory before the
  first write (:func:`sidescanpy.processing.crop_far_range_noise`);
  ``--no-crop-farrange`` keeps the full recorded range.
- ``targets`` — reads the catalog directory with
  :func:`sidescanpy.io.read_target_catalog` (EdgeTech Discover
  ``*.target`` vs. Klein/SonarWiz ``Target_*.tgt``/``.tgd``, chosen from
  the directory's own contents, not the paired survey format). Klein
  positions can be resolved against ``.ss.h5`` files
  (:mod:`sidescanpy.targets.correlate`). Writes ``targets.json`` and
  ``targets.csv``; ``--plot`` adds ``position_scatter.png``.

Usage:
    sidescan-convert survey --input-dir data/<survey>/raw/ \\
        --output-dir data/<survey>/ss_h5/

    sidescan-convert targets --source data/<survey>/Targets/ \\
        --output data/<survey>/Targets/ --plot

    # Klein catalog, resolving position via a calibrated clock offset:
    sidescan-convert targets --source data/<survey>/TargetCatalog/ \\
        --output data/<survey>/TargetCatalog/ --ss-h5-dir data/<survey>/ss_h5/ \\
        --calibrate-offset --local-utc-offset-hours 4

2026 rshomberg@uri.edu
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from sidescanpy.cli._common import (
    add_common_args,
    add_logging_args,
    require_matplotlib,
    setup_logging_from_args,
)
from sidescanpy.cli._config import CLIParser, parse_args
from sidescanpy.io import (
    CATALOG_JSF_DISCOVER,
    detect_format,
    find_sonar_files,
    read_and_convert,
    read_target_catalog,
)
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing import crop_far_range_noise
from sidescanpy.targets.correlate import (
    calibrate_clock_offset,
    resolve_targets_nearest_ping,
    resolve_targets_offset_corrected,
)
from sidescanpy.targets.io import write_targets_csv, write_targets_json

logger = logging.getLogger(__name__)


# ============================================================
# survey
# ============================================================

def _convert_one_file(input_path: Path, *, output_dir: Path, force: bool, crop_farrange: bool = True) -> dict:
    """Convert one file; ``status`` is success / skipped / failed_convert /
    failed_process / failed_write (a failure never stops the batch)."""
    result = {"input_path": input_path, "status": "", "error": "", "cutoffs": {}}
    out_h5 = output_dir / f"{input_path.stem}.ss.h5"
    if out_h5.exists() and not force:
        result["status"] = "skipped"
        return result

    phase = "failed_convert"
    try:
        survey = read_and_convert([input_path], detect_format([input_path]), survey_name=input_path.stem)
        phase = "failed_process"
        if crop_farrange:
            survey, result["cutoffs"] = crop_far_range_noise(survey)
        phase = "failed_write"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_survey(survey, out_h5, overwrite=force)
    except Exception as e:
        result["status"], result["error"] = phase, f"{type(e).__name__}: {e}"
        return result
    result["status"] = "success"
    return result


def run_survey(args: argparse.Namespace) -> int:
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found or not a directory: {input_dir}")
        return 1
    files = find_sonar_files(input_dir, args.input_format)
    if not files:
        logger.error(f"No sonar files found in {input_dir} (format={args.input_format})")
        return 1
    logger.info(f"Found {len(files)} sonar file(s) in {input_dir}")

    start_time = time.time()
    results = []
    for i, path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {path.name}  (elapsed: {time.time() - start_time:.0f}s)")
        result = _convert_one_file(path, output_dir=output_dir, force=args.force,
                                   crop_farrange=getattr(args, "crop_farrange", True))
        results.append(result)
        if result["status"] == "success":
            cropped = {k: v for k, v in result["cutoffs"].items() if v is not None}
            logger.debug(f"    wrote .ss.h5 (far-range cropped: {cropped})" if cropped else "    wrote .ss.h5")
        elif result["status"] == "skipped":
            logger.debug("    skipped (output exists; use --force to reprocess)")
        else:
            logger.error(f"    FAILED {result['status']}: {result['error']}")

    failed = [r for r in results if r["status"].startswith("failed")]
    n_success = sum(r["status"] == "success" for r in results)
    n_skipped = sum(r["status"] == "skipped" for r in results)
    logger.info(
        f"convert survey: {n_success}/{len(files)} converted, {n_skipped} skipped, {len(failed)} failed "
        f"in {time.time() - start_time:.0f}s"
    )
    for r in failed:
        logger.warning(f"  failed: {r['input_path'].name}: {r['error']}")
    return 2 if failed else 0


# ============================================================
# targets
# ============================================================

def _resolve_klein_positions(targets: list, args: argparse.Namespace) -> list:
    """Klein targets carry no position: resolve it against ``--ss-h5-dir``."""
    offset_s = args.time_offset_s
    if args.calibrate_offset:
        logger.info("Calibrating clock offset from .ss.h5 filenames vs. first ping...")
        offset_s = calibrate_clock_offset(args.ss_h5_dir, local_utc_offset_hours=args.local_utc_offset_hours)
    if offset_s is not None:
        targets = resolve_targets_offset_corrected(targets, args.ss_h5_dir, offset_s, max_time_delta_s=args.max_time_delta_s)
        how = "offset-corrected"
    else:
        targets = resolve_targets_nearest_ping(targets, args.ss_h5_dir, max_time_delta_s=args.max_time_delta_s)
        how = "nearest-ping approximation"
    n_resolved = sum(1 for t in targets if t.latitude is not None)
    logger.info(f"Position resolved for {n_resolved}/{len(targets)} target(s) ({how})")
    return targets


def run_targets(args: argparse.Namespace) -> int:
    source, output = Path(args.source), Path(args.output)
    try:
        fmt, targets = read_target_catalog(source)
    except ValueError as e:
        logger.error(str(e))
        return 1
    logger.info(f"Read {len(targets)} target(s) from {source} ({fmt})")

    scatter_title = None
    if fmt == CATALOG_JSF_DISCOVER:
        counts = {c: sum(t.position_confidence == c for t in targets) for c in ("verified", "approximate")}
        logger.info(f"Position confidence: {counts['verified']} verified, {counts['approximate']} approximate, "
                    f"{len(targets) - sum(counts.values())} unknown")
    else:
        if args.ss_h5_dir is not None:
            targets = _resolve_klein_positions(targets, args)
        logger.info("Every target has position_confidence='unknown' (Klein catalog reader is unvalidated)")
        n_pos = sum(1 for t in targets if t.latitude is not None)
        scatter_title = f"Target positions (n={n_pos}, nearest-ping approximation)"

    output.mkdir(parents=True, exist_ok=True)
    # targets.json is the lossless source of truth; targets.csv a review copy.
    write_targets_json(targets, output / "targets.json")
    write_targets_csv(targets, output / "targets.csv")
    if args.plot:
        from sidescanpy.cli import _figures

        if not _figures.save_target_position_scatter_png(targets, str(output / "position_scatter.png"), title=scatter_title):
            logger.warning("No targets had a resolvable position; skipping position scatter")
    logger.info(f"convert targets: wrote {len(targets)} target(s) to {output / 'targets.json'} (+ targets.csv)")
    return 0


# ============================================================
# parser
# ============================================================

def build_parser() -> tuple[CLIParser, Any]:
    """The parser and its subparsers action (``.choices``: name -> subparser)."""
    parser = CLIParser(prog="sidescan-convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p = subparsers.add_parser("survey", help="Convert raw .jsf/.sdf/.xtf files to .ss.h5")
    p.add_argument("--input-dir", required=True, type=str, help="Directory of .jsf/.sdf/.xtf files (non-recursive)")
    p.add_argument("--output-dir", required=True, type=str, help="Where .ss.h5 files are written")
    p.add_argument("--input-format", choices=["jsf", "sdf", "xtf", "auto"], default="auto")
    p.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    p.add_argument(
        "--crop-farrange", action=argparse.BooleanOptionalAction, default=True,
        help="Detect and crop far-range noise before writing (default). --no-crop-farrange keeps the full "
             "recorded range; crop later with sidescan-correct --crop-farrange if wanted.",
    )
    p.set_defaults(func=run_survey)

    p = subparsers.add_parser("targets", help="Convert a target catalog (JSF Discover or Klein SonarWiz) to targets.json")
    p.add_argument("--source", required=True, type=Path, help="Directory of target catalog files")
    p.add_argument("--output", required=True, type=Path, help="Output directory for targets.json/targets.csv")
    p.add_argument(
        "--ss-h5-dir", type=Path, default=None,
        help="Klein only: directory of .ss.h5 files to resolve position from (nearest ping by timestamp)",
    )
    p.add_argument("--max-time-delta-s", type=float, default=300.0)
    p.add_argument(
        "--time-offset-s", type=float, default=None,
        help="Klein only: constant seconds to add to each target's pick_timestamp before matching, "
             "correcting a known clock-offset bug. Overridden by --calibrate-offset if both are given.",
    )
    p.add_argument(
        "--calibrate-offset", action="store_true",
        help="Klein only: auto-derive --time-offset-s from the .ss.h5 files' own filenames vs. their "
             "first ping. Requires --ss-h5-dir.",
    )
    p.add_argument(
        "--local-utc-offset-hours", type=float, default=None,
        help="Klein only: local-time-to-UTC offset in hours (hours to add to the filename's local time "
             "to get UTC). Required with --calibrate-offset; no default.",
    )
    p.add_argument("--plot", action="store_true", help="Also write position_scatter.png (needs matplotlib, the 'viz' extra)")
    p.set_defaults(func=run_targets)

    for sub in subparsers.choices.values():
        add_common_args(sub)
    add_logging_args(parser)
    return parser, subparsers


def main(argv: list[str] | None = None) -> int:
    parser, subparsers = build_parser()
    args = parse_args(parser, sys.argv[1:] if argv is None else argv, subcommands=subparsers.choices)
    if args.func is run_targets and args.calibrate_offset and args.local_utc_offset_hours is None:
        subparsers.choices["targets"].error("--calibrate-offset requires --local-utc-offset-hours (no default)")
    setup_logging_from_args(args)
    if getattr(args, "plot", False) and not require_matplotlib("sidescan-convert targets --plot"):
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
