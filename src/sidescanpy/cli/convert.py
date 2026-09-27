"""
``sidescan-convert``: convert raw survey files and target catalogs to
the format-agnostic standard model.

Two subcommands, replacing the old per-project ``01_convert.py`` and
``11_read_targets.py`` pipeline scripts (confirmed byte-identical
between `project-b` and `project-a`'s own copies of ``01_convert.py``
except docstring text; ``11_read_targets.py`` genuinely differs by
format — 80 lines for JSF's Discover SQLite catalog, 310 for Klein's
``.tgt``/``.tgd``, the latter's real clock-offset-calibration logic now
promoted into ``sidescanpy.io.sdf.targets`` rather than living only in
one consumer project's copy):

- ``survey`` — ``.jsf``/``.sdf`` -> ``.ss.h5``, one output file per input
  file, far-range noise detected/cropped in memory before the first
  write.
- ``targets`` — dispatches on the target-catalog directory's own
  contents (EdgeTech Discover ``*.target`` SQLite files vs. Klein/
  SonarWiz ``Target_*.tgt``/``.tgd`` pairs), not on the paired survey
  format, per ``sidescanpy``'s own target-catalog-dispatch recommendation
  — nothing stops hand-labeling a JSF survey in SonarWiz or vice versa.

Usage:
    sidescan-convert survey --input-dir data/active/unknown/raw/ \\
        --output-dir data/active/unknown/ss_h5/

    sidescan-convert targets --source data/active/unknown/Targets/ \\
        --output data/active/unknown/Targets/

    # Klein catalog, resolving position via a calibrated clock offset:
    sidescan-convert targets \\
        --source "data/active/survey-b/TargetCatalog/" \\
        --output "data/active/survey-b/TargetCatalog/" \\
        --ss-h5-dir "data/active/survey-b/ss_h5/" \\
        --calibrate-offset

2026 rshomberg@uri.edu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede any import that pulls in pyplot (viz.targets below) -- otherwise matplotlib probes for a GUI backend on import and can throw/thrash on a headless machine

from sidescanpy.cli._config import apply_config_defaults, find_config_path, load_config
from sidescanpy.io import detect_format, read_and_convert
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.io.jsf.targets import read_discover_targets
from sidescanpy.io.sdf.targets import (
    calibrate_clock_offset,
    read_klein_targets,
    resolve_targets_nearest_ping,
    resolve_targets_offset_corrected,
)
from sidescanpy.processing import crop_far_range_noise
from sidescanpy.targets.io import write_targets_csv, write_targets_json
from sidescanpy.viz.targets import export_target_position_scatter_png

logger = logging.getLogger(__name__)


# ============================================================
# survey
# ============================================================

def _find_sonar_files(input_dir: Path, input_format: str) -> list[Path]:
    if input_format == "auto":
        return sorted(list(input_dir.glob("*.jsf")) + list(input_dir.glob("*.sdf")))
    if input_format == "jsf":
        return sorted(input_dir.glob("*.jsf"))
    if input_format == "sdf":
        return sorted(input_dir.glob("*.sdf"))
    raise ValueError(f"Unknown input_format: {input_format}")


def _convert_one_file(input_path: Path, *, output_dir: Path, force: bool) -> dict:
    result = {"input_path": input_path, "status": "", "error": "", "cutoffs": {}}
    out_h5 = output_dir / f"{input_path.stem}.ss.h5"

    if out_h5.exists() and not force:
        result["status"] = "skipped"
        return result

    try:
        fmt = detect_format([input_path])
        survey = read_and_convert([input_path], fmt, survey_name=input_path.stem)
    except Exception as e:
        result["status"] = "failed_convert"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    try:
        survey, cutoffs = crop_far_range_noise(survey)
        result["cutoffs"] = cutoffs
    except Exception as e:
        result["status"] = "failed_process"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_survey(survey, out_h5, overwrite=force)
    except Exception as e:
        result["status"] = "failed_write"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["status"] = "success"
    return result


def run_survey(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory not found or not a directory: {input_dir}")
        return 1

    files = _find_sonar_files(input_dir, args.input_format)
    if not files:
        logger.error(f"No sonar files found in {input_dir} (format={args.input_format})")
        return 1
    logger.info(f"Found {len(files)} sonar file(s) in {input_dir}")

    start_time = time.time()
    results = []
    for i, path in enumerate(files, 1):
        elapsed = time.time() - start_time
        logger.debug(f"[{i}/{len(files)}] {path.name}  (elapsed: {elapsed:.0f}s)")
        result = _convert_one_file(path, output_dir=output_dir, force=args.force)
        results.append(result)
        if result["status"] == "success":
            cropped = {k: v for k, v in result["cutoffs"].items() if v is not None}
            logger.debug(f"    ✓ wrote .ss.h5 (far-range cropped: {cropped})" if cropped else "    ✓ wrote .ss.h5")
        elif result["status"] == "skipped":
            logger.debug("    ⤳ skipped (output exists; use --force to reprocess)")
        else:
            logger.error(f"    ✗ {result['status']}: {result['error']}")

    total_elapsed = time.time() - start_time
    n_success = sum(1 for r in results if r["status"] == "success")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_failed = sum(1 for r in results if r["status"].startswith("failed"))

    logger.info("=" * 60)
    logger.info(f"convert survey complete in {total_elapsed:.0f}s")
    logger.info(f"  Success: {n_success}/{len(files)}")
    logger.info(f"  Skipped: {n_skipped}")
    logger.info(f"  Failed:  {n_failed}")

    if n_failed > 0:
        logger.warning("Failures:")
        for r in results:
            if r["status"].startswith("failed"):
                logger.warning(f"  {r['input_path'].name}: {r['error']}")
        return 2
    return 0


# ============================================================
# targets
# ============================================================

def _detect_catalog_format(source: Path) -> str:
    """Dispatch on the catalog directory's own contents, not the paired
    survey format — see module docstring."""
    has_klein = any(source.glob("Target_*.tgt"))
    has_jsf = any(source.glob("*.target"))
    if has_klein and has_jsf:
        raise ValueError(f"{source} contains both Klein (Target_*.tgt) and JSF (*.target) catalog files -- ambiguous")
    if has_klein:
        return "sdf_klein"
    if has_jsf:
        return "jsf_discover"
    raise ValueError(f"No target catalog files found in {source} (expected *.target or Target_*.tgt)")


def run_targets(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        fmt = _detect_catalog_format(source)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Reading target catalog from {source} (detected format: {fmt})")

    if fmt == "jsf_discover":
        targets = read_discover_targets(source)
        logger.info(f"Read {len(targets)} target(s)")
        n_verified = sum(1 for t in targets if t.position_confidence == "verified")
        n_approx = sum(1 for t in targets if t.position_confidence == "approximate")
        n_unknown = len(targets) - n_verified - n_approx
        logger.info(f"Position confidence: {n_verified} verified, {n_approx} approximate, {n_unknown} unknown")
        scatter_title = None
    else:
        targets = read_klein_targets(source)
        logger.info(f"Read {len(targets)} target(s)")
        if args.ss_h5_dir is not None:
            offset_s = args.time_offset_s
            if args.calibrate_offset:
                logger.info("Calibrating clock offset from .ss.h5 filenames vs. first ping...")
                offset_s = calibrate_clock_offset(args.ss_h5_dir, args.local_utc_offset_hours)

            if offset_s is not None:
                targets = resolve_targets_offset_corrected(
                    targets, args.ss_h5_dir, offset_s, max_time_delta_s=args.max_time_delta_s,
                )
                n_resolved = sum(1 for t in targets if t.latitude is not None)
                logger.info(f"Offset-corrected geometry resolved for {n_resolved}/{len(targets)} target(s)")
            else:
                targets = resolve_targets_nearest_ping(targets, args.ss_h5_dir, max_time_delta_s=args.max_time_delta_s)
                n_resolved = sum(1 for t in targets if t.latitude is not None)
                logger.info(f"Position resolved for {n_resolved}/{len(targets)} target(s) (nearest-ping approximation)")
        logger.info("Every target has position_confidence='unknown' (Klein catalog reader is unvalidated)")
        scatter_title = f"Target positions (n={sum(1 for t in targets if t.latitude is not None)}, nearest-ping approximation)"

    write_targets_json(targets, output / "targets.json")
    logger.info(f"Wrote {output / 'targets.json'}")

    # JSON is the source-of-truth format (round-trips every field
    # losslessly, see targets.io's own module docstring) -- CSV is a
    # secondary, always-written convenience view for manual/spreadsheet
    # review, not a second thing a caller has to opt into.
    write_targets_csv(targets, output / "targets.csv")
    logger.info(f"Wrote {output / 'targets.csv'}")

    wrote_scatter = export_target_position_scatter_png(targets, str(output / "position_scatter.png"), title=scatter_title)
    if not wrote_scatter:
        logger.warning("No targets had a resolvable position; skipping position scatter")

    logger.info("convert targets complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_survey = subparsers.add_parser("survey", help="Convert raw .jsf/.sdf files to .ss.h5")
    p_survey.add_argument("--input-dir", required=True, type=str, help="Directory of .jsf/.sdf files (non-recursive)")
    p_survey.add_argument("--output-dir", required=True, type=str, help="Where .ss.h5 files are written")
    p_survey.add_argument("--input-format", choices=["jsf", "sdf", "auto"], default="auto")
    p_survey.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    p_survey.set_defaults(func=run_survey)

    p_targets = subparsers.add_parser("targets", help="Convert a target catalog (JSF Discover or Klein SonarWiz) to targets.json")
    p_targets.add_argument("--source", required=True, type=Path, help="Directory of target catalog files")
    p_targets.add_argument("--output", required=True, type=Path, help="Output directory for targets.json + position scatter")
    p_targets.add_argument(
        "--ss-h5-dir", type=Path, default=None,
        help="Klein only: directory of .ss.h5 files to resolve position from (nearest ping by timestamp)",
    )
    p_targets.add_argument("--max-time-delta-s", type=float, default=300.0)
    p_targets.add_argument(
        "--time-offset-s", type=float, default=None,
        help="Klein only: constant seconds to add to each target's pick_timestamp before matching, "
             "correcting a known clock-offset bug. Overridden by --calibrate-offset if both are given.",
    )
    p_targets.add_argument(
        "--calibrate-offset", action="store_true",
        help="Klein only: auto-derive --time-offset-s from the .ss.h5 files' own filenames vs. their "
             "first ping. Requires --ss-h5-dir.",
    )
    p_targets.add_argument(
        "--local-utc-offset-hours", type=float, default=4.0,
        help="Klein only: local-time-to-UTC offset used only by --calibrate-offset (default 4.0 = EDT)",
    )
    p_targets.set_defaults(func=run_targets)

    for p in (p_survey, p_targets):
        p.add_argument("--config", type=Path, default=None, help="YAML or JSON file seeding this subcommand's defaults -- explicit flags still override")
        # default=SUPPRESS (not False): argparse copies a subparser's
        # *entire* parsed namespace onto the shared one, including
        # unset-flag defaults -- a plain default=False here would
        # silently overwrite a real -v already set by the top-level
        # parser (e.g. `sidescan-convert -v survey ...`) whenever -v
        # isn't *also* repeated after the subcommand. SUPPRESS leaves
        # the attribute absent unless this copy is actually passed, so
        # it only ever adds capability (accepting -v after the
        # subcommand too), never removes it.
        p.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Also accepted after the subcommand, e.g. 'survey ... -v'")

    parser.add_argument("-v", "--verbose", action="store_true")

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        config = load_config(config_path)
        for p in (p_survey, p_targets):
            apply_config_defaults(p, config)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
