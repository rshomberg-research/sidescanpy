"""
``sidescan-correct``: apply far-range crop / TVG / geometric / range
correction / georef to a directory of ``.ss.h5`` files, in place.

A thin loop over :func:`sidescanpy.processing.correct_file`: each file is
read once, every requested step not already recorded in its
``processing_history`` is applied (in canonical order: crop_farrange,
interpolate_aux, tvg, geometric, range_correction, georef), and the file is
rewritten atomically. Aux interpolation runs whenever tvg/geometric/
range_correction/georef is requested. Already-applied steps are skipped,
so re-running is a no-op. No raw copy is kept (a legacy ``backscatter_raw``
is dropped on rewrite); recorded radiometric steps can be inverted with
``sidescanpy.processing.invert``.

Usage:
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ \\
        --crop-farrange --tvg --geometric --georef

    # Hardware already compensated spreading loss -- absorption only:
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ --tvg --no-spreading

    # Bound memory on large files:
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ --tvg --geometric --georef \\
        --ping-chunk-size 1000

    # Absorption from the Francois-Garrison formula for given water conditions:
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ --tvg \\
        --temperature-c 14 --salinity-psu 31 --depth-m 20

    # Empirical range correction on top of tvg+geometric, from a table you
    # fitted earlier (frequencies without a curve are skipped with a warning):
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ --range-correction \\
        --range-correction-table range_table.json

2026 rshomberg@uri.edu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from sidescanpy.cli._common import add_common_args, setup_logging_from_args, ss_h5_files
from sidescanpy.cli._config import CLIParser, parse_args
from sidescanpy.processing import (
    STEP_CROP_FARRANGE,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_INTERPOLATE_AUX,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    correct_file,
    correction_steps,
)

logger = logging.getLogger(__name__)

_FLAG_STEPS = (
    ("crop_farrange", STEP_CROP_FARRANGE),
    ("tvg", STEP_TVG),
    ("geometric", STEP_GEOMETRIC),
    ("range_correction", STEP_RANGE_CORRECTION),
    ("georef", STEP_GEOREF),
)
_WATER = ("temperature_c", "salinity_psu", "depth_m", "ph")


def _steps_from_args(args: argparse.Namespace) -> list[str]:
    return correction_steps(step for flag, step in _FLAG_STEPS if getattr(args, flag))


def _params_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace, steps: list[str]) -> dict | None:
    """``correct_file`` params from the flags; ``parser.error`` on an invalid combination."""
    if not steps:
        parser.error("at least one of --crop-farrange/--tvg/--geometric/--range-correction/--georef is required")
    if args.no_spreading and args.spreading_loss_db is not None:
        parser.error("--no-spreading and --spreading-loss-db are mutually exclusive")
    if args.no_absorption and args.absorption_db_per_m is not None:
        parser.error("--no-absorption and --absorption-db-per-m are mutually exclusive")
    table = args.range_correction_table
    if args.range_correction and table is None:
        parser.error("--range-correction requires --range-correction-table PATH (the library has no built-in table)")
    if table is not None and not args.range_correction:
        parser.error("--range-correction-table requires --range-correction")
    if table is not None and not table.is_file():
        parser.error(f"--range-correction-table: no such file: {table}")

    tvg = {}
    if args.no_spreading or args.spreading_loss_db is not None:
        tvg["spreading_loss_db"] = 0.0 if args.no_spreading else args.spreading_loss_db
    if args.no_absorption or args.absorption_db_per_m is not None:
        tvg["absorption_db_per_m"] = 0.0 if args.no_absorption else args.absorption_db_per_m
    tvg.update({k: getattr(args, k) for k in _WATER if getattr(args, k) is not None})
    if tvg and STEP_TVG not in steps:
        parser.error("TVG term overrides require --tvg")
    if args.ping_chunk_size is not None and args.ping_chunk_size <= 0:
        parser.error("--ping-chunk-size must be a positive integer")

    params = {STEP_TVG: tvg} if tvg else {}
    if args.range_correction:
        params[STEP_RANGE_CORRECTION] = {"table": str(table)}
    return params or None


def build_parser() -> CLIParser:
    parser = CLIParser(prog="sidescan-correct", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ss-h5-dir", required=True, type=Path, help="Directory of .ss.h5 files, edited in place")
    parser.add_argument("--crop-farrange", action="store_true", help="Detect and crop far-range noise")
    parser.add_argument("--tvg", action="store_true", help="Apply radiometric (TVG) correction")
    parser.add_argument("--geometric", action="store_true", help="Apply geometric range correction")
    parser.add_argument(
        "--range-correction", action="store_true",
        help=(
            "Apply an empirical range-correction curve (residual TVG-vs-range "
            "shape TVG's spreading/absorption terms don't capture -- see "
            "processing/range_correction.py). Requires --range-correction-table and "
            "--geometric to have run (this invocation or a prior one). A channel at a "
            "frequency the table has no curve for is skipped with a warning, not guessed."
        ),
    )
    parser.add_argument(
        "--range-correction-table", type=Path, default=None, metavar="PATH",
        help="JSON range-correction table (RangeCorrectionTable.save). Required by --range-correction.",
    )
    parser.add_argument("--georef", action="store_true", help="Apply georeferencing")
    parser.add_argument(
        "--spreading-loss-db", type=float, default=None,
        help="Override TVG spreading-loss coefficient (dB/decade). Default: library default (30.0, the usual sidescan seafloor spreading term; 40.0 for point-target/volume scattering). Requires --tvg.",
    )
    parser.add_argument(
        "--absorption-db-per-m", type=float, default=None,
        help="Override TVG absorption coefficient (dB/m). Default: Francois-Garrison (1982, JASA 72:896, 72:1879) from frequency and water conditions. Requires --tvg.",
    )
    fg = "for the Francois-Garrison (1982, JASA 72:896 and 72:1879) absorption"
    parser.add_argument("--temperature-c", type=float, default=None, help=f"Water temperature (C) {fg}. Default 10. Requires --tvg; ignored with --absorption-db-per-m.")
    parser.add_argument("--salinity-psu", type=float, default=None, help=f"Salinity (psu) {fg}. Default 35. Requires --tvg.")
    parser.add_argument("--depth-m", type=float, default=None, help=f"Water depth (m) {fg}. Default 10. Requires --tvg.")
    parser.add_argument("--ph", type=float, default=None, help=f"Seawater pH {fg}. Default 8. Requires --tvg.")
    parser.add_argument(
        "--no-spreading", action="store_true",
        help="Disable the spreading-loss term entirely. Use when hardware/acquisition already applied it. Requires --tvg.",
    )
    parser.add_argument(
        "--no-absorption", action="store_true",
        help="Disable the absorption-loss term entirely. Use when hardware/acquisition already applied it. Requires --tvg.",
    )
    parser.add_argument(
        "--ping-chunk-size", type=int, default=None,
        help=(
            "Process each file in chunks of this many pings instead of all at once -- "
            "bounds peak memory to roughly one chunk's worth of derived arrays regardless "
            "of file size. Results are identical to the unchunked path. Default: unset, "
            "process the whole file at once."
        ),
    )
    add_common_args(parser, subcommand=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parse_args(parser, sys.argv[1:] if argv is None else argv)
    steps = _steps_from_args(args)
    params = _params_from_args(parser, args, steps)
    setup_logging_from_args(args)
    if params and params.get(STEP_TVG):
        logger.info(f"TVG override: {params[STEP_TVG]}")

    files = ss_h5_files(args.ss_h5_dir)
    if not files:
        return 1
    label = "+".join(s for s in steps if s != STEP_INTERPOLATE_AUX)
    failures = []
    start_all = time.perf_counter()
    for i, path in enumerate(files, 1):
        start = time.perf_counter()
        before = path.stat().st_mtime_ns
        try:
            correct_file(path, steps, params, chunk_pings=args.ping_chunk_size, progress=logger.debug)
        except Exception as e:  # one bad file must not stop the batch
            failures.append(path.name)
            logger.error(f"[{i}/{len(files)}] {path.name}: FAILED {type(e).__name__}: {e}")
            continue
        state = "corrected" if path.stat().st_mtime_ns != before else "already corrected, unchanged"
        logger.info(f"[{i}/{len(files)}] {path.name}: {state} ({label}) in {time.perf_counter() - start:.1f}s")

    logger.info(f"correct ({label}): {len(files) - len(failures)}/{len(files)} ok in {time.perf_counter() - start_all:.0f}s")
    if failures:
        logger.warning(f"Failed: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
