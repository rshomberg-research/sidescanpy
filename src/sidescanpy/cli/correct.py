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

    # Empirical range correction on top of tvg+geometric (bundled table is
    # calibrated for two frequencies only; others are skipped with a warning):
    sidescan-correct --ss-h5-dir data/<survey>/ss_h5/ --range-correction

2026 rshomberg@uri.edu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from sidescanpy.cli._common import add_logging_args, setup_logging_from_args
from sidescanpy.cli._config import (
    apply_config_defaults,
    find_config_path,
    has_strict_flag,
    load_config,
)
from sidescanpy.processing import (
    STEP_CROP_FARRANGE,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_INTERPOLATE_AUX,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    correct_file,
)

logger = logging.getLogger(__name__)

_FLAG_STEPS = (
    ("crop_farrange", STEP_CROP_FARRANGE),
    ("tvg", STEP_TVG),
    ("geometric", STEP_GEOMETRIC),
    ("range_correction", STEP_RANGE_CORRECTION),
    ("georef", STEP_GEOREF),
)
_NEEDS_AUX = {STEP_TVG, STEP_GEOMETRIC, STEP_RANGE_CORRECTION, STEP_GEOREF}
_CANONICAL_ORDER = (
    STEP_CROP_FARRANGE, STEP_INTERPOLATE_AUX, STEP_TVG, STEP_GEOMETRIC, STEP_RANGE_CORRECTION, STEP_GEOREF,
)


def _steps_from_args(args: argparse.Namespace) -> list[str]:
    requested = {step for flag, step in _FLAG_STEPS if getattr(args, flag)}
    if requested & _NEEDS_AUX:
        requested.add(STEP_INTERPOLATE_AUX)
    return [s for s in _CANONICAL_ORDER if s in requested]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ss-h5-dir", required=True, type=Path, help="Directory of .ss.h5 files, edited in place")
    parser.add_argument("--crop-farrange", action="store_true", help="Detect and crop far-range noise")
    parser.add_argument("--tvg", action="store_true", help="Apply radiometric (TVG) correction")
    parser.add_argument("--geometric", action="store_true", help="Apply geometric range correction")
    parser.add_argument(
        "--range-correction", action="store_true",
        help=(
            "Apply the bundled empirical range-correction curve (residual TVG-vs-range "
            "shape TVG's spreading/absorption terms don't capture -- see "
            "processing/range_correction.py). Requires --geometric to have run (this "
            "invocation or a prior one). Only calibrated for hz_1495000/hz_547500 -- a "
            "channel at another frequency is skipped with a warning, not guessed."
        ),
    )
    parser.add_argument("--georef", action="store_true", help="Apply georeferencing")
    parser.add_argument(
        "--spreading-loss-db", type=float, default=None,
        help="Override TVG spreading-loss coefficient (dB/decade). Default: library default (20.0). Requires --tvg.",
    )
    parser.add_argument(
        "--absorption-db-per-m", type=float, default=None,
        help="Override TVG absorption coefficient (dB/m). Default: frequency-derived. Requires --tvg.",
    )
    parser.add_argument(
        "--no-spreading", action="store_true",
        help="Disable the spreading-loss term entirely. Use when hardware/acquisition already applied it. Requires --tvg.",
    )
    parser.add_argument(
        "--no-absorption", action="store_true",
        help="Disable the absorption-loss term entirely. Use when hardware/acquisition already applied it. Requires --tvg.",
    )
    add_logging_args(parser)
    parser.add_argument("--strict-config", action="store_true", help="Treat config-file keys this command does not recognize as an error instead of a warning")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML or JSON file seeding this command's defaults (e.g. spreading_loss_db: 0.0) -- explicit flags still override",
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

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        apply_config_defaults(parser, load_config(config_path), strict=has_strict_flag(argv), subcommand=parser.prog)

    args = parser.parse_args(argv)

    requested = set(_steps_from_args(args))
    if not requested:
        parser.error("at least one of --crop-farrange/--tvg/--geometric/--range-correction/--georef is required")

    if args.no_spreading and args.spreading_loss_db is not None:
        parser.error("--no-spreading and --spreading-loss-db are mutually exclusive")
    if args.no_absorption and args.absorption_db_per_m is not None:
        parser.error("--no-absorption and --absorption-db-per-m are mutually exclusive")
    tvg_override_requested = (
        args.no_spreading or args.spreading_loss_db is not None
        or args.no_absorption or args.absorption_db_per_m is not None
    )
    if tvg_override_requested and "tvg" not in requested:
        parser.error("TVG term overrides require --tvg")

    if args.ping_chunk_size is not None and args.ping_chunk_size <= 0:
        parser.error("--ping-chunk-size must be a positive integer")

    setup_logging_from_args(args)

    tvg_params = {}
    if args.no_spreading:
        tvg_params["spreading_loss_db"] = 0.0
    elif args.spreading_loss_db is not None:
        tvg_params["spreading_loss_db"] = args.spreading_loss_db
    if args.no_absorption:
        tvg_params["absorption_db_per_m"] = 0.0
    elif args.absorption_db_per_m is not None:
        tvg_params["absorption_db_per_m"] = args.absorption_db_per_m
    if tvg_params:
        logger.info(f"TVG override: {tvg_params}")
    params = {STEP_TVG: tvg_params} if tvg_params else None
    steps = _steps_from_args(args)

    files = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
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

    logger.info(
        f"correct ({label}): {len(files) - len(failures)}/{len(files)} ok in "
        f"{time.perf_counter() - start_all:.0f}s"
    )
    if failures:
        logger.warning(f"Failed: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
