"""
``sidescan-correct``: apply far-range crop / TVG / geometric / georef
correction to a directory of ``.ss.h5`` files, in place.

Replaces the old per-project ``02_crop_farrange.py``, ``04_radiometric.py``,
``06_geometric.py``, ``08_georef.py`` pipeline scripts — confirmed
byte-identical between `project-b` and `project-a`'s own copies
except docstring path/stage-number text, each one a thin wrapper adding
its own single correction step to the same in-place-edit loop. Each step
is independently toggleable and independently idempotent (already-applied
steps are skipped, not re-applied) — requesting several steps in one
invocation reads and writes each file once, not once per stage.

``backscatter_raw`` stays untouched (immutable) through every step;
``backscatter_intensity``/the geometry variables are what change. Writes
are atomic (temp file + rename) so a write that fails partway (e.g. disk
full) never corrupts or removes the original.

Usage:
    sidescan-correct --ss-h5-dir data/active/unknown/ss_h5/ \\
        --crop-farrange --tvg --geometric --georef

    # Hardware already compensated spreading loss -- absorption only:
    sidescan-correct --ss-h5-dir data/active/unknown/ss_h5/ \\
        --tvg --no-spreading

    # Run one stage standalone, same as the old numbered scripts:
    sidescan-correct --ss-h5-dir data/active/unknown/ss_h5/ --georef

    # Add the bundled empirical range correction on top of an
    # already-tvg+geometric-corrected directory -- fits/removes a
    # residual, altitude-invariant range-dependent shape plain
    # physics-based TVG doesn't capture (see processing/range_correction.py's
    # module docstring). Only hz_1495000/hz_547500 are calibrated by the
    # bundled default table.
    sidescan-correct --ss-h5-dir data/active/unknown/ss_h5/ --range-correction

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from sidescanpy.cli._common import process_files_in_place
from sidescanpy.cli._config import apply_config_defaults, find_config_path
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing import crop_far_range_noise, run_correction_steps
from sidescanpy.processing.radiometric import TVGParams
from sidescanpy.processing.range_correction import STEP_NAME as _RANGE_CORRECTION_STEP_NAME

logger = logging.getLogger(__name__)

# Each multiplier is that step's own conservative "how much bigger could
# this make the file, relative to its pre-step size" estimate, carried
# over unchanged from the old standalone scripts. Summing the requested
# steps' (multiplier - 1.0) terms against the file's *original* size is a
# deliberate overestimate when several steps are combined in one pass
# (each step's true growth is relative to a smaller preceding size, not
# the original) -- the safe direction for a disk-space guard.
_DISK_MULTIPLIERS = {
    "crop_farrange": 1.5,
    "tvg": 3.0,
    "geometric": 4.0,
    "range_correction": 1.0,  # shifts backscatter_intensity in place, adds no new arrays
    "georef": 2.0,
}
_CORRECTION_STEPS = ("tvg", "geometric", "range_correction", "georef")


def _crop_already_done(survey: SideScanSurvey) -> bool:
    for channel in survey.freq_channels.values():
        history = json.loads(channel.dataset.attrs.get("processing_history", "[]"))
        if not any(h.get("step") == "crop_far_range_noise" for h in history):
            return False
    return True


def _tvg_already_done(survey: SideScanSurvey) -> bool:
    return all(
        channel.dataset.attrs.get("processing_level") != "raw"
        for channel in survey.freq_channels.values()
    )


def _geometric_already_done(survey: SideScanSurvey) -> bool:
    return all(
        "ground_range_m" in channel.dataset.data_vars
        for channel in survey.freq_channels.values()
    )


def _georef_already_done(survey: SideScanSurvey) -> bool:
    return all(
        "latitude_per_sample" in channel.dataset.data_vars
        for channel in survey.freq_channels.values()
    )


def _range_correction_already_done(survey: SideScanSurvey) -> bool:
    for channel in survey.freq_channels.values():
        history = json.loads(channel.dataset.attrs.get("processing_history", "[]"))
        if not any(h.get("step") == _RANGE_CORRECTION_STEP_NAME for h in history):
            return False
    return True


_ALREADY_DONE = {
    "crop_farrange": _crop_already_done,
    "tvg": _tvg_already_done,
    "geometric": _geometric_already_done,
    "range_correction": _range_correction_already_done,
    "georef": _georef_already_done,
}


def _make_already_done(requested: set[str]):
    def already_done(survey: SideScanSurvey) -> bool:
        return all(_ALREADY_DONE[step](survey) for step in requested)

    return already_done


def _make_step_fn(requested: set[str], tvg_params: TVGParams | None):
    def step_fn(survey: SideScanSurvey) -> SideScanSurvey:
        if "crop_farrange" in requested and not _crop_already_done(survey):
            survey, _cutoffs = crop_far_range_noise(survey)

        pending = {
            s for s in _CORRECTION_STEPS
            if s in requested and not _ALREADY_DONE[s](survey)
        }
        if pending:
            step_params = {"tvg": {"params": tvg_params}} if tvg_params is not None else None
            survey = run_correction_steps(survey, pending, step_params=step_params)
        return survey

    return step_fn


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
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON file seeding this command's defaults (e.g. {\"spreading_loss_db\": 0.0}) -- explicit flags still override",
    )

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        apply_config_defaults(parser, json.loads(config_path.read_text()))

    args = parser.parse_args(argv)

    requested = {
        step for step, flag in (
            ("crop_farrange", args.crop_farrange),
            ("tvg", args.tvg),
            ("geometric", args.geometric),
            ("range_correction", args.range_correction),
            ("georef", args.georef),
        ) if flag
    }
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tvg_kwargs = {}
    if args.no_spreading:
        tvg_kwargs["spreading_loss_db"] = 0.0
    elif args.spreading_loss_db is not None:
        tvg_kwargs["spreading_loss_db"] = args.spreading_loss_db
    if args.no_absorption:
        tvg_kwargs["absorption_db_per_m"] = 0.0
    elif args.absorption_db_per_m is not None:
        tvg_kwargs["absorption_db_per_m"] = args.absorption_db_per_m
    tvg_params = TVGParams(**tvg_kwargs) if tvg_kwargs else None
    if tvg_params is not None:
        logger.info(
            f"TVG override: spreading_loss_db={tvg_params.spreading_loss_db}, "
            f"absorption_db_per_m={tvg_params.absorption_db_per_m}"
        )

    disk_multiplier = 1.0 + sum(_DISK_MULTIPLIERS[s] - 1.0 for s in requested)
    files = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1

    stage_label = f"correct ({'+'.join(sorted(requested))})"
    return process_files_in_place(
        files,
        step_fn=_make_step_fn(requested, tvg_params),
        already_done_fn=_make_already_done(requested),
        disk_multiplier=disk_multiplier,
        stage_label=stage_label,
    )


if __name__ == "__main__":
    sys.exit(main())
