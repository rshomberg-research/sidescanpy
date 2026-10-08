"""
Build a YOLO/COCO training dataset (targets plus background images) with
:func:`sidescanpy.export.ml.build_target_dataset`.

With no inputs, a small synthetic survey with three bright point targets
is generated first, so the example runs anywhere.

Usage:
    python examples/build_yolo_dataset.py --output-dir out/
    python examples/build_yolo_dataset.py --output-dir out/ \\
        --ss-h5-dir corrected/ --targets-json targets.json --spec dataset.yaml

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.core import SideScanSurvey
from sidescanpy.core.target import Target
from sidescanpy.export.ml import DatasetSpec, build_target_dataset
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct_survey
from sidescanpy.targets.io import read_targets_json

logger = logging.getLogger("build_yolo_dataset")

N_PINGS, N_SAMPLES = 400, 100
TARGETS = [(80, 1, 60), (200, 0, 40), (320, 1, 30)]  # (ping, channel, sample)


def synthetic_survey(out_dir: Path) -> tuple[Path, list[Target]]:
    """A corrected one-frequency survey (~1.9 m swath per side, 0.12 m/ping)
    with a bright 3 x 5 patch at each target, and its catalog."""
    rng = np.random.default_rng(0)
    lat = np.linspace(41.5, 41.5004, N_PINGS)
    lon = np.linspace(-71.5, -71.5004, N_PINGS)
    backscatter = rng.uniform(5.0, 10.0, (N_PINGS, 2, N_SAMPLES)).astype(np.float32)
    for p, ch, s in TARGETS:
        backscatter[p - 1:p + 2, ch, s - 2:s + 3] = 60.0
    per_ping = {name: (["ping"], np.full(N_PINGS, value, dtype=np.float32)) for name, value in (
        ("heading_deg", 45.0), ("depth_m", 5.0), ("roll_deg", 0.0), ("pitch_deg", 0.0), ("heave_m", 0.0),
        ("altitude_m", 0.1), ("sound_vel_ms", 1500.0), ("fathometer_reading_m", 0.1),
    )}
    ds = xr.Dataset(
        {"backscatter_intensity": (["ping", "channel", "sample"], backscatter),
         "latitude": (["ping"], lat), "longitude": (["ping"], lon), **per_ping},
        coords={
            "ping_number": (["ping"], np.arange(1, N_PINGS + 1, dtype=np.int32)),
            "time": (["ping"], pd.date_range("2026-01-01", periods=N_PINGS, freq="100ms").values),
            "channel_id": (["channel"], np.array(["port", "starboard"])),
        },
        attrs={"freq_hz_start": 900_000, "freq_hz_end": 900_000, "pulse_length_us": 0.5,
               "sample_rate_hz": 40_000.0, "processing_history": "[]"},
    )
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_900000", ds)
    ss_dir = out_dir / "ss_h5"
    ss_dir.mkdir(parents=True, exist_ok=True)
    write_survey(correct_survey(survey), ss_dir / "synthetic.ss.h5", overwrite=True)
    catalog = [
        Target(target_id=f"{i:04d}", source_format="synthetic", catalog_source="example", latitude=float(lat[p]),
               longitude=float(lon[p]), channel=ch, sample=s, position_confidence="verified")
        for i, (p, ch, s) in enumerate(TARGETS)
    ]
    return ss_dir, catalog


def print_layout(root: Path, max_files: int = 4) -> None:
    """Directory tree with at most ``max_files`` files listed per directory."""
    for d in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        shown = ", ".join(files[:max_files]) + (f", ... (+{len(files) - max_files})" if len(files) > max_files else "")
        print(f"{d.relative_to(root.parent)}/  {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ss-h5-dir", type=Path, help="Corrected .ss.h5 files (default: build a synthetic survey)")
    parser.add_argument("--targets-json", type=Path, help="Target catalog (required with --ss-h5-dir)")
    parser.add_argument("--spec", type=Path, help="DatasetSpec YAML (default: the settings below)")
    args = parser.parse_args()
    configure_logging("WARNING")

    if args.ss_h5_dir is None:
        ss_h5_dir, catalog = synthetic_survey(args.output_dir / "input")
    else:
        ss_h5_dir, catalog = args.ss_h5_dir, read_targets_json(args.targets_json)

    # One spec drives everything; it round-trips to YAML for a project config.
    spec = DatasetSpec.from_yaml(args.spec) if args.spec else DatasetSpec(
        styles=["chip", "dual-swath"], render_level="geometric", size_px=32, swath_size_px=32,
        n_augment=2, jitter_px=4, random_flip=True, n_background=2, background_max_nan_frac=0.6,
        val_fraction=0.3, layout="ultralytics", seed=1,
    )
    dataset_dir = args.output_dir / "dataset"
    result = build_target_dataset(ss_h5_dir, catalog, spec, dataset_dir)

    for key, style in result.styles.items():
        print(f"{key}: {style.n_target_images} target + {style.n_background_images} background images, "
              f"{style.n_annotations} annotations, splits {style.split_counts}")
    for skipped in result.skipped:
        print(f"skipped {skipped.target_id} ({skipped.style or 'all styles'}): {skipped.reason}")
    print_layout(dataset_dir)
    print(json.dumps({"manifest": str(result.manifest_path)}))


if __name__ == "__main__":
    main()
