"""
Smoke-run the scripts in ``examples/`` that do not need downloaded data
(the project template), on small synthetic surveys. The gallery and Theory
pages run on public recordings during the docs build; ``test_gallery.py``
checks them offline.

Run with ``PYTHONPATH="" uv run --all-extras pytest tests/examples -q``.

2026 rshomberg@uri.edu
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sidescanpy.core import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


# ============================================================
# examples/project_template: the five step scripts end to end
# ============================================================

def _template_inputs(root: Path) -> None:
    """An uncorrected survey (as 01_convert would write it) with three bright
    targets, plus their targets.json, under ``root/output``."""
    import pandas as pd
    import xarray as xr

    from sidescanpy.core.target import Target
    from sidescanpy.targets.io import write_targets_json

    n_pings, n_samples = 400, 100
    rng = np.random.default_rng(0)
    lat = np.linspace(41.5, 41.5004, n_pings)
    lon = np.linspace(-71.5, -71.5004, n_pings)
    backscatter = rng.uniform(5.0, 10.0, (n_pings, 2, n_samples)).astype(np.float32)
    picks = [(80, 1, 60), (200, 0, 40), (320, 1, 30)]
    for p, ch, s in picks:
        backscatter[p - 1:p + 2, ch, s - 2:s + 3] = 60.0
    per_ping = {name: (["ping"], np.full(n_pings, value, dtype=np.float32)) for name, value in (
        ("heading_deg", 45.0), ("depth_m", 5.0), ("roll_deg", 0.0), ("pitch_deg", 0.0), ("heave_m", 0.0),
        ("altitude_m", 0.1), ("sound_vel_ms", 1500.0), ("fathometer_reading_m", 0.1),
    )}
    ds = xr.Dataset(
        {"backscatter_intensity": (["ping", "channel", "sample"], backscatter),
         "latitude": (["ping"], lat), "longitude": (["ping"], lon), **per_ping},
        coords={"ping_number": (["ping"], np.arange(1, n_pings + 1, dtype=np.int32)),
                "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="100ms").values),
                "channel_id": (["channel"], np.array(["port", "starboard"]))},
        attrs={"freq_hz_start": 900_000, "freq_hz_end": 900_000, "pulse_length_us": 0.5,
               "sample_rate_hz": 40_000.0, "processing_history": "[]"},
    )
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_900000", ds)
    out = root / "output"
    (out / "ss_h5").mkdir(parents=True)
    write_survey(survey, out / "ss_h5" / "synthetic.ss.h5")
    (out / "targets").mkdir()
    write_targets_json([
        Target(target_id=f"{i:04d}", source_format="synthetic", catalog_source="test", latitude=float(lat[p]),
               longitude=float(lon[p]), channel=ch, sample=s, position_confidence="verified")
        for i, (p, ch, s) in enumerate(picks)
    ], out / "targets" / "targets.json")


def test_project_template_steps_run_end_to_end(tmp_path):
    import yaml

    template = EXAMPLES / "project_template"
    _template_inputs(tmp_path)
    cfg = yaml.safe_load((template / "config.yaml").read_text())
    cfg["paths"] = {"output_root": "output"}  # no raw files or catalog: start from converted files
    cfg["frequencies"] = ["hz_900000"]
    cfg["analyze"]["pings"] = [0, 400]
    cfg["analyze"]["bin_width_m"] = 0.1
    cfg["correct"]["steps"] = ["tvg", "geometric", "georef"]
    cfg["correct"]["chunk_pings"] = 150
    cfg["export"]["waterfall"].update(window_pings=200, overlap_pings=50, overview_rows=20, dpi=40)
    cfg["export"]["grid"].update(segment_pings=200, resolution_m=0.2, dpi=40)
    cfg["export"]["track"]["dpi"] = 40
    cfg["dataset"].update(size_px=32, swath_size_px=32, jitter_px=4, box_size=8.0, with_context=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))

    for step in ("01_convert", "02_analyze", "03_correct", "04_export", "05_dataset"):
        proc = subprocess.run([sys.executable, str(template / f"{step}.py"), str(config)],
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, f"{step}:\n{proc.stderr[-3000:]}"
        assert "Traceback" not in proc.stderr
        report = json.loads((tmp_path / "output" / "reports" / f"{step}.json").read_text())
        assert report["step"] == step

    out = tmp_path / "output"
    corrected = json.loads((out / "reports" / "03_correct.json").read_text())["results"]
    assert set(corrected["applied"]["hz_900000"]) >= {"tvg", "geometric", "georef"}
    assert list((out / "exports" / "waterfall" / "hz_900000").glob("segment_*.png"))
    assert (out / "exports" / "track" / "hz_900000" / "track.png").exists()
    dataset = json.loads((out / "reports" / "05_dataset.json").read_text())["results"]
    assert dataset["n_matched"] == 3 and dataset["n_images"] > 0
