"""
Smoke-run the scripts in ``examples/`` that do not need downloaded data
(the project template), on small synthetic surveys. The gallery and Theory
pages run on public recordings during the docs build; ``test_gallery.py``
checks them offline.

Run with ``PYTHONPATH="" uv run --all-extras pytest tests/examples -q``.

2026 rshomberg@uri.edu
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from sidescanpy.core import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run_example(script: str, *args) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *map(str, args)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    return proc


def _load_example(script: str):
    spec = importlib.util.spec_from_file_location(Path(script).stem, EXAMPLES / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _two_pass_survey(ss_dir: Path, objects) -> Path:
    """The same corrected synthetic survey written as two passes two hours
    apart, each with bright objects at the given (ping, channel, sample);
    returns a detections CSV naming them in both passes."""
    import pandas as pd
    import xarray as xr

    from sidescanpy.processing import correct_survey

    n_pings, n_samples = 400, 400
    rng = np.random.default_rng(0)
    backscatter = rng.uniform(5.0, 10.0, (n_pings, 2, n_samples)).astype(np.float32)
    for p, ch, s in objects:
        backscatter[p - 3:p + 4, ch, s - 4:s + 5] = 60.0
        backscatter[p - 3:p + 4, ch, s + 5:s + 25] = 0.5  # shadow
    per_ping = {name: (["ping"], np.full(n_pings, value, dtype=np.float32)) for name, value in (
        ("heading_deg", 45.0), ("depth_m", 5.0), ("roll_deg", 0.0), ("pitch_deg", 0.0), ("heave_m", 0.0),
        ("altitude_m", 0.5), ("sound_vel_ms", 1500.0), ("fathometer_reading_m", 0.5),
    )}
    rows = []
    for k, name in enumerate(("a", "b")):
        ds = xr.Dataset(
            {"backscatter_intensity": (["ping", "channel", "sample"], backscatter),
             "latitude": (["ping"], np.linspace(41.5, 41.5004, n_pings)),
             "longitude": (["ping"], np.linspace(-71.5, -71.4995, n_pings)), **per_ping},
            coords={"ping_number": (["ping"], np.arange(1, n_pings + 1, dtype=np.int32)),
                    "time": (["ping"], pd.date_range(f"2026-01-01 0{2 * k}:00", periods=n_pings, freq="100ms").values),
                    "channel_id": (["channel"], np.array(["port", "starboard"]))},
            attrs={"freq_hz_start": 900_000, "freq_hz_end": 900_000, "pulse_length_us": 0.5,
                   "sample_rate_hz": 40_000.0, "processing_history": "[]"},
        )
        survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
        survey.add_frequency("hz_900000", ds)
        corrected = correct_survey(survey, steps=["geometric", "georef"])
        write_survey(corrected, ss_dir / f"{name}.ss.h5")
        c = corrected.get_frequency("hz_900000").dataset
        for p, ch, s in objects:
            rows.append({"lat": float(c["latitude_per_sample"][p, ch, s]), "lon": float(c["longitude_per_sample"][p, ch, s]),
                         "file": f"{name}.ss.h5", "ping": p, "channel": ch, "sample": s,
                         "time": str(c["time"].values[p]), "conf": 0.9})
    csv = ss_dir.parent / "detections.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_look_consistency_pairs_the_same_object_across_passes(tmp_path):
    ss_dir = tmp_path / "ss_h5"
    ss_dir.mkdir()
    detections = _two_pass_survey(ss_dir, [(120, 1, 200), (300, 0, 150)])
    out = tmp_path / "out"
    _run_example("look_consistency.py", ss_dir, "--detections", detections, "--frequency", "hz_900000",
                 "--out", out, "--chip-m", "2", "--chip-px", "32", "--score-px", "16", "--max-shift-m", "0.2",
                 "--min-ground-m", "0", "--link-m", "1", "--isolation-m", "0.5", "--no-extents",
                 "--rotation-step", "90", "--sheets", "2")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["objects"] == 2 and summary["looks"] == 4
    # identical passes: the same object correlates perfectly in every geometry and frame
    assert summary["scores"]["slant_track"]["same_median"] > 0.95
    assert summary["scores"]["ground_north_up"]["same_median"] > 0.95
    for f in ("similarity_summary.png", "gallery.png", "looks.csv", "pairs.csv"):
        assert (out / f).exists()
    assert len(list((out / "objects").glob("object_*.png"))) == 2
