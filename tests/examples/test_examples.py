"""
Smoke-run the scripts in ``examples/`` on a small synthetic survey.

Run with ``.venv/bin/python -m pytest tests/examples -q -o addopts=""``.

2026 rshomberg@uri.edu
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sidescanpy.core import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
from tests.unit.synthetic import make_raw_dataset

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _synthetic_survey(path: Path) -> Path:
    n_pings, n_samples = 30, 400
    rng = np.random.default_rng(0)
    decay = 100.0 * np.exp(-np.arange(n_samples) / 120.0) + 1.0
    data = decay[None, None, :] * (1.0 + 0.05 * rng.standard_normal((n_pings, 2, n_samples)))
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    survey.add_frequency("hz_200000", make_raw_dataset(backscatter=data.astype(np.float32)))
    write_survey(survey, path)
    return path


def test_investigate_dataset_writes_metrics_and_plots(tmp_path):
    source = _synthetic_survey(tmp_path / "synthetic.ss.h5")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "investigate_dataset.py"), str(source), "--output-dir", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["freq_key"] == "hz_200000"
    for key in ("values", "outlier_comparison", "nadir", "usable_range", "beam_geometry", "range_decay", "units"):
        assert isinstance(metrics[key], dict), key
    pngs = {p.name for p in out.glob("*.png")}
    assert {"value_histograms.png", "nadir_zoom.png", "usable_range.png", "range_decay_fit.png"} <= pngs
    assert len(pngs) == 11
