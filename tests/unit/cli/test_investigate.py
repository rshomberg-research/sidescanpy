"""
Tests for ``sidescan-analyze investigate`` (``cli/analyze.run_investigate``
and its run-everything flow in ``cli/_investigate``).

2026 rshomberg@uri.edu
"""

import json
import re

import numpy as np
import pytest

from sidescanpy.cli import _investigate, analyze
from sidescanpy.core import SideScanSurvey
from sidescanpy.io.hdf5.writer import write_survey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, absorption_for_frequency, tvg_gain_db
from tests.unit.synthetic import make_raw_dataset

_FREQ_HZ = 200_000  # low enough that full-physics loss stays above the dB floor at the test range

_EXPECTED_PNGS = {
    "value_histograms.png", "sample_ping.png", "outlier_comparison.png", "cross_track_profile.png",
    "nadir_zoom.png", "nadir_waterfall.png", "beam_geometry.png", "usable_range.png",
    "range_decay_fit.png", "range_decay_residuals.png", "units_distribution.png",
}
_EXPECTED_RESULTS = {"values", "outlier_comparison", "nadir", "beam_geometry", "usable_range", "range_decay", "units"}


def _two_channel_dataset(n_pings=40, n_samples=300):
    """Port follows the full uncompensated decay (spreading + absorption);
    starboard is flat (already compensated). Pooled, neither curve is
    clean; per side, each classifies as its own TVG status."""
    base = make_raw_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz=_FREQ_HZ)
    base["altitude_m"] = ("ping", np.full(n_pings, 5.0, dtype=np.float32))
    base["fathometer_reading_m"] = ("ping", np.full(n_pings, 5.0, dtype=np.float32))
    ground_range = compute_geometric_ranges(base)["ground_range_m"].values

    full_physics = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=absorption_for_frequency(_FREQ_HZ))
    loss_db = tvg_gain_db(np.clip(ground_range, full_physics.min_range_m, None), full_physics)
    ref = np.nanmin(loss_db[np.isfinite(loss_db)])
    port = np.nan_to_num(10.0 ** (-(loss_db - ref) / 20.0), nan=0.0)
    starboard = np.where(np.isfinite(ground_range), 1.0, 0.0)

    backscatter = np.empty((n_pings, 2, n_samples), dtype=np.float32)
    backscatter[:, 0, :] = port[:, 0, :]
    backscatter[:, 1, :] = starboard[:, 1, :]
    base["backscatter_intensity"] = (["ping", "channel", "sample"], backscatter)
    return base


def _write(tmp_path, datasets: dict, name="two_channel.ss.h5"):
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    for freq_key, ds in datasets.items():
        survey.add_frequency(freq_key, ds)
    path = tmp_path / name
    write_survey(survey, path)
    return path


@pytest.fixture
def two_channel_path(tmp_path):
    return _write(tmp_path, {"hz_200000": _two_channel_dataset()})


def _decay(path, **kwargs):
    raw, _ = _investigate.load_frequency(path)
    _, _, params, results = _investigate.investigate(raw, **kwargs)
    return params, results["range_decay"]


class TestInvestigateCommand:
    def test_writes_plots_metrics_and_summary(self, two_channel_path, tmp_path, monkeypatch):
        out = tmp_path / "out"
        monkeypatch.setattr(analyze, "setup_logging_from_args", lambda args: None)
        monkeypatch.setattr("sys.argv", [
            "sidescan-analyze", "investigate", "--input", str(two_channel_path), "--output-dir", str(out),
        ])
        assert analyze.main() == 0

        freq_dir = out / "hz_200000"
        assert {p.name for p in freq_dir.glob("*.png")} == _EXPECTED_PNGS
        metrics = json.loads((freq_dir / "metrics.json").read_text())
        assert set(metrics["results"]) == _EXPECTED_RESULTS
        assert metrics["freq_key"] == "hz_200000" and metrics["n_pings"] == 40
        assert metrics["params"]["usable_range_bin_width_m"] == 2.0
        assert metrics["params"]["decay_bin_width_m"] == 5.0
        assert not re.search(r"stage", json.dumps(metrics), re.IGNORECASE)

        summary = json.loads((out / "summary.json").read_text())
        assert len(summary) == 1
        assert summary[0]["freq_key"] == "hz_200000"
        assert summary[0]["tvg_status"] == metrics["results"]["range_decay"]["tvg_status"]

    def test_all_frequencies_writes_one_directory_each(self, tmp_path, monkeypatch):
        ds = make_raw_dataset(n_pings=10, n_samples=100)
        path = _write(tmp_path, {"hz_200000": ds, "hz_400000": ds.assign_attrs(freq_hz_start=400_000.0)})
        out = tmp_path / "out"
        monkeypatch.setattr(analyze, "setup_logging_from_args", lambda args: None)
        monkeypatch.setattr("sys.argv", [
            "sidescan-analyze", "investigate", "--input", str(path), "--output-dir", str(out), "--all-frequencies",
        ])
        assert analyze.main() == 0
        assert (out / "hz_200000" / "metrics.json").exists()
        assert (out / "hz_400000" / "metrics.json").exists()
        assert len(json.loads((out / "summary.json").read_text())) == 2

    def test_channel_flag_restricts_decay_fit(self, two_channel_path, tmp_path, monkeypatch):
        out = tmp_path / "out"
        monkeypatch.setattr(analyze, "setup_logging_from_args", lambda args: None)
        monkeypatch.setattr("sys.argv", [
            "sidescan-analyze", "investigate", "--input", str(two_channel_path), "--output-dir", str(out),
            "--channel", "1",
        ])
        assert analyze.main() == 0
        metrics = json.loads((out / "hz_200000" / "metrics.json").read_text())
        assert metrics["params"]["channel"] == 1
        assert metrics["results"]["range_decay"]["tvg_status"] == "full"


class TestInvestigateChannel:
    """``channel`` restricts only the usable-range and decay binning to one
    physical side; the default pools both (which silently blends sides)."""

    def test_default_pools_both_sides(self, two_channel_path):
        params, pooled = _decay(two_channel_path)
        _, explicit_none = _decay(two_channel_path, channel=None)
        assert params["channel"] is None
        assert pooled.best_candidate == explicit_none.best_candidate
        assert pooled.best_rmse_db == pytest.approx(explicit_none.best_rmse_db)

    def test_each_side_classified_independently(self, two_channel_path):
        _, port = _decay(two_channel_path, channel=0)
        _, starboard = _decay(two_channel_path, channel=1)
        assert port.best_candidate == "full_physics"
        assert port.tvg_status == "none"  # nothing compensated yet
        assert starboard.tvg_status == "full"  # already flat

    def test_channel_name_matches_index(self, two_channel_path):
        _, by_name = _decay(two_channel_path, channel="starboard")
        _, by_index = _decay(two_channel_path, channel=1)
        assert by_name.best_candidate == by_index.best_candidate

    def test_pooled_differs_from_either_side(self, two_channel_path):
        _, pooled = _decay(two_channel_path)
        _, port = _decay(two_channel_path, channel=0)
        _, starboard = _decay(two_channel_path, channel=1)
        assert pooled.best_rmse_db != pytest.approx(port.best_rmse_db)
        assert pooled.best_rmse_db != pytest.approx(starboard.best_rmse_db)

    def test_unknown_channel_name_raises(self, two_channel_path):
        with pytest.raises(ValueError, match="not found in channel_id"):
            _decay(two_channel_path, channel="midship")
