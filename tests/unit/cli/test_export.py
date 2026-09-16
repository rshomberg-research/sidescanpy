"""
Unit tests for the sidescan-export CLI's new/promoted subcommands.

``waterfall``/``grid``/``track`` are new consolidations (of the old
03/05/07/09/12 numbered scripts) tested here directly against small
synthetic surveys. ``chip``/``survey-window``/``progression`` are
promoted essentially unchanged from ``project-a/scripts/
14_export_dataset.py``, which was already cross-project real-data
verified — not re-tested from scratch here.

2026 rshomberg@uri.edu
"""

import argparse

import numpy as np
import pytest

from sidescanpy.cli import export
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing import run_correction_steps
from sidescanpy.targets.core import Target
from sidescanpy.targets.io import write_targets_json
from tests.conftest import make_minimal_sidescan_dataset


def _geo_friendly_survey():
    """A survey where slant range genuinely exceeds altitude (unlike the
    default 10-ping/100-sample fixture, whose max slant range is a few
    cm -- well inside the water column at the default 5m altitude,
    which leaves every ground_range_m/latitude_per_sample NaN and
    breaks grid/ground-waterfall, which both need real coverage).

    Also overrides latitude/longitude to a tight, realistic ~100m
    survey-track span -- ``make_minimal_sidescan_dataset``'s own default
    (``np.linspace(40.0, 40.5, n_pings)``, a 0.5-degree/~55km spread) is
    harmless for correction-step tests but pathological for
    ``resample_channel_to_grid``'s plain nested Python loop over grid
    cells: a 55km bbox at 1m resolution is tens of thousands of rows,
    each doing a KDTree query -- reproduced directly as a many-minute
    hang before this override was added, not a bug in the CLI code
    being tested here.
    """
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
    for freq_key, freq_hz in (("hz_1495000", 1_495_000), ("hz_547500", 547_500)):
        ds = make_minimal_sidescan_dataset(n_pings=20, n_samples=2000, freq_hz_start=freq_hz)
        ds["altitude_m"] = ("ping", np.full(20, 0.5, dtype=np.float32))
        ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, 20))
        ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, 20))
        survey.add_frequency(freq_key, ds)
    return survey


@pytest.fixture
def ss_h5_dir(tmp_path, sample_multi_frequency_survey):
    d = tmp_path / "ss_h5"
    d.mkdir()
    write_survey(sample_multi_frequency_survey, d / "survey_a.ss.h5")
    return d


@pytest.fixture
def georef_ss_h5_dir(tmp_path):
    d = tmp_path / "ss_h5"
    d.mkdir()
    corrected = run_correction_steps(_geo_friendly_survey(), {"tvg", "geometric", "georef"})
    write_survey(corrected, d / "survey_a.ss.h5")
    return d


class TestRunWaterfall:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            segment_size=2000, range_mode="slant", x_axis="sample", y_axis="ping",
            cmap="copper", vmin=None, vmax=None, vmin_pct=2.0, vmax_pct=98.0,
            gamma=2.5, dpi=50,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_slant_mode_writes_pngs_for_every_frequency(self, tmp_path, ss_h5_dir):
        assert export.run_waterfall(self._args(tmp_path, ss_h5_dir)) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "survey_a.ss" / "segment_001.png").exists()
        assert (out / "hz_547500" / "survey_a.ss" / "segment_001.png").exists()

    def test_ground_mode_requires_geometric_ranges(self, tmp_path, georef_ss_h5_dir):
        args = self._args(tmp_path, georef_ss_h5_dir, range_mode="ground")
        assert export.run_waterfall(args) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "survey_a.ss" / "segment_001.png").exists()

    def test_ground_mode_without_geometric_ranges_fails_cleanly(self, tmp_path, ss_h5_dir):
        args = self._args(tmp_path, ss_h5_dir, range_mode="ground")
        assert export.run_waterfall(args) == 1  # no successful segments

    def test_no_files_found_returns_1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert export.run_waterfall(self._args(tmp_path, empty)) == 1


class TestRunGrid:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            segment_size=2000, grid_resolution_m=1.0, grid_method="mean",
            cmap="copper", vmin=None, vmax=None, vmin_pct=2.0, vmax_pct=98.0,
            gamma=2.5, dpi=50,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_gridded_pngs(self, tmp_path, georef_ss_h5_dir):
        assert export.run_grid(self._args(tmp_path, georef_ss_h5_dir)) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "survey_a.ss" / "segment_001.png").exists()


class TestRunTrack:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            targets_json=None, scanline_stride=5, dpi=50,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_track_png_per_frequency(self, tmp_path, georef_ss_h5_dir):
        assert export.run_track(self._args(tmp_path, georef_ss_h5_dir)) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "track.png").exists()
        assert (out / "hz_547500" / "track.png").exists()

    def test_with_targets_overlay(self, tmp_path, georef_ss_h5_dir):
        targets = [
            Target(target_id="1", source_format="x", catalog_source="x", latitude=40.1, longitude=-70.05),
        ]
        targets_json = tmp_path / "targets.json"
        write_targets_json(targets, targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, frequency="1495000", targets_json=targets_json)
        assert export.run_track(args) == 0
        assert (tmp_path / "out" / "hz_1495000" / "track.png").exists()

    def test_no_files_found_returns_1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert export.run_track(self._args(tmp_path, empty)) == 1
