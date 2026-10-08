"""
Unit tests for the sidescan-export CLI's new/promoted subcommands.

``waterfall``/``grid``/``track``/``chip``/``survey-window``/``progression``
are tested here directly against small synthetic surveys.

2026 rshomberg@uri.edu
"""

import argparse
import importlib
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from sidescanpy.cli import export
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.core.target import Target
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing import correct_survey
from sidescanpy.targets.io import write_targets_json
from tests.conftest import make_minimal_sidescan_dataset
from tests.unit.export.surveys import geo_friendly_survey, swath_friendly_survey

# The module, not the same-named function viz/__init__ re-exports; run_chip
# imports its plotting helpers from here at call time, so spies go here.
viz_contact_sheet = importlib.import_module("sidescanpy.viz.contact_sheet")
# Window extraction lives in the library; spy on the module that calls the resamplers.
geo_windows = importlib.import_module("sidescanpy.export.windows.geo")
neighborhood = importlib.import_module("sidescanpy.export.windows.neighborhood")


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
    corrected = correct_survey(geo_friendly_survey())
    write_survey(corrected, d / "survey_a.ss.h5")
    return d


@pytest.fixture
def two_file_ss_h5_dir(tmp_path):
    """Two independent 20-ping files, same shape as georef_ss_h5_dir's
    single file -- for testing that waterfall windows/overview span the
    whole survey without restarting at each file boundary."""
    d = tmp_path / "ss_h5"
    d.mkdir()
    write_survey(correct_survey(geo_friendly_survey()), d / "survey_a.ss.h5")
    write_survey(correct_survey(geo_friendly_survey()), d / "survey_b.ss.h5")
    return d


@pytest.fixture
def swath_ss_h5_dir(tmp_path):
    d = tmp_path / "ss_h5"
    d.mkdir()
    corrected = correct_survey(swath_friendly_survey())
    write_survey(corrected, d / "survey_a.ss.h5")
    return d


class TestRunWaterfall:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            window_pings=2000, overlap_pings=0, overview_rows=None, value_var="radiometric",
            range_mode="slant", x_axis="sample", y_axis="ping",
            cmap="copper", vmin=None, vmax=None, vmin_pct=2.0, vmax_pct=98.0,
            gamma=2.5, dpi=50, targets_json=None, max_match_dist_m=25.0, box_size=40.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_slant_mode_writes_pngs_for_every_frequency(self, tmp_path, ss_h5_dir):
        assert export.run_waterfall(self._args(tmp_path, ss_h5_dir)) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "segment_001.png").exists()
        assert (out / "hz_547500" / "segment_001.png").exists()
        # no more per-file subfolder
        assert not (out / "hz_1495000" / "survey_a.ss").exists()

    def test_ground_mode_requires_geometric_ranges(self, tmp_path, georef_ss_h5_dir):
        args = self._args(tmp_path, georef_ss_h5_dir, range_mode="ground")
        assert export.run_waterfall(args) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "segment_001.png").exists()

    def test_ground_mode_without_geometric_ranges_fails_cleanly(self, tmp_path, ss_h5_dir):
        args = self._args(tmp_path, ss_h5_dir, range_mode="ground")
        assert export.run_waterfall(args) == 1  # no successful segments

    def test_no_files_found_returns_1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert export.run_waterfall(self._args(tmp_path, empty)) == 1

    def test_targets_json_unsupported_axes_still_rejected(self, tmp_path, georef_ss_h5_dir):
        """Ground mode itself is now supported (see the tests below) --
        this only checks the rejection still fires for a genuinely
        unsupported combo, i.e. slant mode with a non-default axis."""
        targets_json = tmp_path / "targets.json"
        write_targets_json([Target(target_id="1", source_format="x", catalog_source="x", latitude=40.0, longitude=-70.0)], targets_json)
        args = self._args(tmp_path, georef_ss_h5_dir, targets_json=targets_json, x_axis="ground_range")
        assert export.run_waterfall(args) == 1

    def test_ground_mode_matched_target_gets_a_real_box(self, tmp_path, georef_ss_h5_dir, mocker):
        """Ground-mode target boxes -- the feature this test class exists
        to cover. Same match as test_matched_target_gets_a_box_drawn but
        with --range-mode ground: confirms _target_waterfall_xy actually
        resolves a real (non-None) across_track/along_track position for
        the matched target, not just that the run completes."""
        lat = float(np.linspace(40.0000, 40.0010, 20)[10])
        lon = float(np.linspace(-70.0000, -70.0010, 20)[10])
        target = Target(target_id="42", source_format="x", catalog_source="x", latitude=lat, longitude=lon, channel=1, sample=500)
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)
        spy = mocker.spy(export, "_target_waterfall_xy")

        args = self._args(tmp_path, georef_ss_h5_dir, range_mode="ground", targets_json=targets_json)
        assert export.run_waterfall(args) == 0
        assert spy.call_count >= 1
        assert spy.spy_return is not None

    def test_matched_target_gets_a_box_drawn(self, tmp_path, georef_ss_h5_dir, caplog):
        # ping 10's exact lat/lon from geo_friendly_survey()'s own
        # np.linspace(...) -- guarantees a real, non-ambiguous match.
        lat = float(np.linspace(40.0000, 40.0010, 20)[10])
        lon = float(np.linspace(-70.0000, -70.0010, 20)[10])
        target = Target(target_id="42", source_format="x", catalog_source="x", latitude=lat, longitude=lon, channel=1, sample=500)
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json=targets_json)
        with caplog.at_level("INFO"):
            assert export.run_waterfall(args) == 0
        assert "Overlaying 1/1 target(s) matched to a real ping" in caplog.text

    def test_target_projected_onto_sibling_frequency(self, tmp_path, georef_ss_h5_dir, mocker):
        """A target correlated to hz_1495000 (its own recorded position)
        must also get drawn on hz_547500's own windows, via
        project_target_to_frequency -- same ping, slant-range-equivalent
        sample, since the source catalog never marks a contact on both
        channels itself."""
        lat = float(np.linspace(40.0000, 40.0010, 20)[10])
        lon = float(np.linspace(-70.0000, -70.0010, 20)[10])
        target = Target(target_id="42", source_format="x", catalog_source="x", latitude=lat, longitude=lon, channel=1, sample=500)
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)
        spy = mocker.spy(export, "project_target_to_frequency")

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json=targets_json, frequency="all")
        assert export.run_waterfall(args) == 0
        assert spy.call_count >= 1  # projected onto at least the sibling frequency

    def test_window_spans_across_file_boundary_no_restart(self, tmp_path, two_file_ss_h5_dir):
        """Two 20-ping files, 40 pings total. A 25-ping window must
        produce 2 windows (25 + 15), not 2 restarts of 1-each-with-a-
        leftover per file (the old per-file behavior)."""
        args = self._args(tmp_path, two_file_ss_h5_dir, window_pings=25, frequency="1495000")
        assert export.run_waterfall(args) == 0
        out = tmp_path / "out" / "hz_1495000"
        assert (out / "segment_001.png").exists()
        assert (out / "segment_002.png").exists()
        assert not (out / "segment_003.png").exists()

    def test_overlap_pings_produces_more_windows(self, tmp_path, two_file_ss_h5_dir):
        no_overlap = self._args(tmp_path, two_file_ss_h5_dir, window_pings=10, overlap_pings=0, frequency="1495000", output_dir=tmp_path / "no_overlap")
        overlap = self._args(tmp_path, two_file_ss_h5_dir, window_pings=10, overlap_pings=5, frequency="1495000", output_dir=tmp_path / "overlap")
        assert export.run_waterfall(no_overlap) == 0
        assert export.run_waterfall(overlap) == 0
        n_no_overlap = len(list((tmp_path / "no_overlap" / "hz_1495000").glob("segment_*.png")))
        n_overlap = len(list((tmp_path / "overlap" / "hz_1495000").glob("segment_*.png")))
        assert n_overlap > n_no_overlap

    def test_overlap_not_smaller_than_window_returns_1(self, tmp_path, ss_h5_dir):
        args = self._args(tmp_path, ss_h5_dir, window_pings=10, overlap_pings=10)
        assert export.run_waterfall(args) == 1

    def test_overview_rows_writes_overview_png(self, tmp_path, two_file_ss_h5_dir):
        args = self._args(tmp_path, two_file_ss_h5_dir, overview_rows=10, frequency="1495000")
        assert export.run_waterfall(args) == 0
        assert (tmp_path / "out" / "hz_1495000" / "overview.png").exists()

    def test_value_var_raw_and_radiometric_produce_different_images(self, tmp_path, georef_ss_h5_dir):
        # georef_ss_h5_dir (unlike ss_h5_dir) actually ran TVG -- ss_h5_dir's
        # backscatter_intensity is just an uncorrected copy of
        # backscatter_raw, so the two value_vars would be byte-identical
        # there and this test would prove nothing.
        from PIL import Image

        raw_args = self._args(tmp_path, georef_ss_h5_dir, value_var="uncorrected", frequency="1495000", output_dir=tmp_path / "raw_out")
        rad_args = self._args(tmp_path, georef_ss_h5_dir, value_var="radiometric", frequency="1495000", output_dir=tmp_path / "rad_out")
        assert export.run_waterfall(raw_args) == 0
        assert export.run_waterfall(rad_args) == 0

        raw_img = np.array(Image.open(tmp_path / "raw_out" / "hz_1495000" / "segment_001.png"))
        rad_img = np.array(Image.open(tmp_path / "rad_out" / "hz_1495000" / "segment_001.png"))
        assert not np.array_equal(raw_img, rad_img)


class TestTargetWaterfallXY:
    def test_port_channel_is_negative_x(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=300)
        x, y, x_scale, y_scale = export._target_waterfall_xy(t, None, 57, 0, "sample", "ping")
        assert x == -300.0
        assert y == 57.0
        assert (x_scale, y_scale) == (1.0, 1.0)

    def test_starboard_channel_is_positive_x(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=1, sample=300)
        x, y, x_scale, y_scale = export._target_waterfall_xy(t, None, 57, 0, "sample", "ping")
        assert x == 300.0
        assert y == 57.0
        assert (x_scale, y_scale) == (1.0, 1.0)

    def test_missing_channel_defaults_to_port(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=None, sample=300)
        x, _y, _xs, _ys = export._target_waterfall_xy(t, None, 0, 0, "sample", "ping")
        assert x == -300.0

    def test_y_is_the_global_ping_index_unmodified(self):
        """The whole point of this helper -- plot_waterfall's own
        ping_offset already bakes the segment's start into the y-axis,
        so a target's file-global ping_idx must NOT be re-based to the
        segment here (that was a real bug caught before this shipped)."""
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=0)
        _x, y, _xs, _ys = export._target_waterfall_xy(t, None, 4321, 0, "sample", "ping")
        assert y == 4321.0

    def test_ground_mode_reads_across_track_at_targets_own_position(self):
        """x_axis="across_track"/y_axis="along_track" (--range-mode
        ground): x comes from the real across_track_m value at the
        target's own (local ping, channel, sample) -- not derivable from
        sample index alone the way slant mode's x is."""
        across_track = xr.DataArray(
            np.arange(2 * 2 * 10, dtype=np.float64).reshape(2, 2, 10),
            dims=("ping", "channel", "sample"),
        )
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=1, sample=3)
        # ping_offset=100, target's global ping_idx=101 -> local ping 1
        x, y, _xs, _ys = export._target_waterfall_xy(t, ds, 101, 100, "across_track", "along_track", along_track=np.array([0.0, 12.5]))
        assert x == float(across_track.isel(ping=1, channel=1, sample=3).values)
        assert y == 12.5

    def test_ground_mode_scale_factors_are_real_meters_not_one(self):
        """The bug this whole helper's scale-factor return exists to fix:
        a shared --box-size (a sample/ping pixel count) must NOT be
        applied as literal meters in ground mode -- x_scale/y_scale carry
        the real meters-per-sample/meters-per-ping conversion so the
        caller can size the box correctly."""
        # across_track steps by 2.0m per sample (0, 2, 4, ...) on this ping/channel.
        across_track = xr.DataArray(
            (np.arange(10, dtype=np.float64) * 2.0).reshape(1, 1, 10),
            dims=("ping", "channel", "sample"),
        )
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        # along_track steps by 5.0m per ping.
        along_track = np.array([0.0, 5.0, 10.0])
        _x, _y, x_scale, y_scale = export._target_waterfall_xy(t, ds, 100, 100, "across_track", "along_track", along_track=along_track)
        assert x_scale == pytest.approx(2.0)
        assert y_scale == pytest.approx(5.0)

    def test_ground_mode_out_of_window_ping_returns_none(self):
        ds = xr.Dataset({"across_track_m": xr.DataArray(np.zeros((2, 2, 10)), dims=("ping", "channel", "sample"))})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert export._target_waterfall_xy(t, ds, 999, 100, "across_track", "along_track", along_track=np.array([0.0, 1.0])) is None

    def test_ground_mode_nan_across_track_returns_none(self):
        """A target still in the water column at its own position has no
        real across-track value -- skip its box rather than drawing one
        at a garbage/NaN location."""
        across_track = xr.DataArray(np.full((1, 2, 10), np.nan), dims=("ping", "channel", "sample"))
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert export._target_waterfall_xy(t, ds, 100, 100, "across_track", "along_track", along_track=np.array([0.0])) is None

    def test_unsupported_axis_combo_returns_none(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert export._target_waterfall_xy(t, None, 0, 0, "ground_range", "time") is None


class TestRunGrid:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            segment_size=2000, grid_resolution_m=1.0, grid_method="mean",
            cmap="copper", vmin=None, vmax=None, vmin_pct=2.0, vmax_pct=98.0,
            gamma=2.5, dpi=50, show_land=False, show_locator=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_gridded_pngs(self, tmp_path, georef_ss_h5_dir):
        assert export.run_grid(self._args(tmp_path, georef_ss_h5_dir)) == 0
        out = tmp_path / "out"
        assert (out / "hz_1495000" / "survey_a.ss" / "segment_001.png").exists()

    def test_reads_lazily_chunked_by_segment_size(self, tmp_path, georef_ss_h5_dir, mocker):
        """Out-of-memory regression: this used to call read_survey(path)
        with no chunks= at all, loading the whole file eagerly before any
        segmenting happened -- defeating --segment-size's own purpose on
        a large real survey. Every other exporter (waterfall/track/chip)
        already read lazily; this one didn't."""
        spy = mocker.spy(export, "read_survey")
        assert export.run_grid(self._args(tmp_path, georef_ss_h5_dir, segment_size=37)) == 0
        assert spy.call_count >= 1
        _args, kwargs = spy.call_args_list[0]
        assert kwargs.get("chunks") == {"ping": 37}


class TestRunTrack:
    def _args(self, tmp_path, ss_h5_dir, **overrides):
        defaults = dict(
            ss_h5_dir=ss_h5_dir, output_dir=tmp_path / "out", frequency="all",
            targets_json=None, scanline_stride=5, dpi=50, show_land=False, show_locator=False,
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


class TestRunChip:
    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="radiometric",
            size_px=200, swath_size_px=200, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _matched_target(self):
        lat = float(np.linspace(40.0000, 40.0010, 20)[10])
        lon = float(np.linspace(-70.0000, -70.0010, 20)[10])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=500,
            position_confidence="verified",
        )

    def test_without_context_no_context_dir(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json)
        assert export.run_chip(args) == 0

        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        assert not (tmp_path / "out" / "context").exists()

    def test_with_context_writes_boxed_context_image(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, with_context=True)
        assert export.run_chip(args) == 0

        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        context_path = tmp_path / "out" / "context" / "context_7.png"
        assert context_path.exists()
        assert context_path.stat().st_size > 0

    def test_with_context_geometric_render_level_writes_boxed_context_image(self, tmp_path, georef_ss_h5_dir):
        """Real user ask, 2026-09-28: context in geometric/georef
        coordinates, not just native pixel space -- context now follows
        --render-level's own coordinate choice instead of always being a
        fixed pixel-space view."""
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, with_context=True, render_level="geometric")
        assert export.run_chip(args) == 0

        context_path = tmp_path / "out" / "context" / "context_7.png"
        assert context_path.exists()
        assert context_path.stat().st_size > 0

    def test_with_context_georef_render_level_writes_boxed_context_image(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, with_context=True, render_level="georef")
        assert export.run_chip(args) == 0

        context_path = tmp_path / "out" / "context" / "context_7.png"
        assert context_path.exists()
        assert context_path.stat().st_size > 0

    def test_value_var_raw_changes_geometric_chip_pixels(self, tmp_path, georef_ss_h5_dir):
        """--value-var is the direct answer to the other half of the same
        2026-09-28 request: raw vs. radiometric independently selectable
        from the geometric/georef coordinate choice, not fixed to
        radiometric. A real pixel-value difference, not just "doesn't
        crash" -- backscatter_raw vs. backscatter_intensity are genuinely
        different arrays (TVG correction changes the values)."""
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        out_radiometric = tmp_path / "out_radiometric"
        args_radiometric = self._args(
            tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
            value_var="radiometric", output=out_radiometric, seed=0,
        )
        assert export.run_chip(args_radiometric) == 0

        out_raw = tmp_path / "out_raw"
        args_raw = self._args(
            tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
            value_var="uncorrected", output=out_raw, seed=0,
        )
        assert export.run_chip(args_raw) == 0

        img_radiometric = np.asarray(Image.open(out_radiometric / "chip" / "target_7.png"))
        img_raw = np.asarray(Image.open(out_raw / "chip" / "target_7.png"))
        assert img_radiometric.shape == img_raw.shape
        assert not np.array_equal(img_radiometric, img_raw)

    def test_value_var_db_changes_geometric_chip_pixels(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        imgs = {}
        for value_var in ("uncorrected", "db"):
            out = tmp_path / f"out_{value_var}"
            args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric", value_var=value_var, output=out, seed=0)
            assert export.run_chip(args) == 0
            imgs[value_var] = np.asarray(Image.open(out / "chip" / "target_7.png"))
        assert imgs["db"].shape == imgs["uncorrected"].shape
        assert not np.array_equal(imgs["db"], imgs["uncorrected"])

    def test_value_var_uncorrected_has_no_effect_and_warns_under_radiometric_render_level(self, tmp_path, georef_ss_h5_dir, caplog):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", value_var="uncorrected")
        assert export.run_chip(args) == 0
        assert any("--value-var=uncorrected has no effect" in r.message for r in caplog.records)

    def test_min_confidence_reaches_coco_yolo_writers(self, tmp_path, georef_ss_h5_dir):
        """Real bug found running this against real Klein SDF data: Klein
        targets are unconditionally position_confidence='unknown', and
        write_coco()/write_yolo() have their OWN independent
        min_confidence='verified' default, separate from run_chip's own
        target-list filter above -- passing --min-confidence unknown
        got real crops written for every target but zero COCO/YOLO
        annotations for any of them, because run_chip never threaded
        args.min_confidence through to either writer call."""
        target = replace(self._matched_target(), position_confidence="unknown")
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, min_confidence="unknown")
        assert export.run_chip(args) == 0

        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1

    def test_no_matched_targets_returns_1_not_silent_success(self, tmp_path, georef_ss_h5_dir):
        """Regression: this
        path logged an ERROR but returned None, which main()'s own
        `result if isinstance(result, int) else 0` fallback then turned
        into a *successful* exit code -- a script relying on this to
        fail loudly (e.g. `set -e`) would silently continue instead."""
        far_away = Target(
            target_id="9", source_format="x", catalog_source="x",
            latitude=1.0, longitude=1.0, channel=0, sample=0,
            position_confidence="verified",
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([far_away], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json)
        assert export.run_chip(args) == 1

    def _swath_matched_target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    def test_default_styles_produce_three_parallel_datasets(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json,
            styles=["chip", "single-swath", "dual-swath"], swath_size_px=64,
        )
        assert export.run_chip(args) == 0

        for style in ("chip", "single-swath", "dual-swath"):
            style_dir = tmp_path / "out" / style
            assert (style_dir / "target_7.png").exists()
            coco = json.loads((style_dir / "targets.coco.json").read_text())
            assert len(coco["images"]) == 1
            assert len(coco["annotations"]) == 1

    def test_contact_sheet_gets_boxes(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(viz_contact_sheet, "contact_sheet")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, styles=["chip"])
        assert export.run_chip(args) == 0

        assert spy.call_count == 1
        _call_args, kwargs = spy.call_args
        # one thumbnail, and the primary target is always inside its own
        # window (targets_within_window's own contract) -- so exactly
        # one box list with exactly the primary target in it.
        assert len(kwargs["boxes"]) == 1
        assert len(kwargs["boxes"][0]) == 1
        assert kwargs["boxes"][0][0].target_id == "7"

    def test_context_combines_both_channels(self, tmp_path, swath_ss_h5_dir, mocker):
        """The user asked for context plots to show both port and
        starboard, not just the target's own side -- verify by spying on
        render_target_context's own array argument, since the saved PNG's
        pixel dimensions don't directly reflect the source array's shape
        (matplotlib renders it at a fixed figure size/DPI)."""
        spy = mocker.spy(viz_contact_sheet, "render_target_context")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, with_context=True, styles=["chip"])
        assert export.run_chip(args) == 0

        assert spy.call_count == 1
        combined_array = spy.call_args[0][0]
        assert combined_array.shape[1] == 2 * 30  # n_samples from _swath_friendly_survey's default

    def test_geometric_render_level_calls_geometric_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        # extract_geometric_window_griddata (scipy, the DEFAULT
        # resample_method) -- 2026-09-29, real user direction after
        # reviewing a real-data comparison of kdtree/binned/griddata:
        # griddata chosen as the default, all three kept selectable via
        # --resample-method (see cli/export.py's own comment at the call
        # site and TestRunChipResampleMethod below for the dispatch
        # coverage across all three options).
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric",
            styles=["chip", "single-swath", "dual-swath"], swath_size_px=64,
        )
        assert export.run_chip(args) == 0
        assert spy.call_count == 3  # once per style
        for style in ("chip", "single-swath", "dual-swath"):
            assert (tmp_path / "out" / style / "target_7.png").exists()

    def test_geometric_context_reuses_dual_swath_window(self, tmp_path, swath_ss_h5_dir, mocker):
        """--with-context used to run a second, identical dual-swath
        griddata pass per target (~11s each on real data)."""
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric",
            styles=["dual-swath"], swath_size_px=64, with_context=True,
        )
        assert export.run_chip(args) == 0
        assert spy.call_count == 1
        assert (tmp_path / "out" / "context" / "context_7.png").exists()

    def test_resolution_m_override_reaches_geometric_extraction_for_every_style(self, tmp_path, swath_ss_h5_dir, mocker):
        """Real production report: some targets' chip/single-swath/dual-swath
        came out 'way zoomed out', mostly-black with the real swath tiny in
        the center -- traced to the per-target auto-calculated resolution_m
        (derived from a wide ping-neighborhood-average spacing) blowing up
        for a target sitting in a locally slow/dense-ping region. --resolution-m
        lets the operator set a known-good value directly, bypassing that
        auto-calculation entirely for every style."""
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric",
            styles=["chip", "single-swath", "dual-swath"], swath_size_px=64,
            resolution_m=0.25,
        )
        assert export.run_chip(args) == 0
        assert spy.call_count == 3
        for call in spy.call_args_list:
            assert call.kwargs["resolution_m"] == 0.25


    def test_resolution_m_override_reaches_georef_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_georef_window")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="georef",
            styles=["chip"], swath_size_px=64, resolution_m=0.1,
        )
        assert export.run_chip(args) == 0
        assert spy.call_count == 1
        assert spy.call_args.kwargs["resolution_m"] == 0.1

    def test_resolution_m_override_ignored_with_a_warning_for_pixel_styles(self, tmp_path, swath_ss_h5_dir, caplog):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric",
            styles=["chip"], resolution_m=0.25,
        )
        with caplog.at_level("WARNING"):
            assert export.run_chip(args) == 0
        assert any("has no effect" in r.message for r in caplog.records)

    def test_georef_render_level_calls_georef_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_georef_window")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="georef",
            styles=["chip", "dual-swath"], swath_size_px=64,
        )
        assert export.run_chip(args) == 0
        assert spy.call_count == 2
        for style in ("chip", "dual-swath"):
            assert (tmp_path / "out" / style / "target_7.png").exists()

    def test_geometric_render_level_annotates_co_located_targets(self, tmp_path, swath_ss_h5_dir):
        """Real-data gap: _extract_geo_style_window used to only ever
        annotate the primary target, unlike the raw-pixel styles' own
        targets_within_window search -- two targets close enough to share
        one geometric chip must both end up in that chip's own COCO
        annotations, not just the one the crop was centered on."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        primary = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[150]), longitude=float(lon[150]), channel=1, sample=15,
            position_confidence="verified",
        )
        co_located = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[152]), longitude=float(lon[152]), channel=1, sample=16,
            position_confidence="verified",
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([primary, co_located], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"])
        assert export.run_chip(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        file_names = {img["file_name"] for img in coco["images"]}
        # each target still gets its own centered chip file...
        assert {"target_7.png", "target_8.png"} <= file_names
        # ...but target 8's chip should also carry an annotation for target 7
        # (found via co-target search), not just its own -- 2+ annotations
        # total for at least one of the two images.
        by_image = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], 0)
            by_image[ann["image_id"]] += 1
        assert max(by_image.values()) >= 2

    def test_raw_and_radiometric_render_levels_produce_different_images(self, tmp_path, swath_ss_h5_dir):
        """backscatter_raw and backscatter_intensity are byte-identical in
        this fixture only if TVG is a no-op there -- correct_survey
        with 'tvg' genuinely modifies backscatter_intensity, so raw and
        radiometric renders of the same target must differ."""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        raw_args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="uncorrected", styles=["chip"], output=tmp_path / "raw_out")
        radiometric_args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=tmp_path / "rad_out")
        assert export.run_chip(raw_args) == 0
        assert export.run_chip(radiometric_args) == 0

        raw_img = np.array(Image.open(tmp_path / "raw_out" / "chip" / "target_7.png"))
        rad_img = np.array(Image.open(tmp_path / "rad_out" / "chip" / "target_7.png"))
        assert not np.array_equal(raw_img, rad_img)

    def test_bit_depth_16_writes_16bit_png_with_same_annotations(self, tmp_path, swath_ss_h5_dir):
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        out8, out16 = tmp_path / "out8", tmp_path / "out16"
        assert export.run_chip(self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=out8)) == 0
        assert export.run_chip(self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=out16, bit_depth=16)) == 0

        img8 = np.array(Image.open(out8 / "chip" / "target_7.png"))
        img16 = np.array(Image.open(out16 / "chip" / "target_7.png"))
        assert img8.dtype == np.uint8
        assert img16.dtype == np.uint16
        assert img16.max() > 255
        assert np.abs((img16 >> 8).astype(int) - img8.astype(int)).max() <= 1
        coco8 = json.loads((out8 / "chip" / "targets.coco.json").read_text())
        coco16 = json.loads((out16 / "chip" / "targets.coco.json").read_text())
        assert coco8["annotations"] == coco16["annotations"]

    def test_db_render_level_differs_from_raw(self, tmp_path, swath_ss_h5_dir):
        """'db' is 20*log10(backscatter_raw) with no TVG -- a nonlinear
        transform of raw, so its percentile-stretched 8-bit render differs
        from raw's. (Not compared against radiometric: this fixture's whole
        swath is ~0.06m, all inside TVG's min_range_m clamp, so its gain is
        a constant offset that percentile stretching removes.)"""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        imgs = {}
        for level in ("uncorrected", "db"):
            args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level=level, styles=["chip"], output=tmp_path / f"{level}_out")
            assert export.run_chip(args) == 0
            imgs[level] = np.array(Image.open(tmp_path / f"{level}_out" / "chip" / "target_7.png"))
        assert not np.array_equal(imgs["db"], imgs["uncorrected"])

    def test_geometric_level_without_geometric_correction_returns_1(self, tmp_path, ss_h5_dir):
        """ss_h5_dir's survey was never run through sidescan-correct
        --geometric -- across_track_m doesn't exist at all. Must fail
        gracefully (return 1, real per-target warnings), not crash with
        an uncaught KeyError."""
        lat = float(np.linspace(40.0, 40.5, 10)[5])
        lon = float(np.linspace(-70.0, -69.9, 10)[5])
        target = Target(
            target_id="1", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=0, sample=10,
            position_confidence="verified",
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, ss_h5_dir, targets_json, render_level="geometric", styles=["chip"])
        assert export.run_chip(args) == 1

    def test_pixel_style_uses_real_bbox_size(self, tmp_path, georef_ss_h5_dir):
        """Regression: chip/COCO output had every target's box
        at the same size, regardless of each target's own real, measured
        extent (e.g. a catalog's 4-corner columns). "chip"
        style's native window IS size_px already (extract_chip produces
        that size directly, no separate resize step) -- size_px=8 here,
        well inside georef_ss_h5_dir's 20 real pings, so no resize is
        triggered on either axis and the expected bbox comes through
        unscaled, an exact check."""
        t = replace(self._matched_target(), bbox=(490.0, 95.0, 4.0, 3.0))
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], size_px=8)
        assert export.run_chip(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        ann = coco["annotations"][0]
        assert ann["bbox"][2] == pytest.approx(4.0)
        assert ann["bbox"][3] == pytest.approx(3.0)

    def test_contact_sheet_and_context_draw_the_coco_box(self, tmp_path, georef_ss_h5_dir, mocker):
        """QA images used to draw the --box-size placeholder even when the
        COCO/YOLO files carried the catalog's real extent, so real boxes
        looked generated."""
        from sidescanpy.viz.contact_sheet import _target_box_size

        sheet = mocker.spy(viz_contact_sheet, "contact_sheet")
        context = mocker.spy(viz_contact_sheet, "render_target_context")
        t = replace(self._matched_target(), bbox=(490.0, 95.0, 4.0, 3.0))
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"],
                          size_px=16, box_size=40.0, with_context=True)
        assert export.run_chip(args) == 0

        ann = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())["annotations"][0]
        drawn = _target_box_size(sheet.call_args.kwargs["boxes"][0][0], (40.0, 40.0))
        assert drawn == pytest.approx((ann["bbox"][2], ann["bbox"][3]))
        assert drawn != pytest.approx((40.0, 40.0))
        context_target = context.call_args.args[1]
        assert _target_box_size(context_target, (40.0, 40.0)) != pytest.approx((40.0, 40.0))

    def test_pixel_style_no_bbox_falls_back_to_shared_box_size(self, tmp_path, georef_ss_h5_dir):
        """A center-only pick (bbox=None, e.g. JSF/EdgeTech Discover's own
        catalog) must still fall back to the shared --box-size placeholder
        exactly as before this fix existed -- not a regression for the
        format this project was originally built around."""
        t = self._matched_target()
        assert t.bbox is None
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], box_size=40.0)
        assert export.run_chip(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        ann = coco["annotations"][0]
        assert ann["bbox"][2] == pytest.approx(40.0)
        assert ann["bbox"][3] == pytest.approx(40.0)

    def test_pixel_style_different_targets_get_different_bbox_sizes(self, tmp_path, swath_ss_h5_dir):
        """The actual "all bounding boxes are the same size" symptom,
        reproduced directly: two targets with genuinely different
        measured extents must come out with genuinely different COCO box
        sizes, not the one shared placeholder for both. size_px=10 here,
        well inside swath_ss_h5_dir's 30 real samples (and 300 real
        pings), so no resize is triggered on either axis."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        small = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[100]), longitude=float(lon[100]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 145.0, 3.0, 2.0),
        )
        big = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[200]), longitude=float(lon[200]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 245.0, 9.0, 7.0),
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([small, big], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], size_px=10)
        assert export.run_chip(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        sizes = {
            next(img["file_name"] for img in coco["images"] if img["id"] == ann["image_id"]): tuple(ann["bbox"][2:])
            for ann in coco["annotations"]
        }
        assert sizes["target_7.png"] != sizes["target_8.png"]
        assert sizes["target_7.png"] == pytest.approx((3.0, 2.0))
        assert sizes["target_8.png"] == pytest.approx((9.0, 7.0))

    def test_geometric_style_different_targets_get_different_bbox_sizes(self, tmp_path, swath_ss_h5_dir):
        """Same real bug, geo-rendered path -- this is the render level
        used for geometric export (--render-level geometric). Exact pixel values
        depend on real meters-per-sample/meters-per-ping/resolution_m
        conversion, so this checks the sizes differ and are neither the
        shared placeholder, not exact numbers."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        small = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[100]), longitude=float(lon[100]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 145.0, 4.0, 4.0),
        )
        big = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[200]), longitude=float(lon[200]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 245.0, 40.0, 40.0),
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([small, big], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"], box_size=99.0)
        assert export.run_chip(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        sizes = {
            next(img["file_name"] for img in coco["images"] if img["id"] == ann["image_id"]): tuple(ann["bbox"][2:])
            for ann in coco["annotations"]
        }
        # Not asserting *which* one comes out bigger -- resolution_m
        # (meters per output pixel) is recomputed independently per
        # target/window from that window's own local geometry, so it
        # isn't necessarily comparable across two different ping
        # positions the way it would be for two crops of the same
        # window. The real bug this test guards is "every target gets
        # the same box regardless of its own measured extent" -- these
        # two assertions are the direct, robust check for that.
        assert sizes["target_7.png"] != sizes["target_8.png"]
        # neither equals the shared placeholder (chosen far from either
        # target's real extent to avoid a coincidental collision) --
        # confirms both actually used their own measured extent, not a
        # fallback.
        assert sizes["target_7.png"] != (99.0, 99.0)
        assert sizes["target_8.png"] != (99.0, 99.0)

    def test_geometric_level_out_of_bounds_sample_skips_gracefully(self, tmp_path, swath_ss_h5_dir):
        """Real crash found running this against real Klein SDF data: a
        target whose catalog entry carries no ping_meters_per_sample
        (match_frequency then picks a frequency arbitrarily) can be
        correlated to a channel whose own sample count is smaller than
        the sample index the target's position was originally captured
        against elsewhere -- an exact .isel(sample=t.sample) then raised
        an uncaught IndexError instead of being skipped like every other
        unavailable-window case."""
        target = replace(self._swath_matched_target(), sample=999)  # fixture's n_samples=30
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"])
        assert export.run_chip(args) == 1  # gracefully -- not a crash


class TestRunChipResampleMethod:
    """2026-09-29: --resample-method selects which of the three
    extract_geometric_window* functions the 'geometric' render level
    uses, default 'griddata' -- real user direction after reviewing a
    real-data comparison of all three."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    @pytest.mark.parametrize(
        "resample_method,spied_fn",
        [
            ("kdtree", "extract_geometric_window"),
            ("binned", "extract_geometric_window_binned"),
            ("griddata", "extract_geometric_window_griddata"),
        ],
    )
    def test_each_option_calls_its_own_function(self, tmp_path, swath_ss_h5_dir, mocker, resample_method, spied_fn):
        spy = mocker.spy(geo_windows, spied_fn)
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, resample_method=resample_method)
        assert export.run_chip(args) == 0
        assert spy.call_count == 1

    def test_omitting_resample_method_defaults_to_griddata(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        # No resample_method key at all -- matches an argparse.Namespace
        # built before this flag existed, or a caller that never sets it.
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=swath_ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        args = argparse.Namespace(**defaults)
        assert export.run_chip(args) == 0
        assert spy.call_count == 1

    def test_all_three_options_produce_square_output(self, tmp_path, swath_ss_h5_dir):
        # Real, user-confirmed requirement: no resample_method should
        # ever produce non-square "geometric" render-level output.
        for resample_method in ("kdtree", "binned", "griddata"):
            targets_json = tmp_path / f"targets_{resample_method}.json"
            write_targets_json([self._target()], targets_json)
            args = self._args(
                tmp_path, swath_ss_h5_dir, targets_json, resample_method=resample_method,
                output=tmp_path / f"out_{resample_method}",
            )
            assert export.run_chip(args) == 0
            from PIL import Image
            img = Image.open(tmp_path / f"out_{resample_method}" / "chip" / "target_7.png")
            assert img.size == (64, 64)


class TestPingChunksAlignment:
    """Real user report, 2026-09-28: a UserWarning ("specified chunks
    separate the stored chunks...") fired repeatedly during real
    pipeline runs. Root cause here: _DEFAULT_SEGMENT_SIZE (shared by
    --window-pings and --segment-size, the latter feeding directly into
    read_survey(chunks=...) for run_grid) and run_track's own chunk size
    were both 2000, not a multiple of HDF5_CHUNK_PING (the on-disk chunk
    size write_survey() actually stores, 256)."""

    def test_default_segment_size_is_aligned(self):
        assert export._DEFAULT_SEGMENT_SIZE % HDF5_CHUNK_PING == 0

    def test_reading_a_real_multi_chunk_survey_raises_no_alignment_warning(self, tmp_path):
        n_pings = HDF5_CHUNK_PING * 3
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=200, freq_hz_start=1_495_000)
        survey.add_frequency("hz_1495000", ds)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_survey(path, chunks={"ping": export._DEFAULT_SEGMENT_SIZE})

        alignment_warnings = [w for w in caught if "specified chunks separate the stored chunks" in str(w.message)]
        assert not alignment_warnings


class TestSwathPreResampleDownsample:
    """Block-averaging the neighborhood before resampling must not move annotations."""

    def test_annotation_matches_full_resolution_resample(self, tmp_path, georef_ss_h5_dir, monkeypatch):
        targets_json = tmp_path / "targets.json"
        write_targets_json([TestRunChip()._matched_target()], targets_json)

        def run(out, oversample):
            monkeypatch.setattr(neighborhood, "SWATH_OVERSAMPLE", oversample)
            args = TestRunChip()._args(
                tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
                styles=["single-swath", "dual-swath"], swath_size_px=64, output=out,
            )
            assert export.run_chip(args) == 0
            return {
                style: json.loads((out / style / "targets.coco.json").read_text())["annotations"][0]["bbox"]
                for style in ("single-swath", "dual-swath")
            }

        averaged = run(tmp_path / "averaged", 2.0)
        full = run(tmp_path / "full", 1e9)  # every factor clamps to 1: no averaging
        for style in averaged:
            np.testing.assert_allclose(averaged[style][:2], full[style][:2], atol=2.0)


class TestRunChipAugmentation:
    """--n-augment / --random-flip / --across-axis slant / --box-size-m."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=10.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    def _yolo_centers(self, out_dir, style="chip"):
        labels = sorted((out_dir / style / "targets.yolo" / "labels").glob("*.txt"))
        return {p.stem: [tuple(map(float, line.split()[1:3])) for line in p.read_text().splitlines()] for p in labels}

    def test_n_augment_writes_one_image_and_label_per_draw(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, n_augment=4, jitter_px=20)
        assert export.run_chip(args) == 0

        chip_dir = tmp_path / "out" / "chip"
        assert sorted(p.name for p in chip_dir.glob("target_*.png")) == [f"target_7_a{k}.png" for k in range(4)]
        centers = self._yolo_centers(tmp_path / "out")
        assert set(centers) == {f"target_7_a{k}" for k in range(4)}
        # independent jitter draws -> target lands in different places
        assert len({c[0] for c in centers.values()}) > 1
        log = json.loads((chip_dir / "augmentations.json").read_text())
        assert [e["augmentation"] for e in log] == [0, 1, 2, 3]

    def test_n_augment_default_keeps_single_file_naming(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        assert export.run_chip(self._args(tmp_path, swath_ss_h5_dir, targets_json)) == 0
        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        assert not (tmp_path / "out" / "chip" / "augmentations.json").exists()

    def test_random_flip_mirrors_image_and_label_together(self, tmp_path, swath_ss_h5_dir):
        """Same seed, same jitter draw: the flipped run's image must be the
        unflipped image mirrored, and its label center mirrored to match."""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        plain = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "plain", n_augment=6)
        flipped = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "flip", n_augment=6, random_flip=True)
        assert export.run_chip(plain) == 0
        assert export.run_chip(flipped) == 0

        log = json.loads((tmp_path / "flip" / "chip" / "augmentations.json").read_text())
        assert any(e["flip_h"] or e["flip_v"] for e in log)
        plain_c, flip_c = self._yolo_centers(tmp_path / "plain"), self._yolo_centers(tmp_path / "flip")
        for e in log:
            stem = Path(e["file_name"]).stem
            a = np.array(Image.open(tmp_path / "plain" / "chip" / e["file_name"]))
            b = np.array(Image.open(tmp_path / "flip" / "chip" / e["file_name"]))
            if e["flip_h"]:
                a = a[:, ::-1]
            if e["flip_v"]:
                a = a[::-1, :]
            np.testing.assert_array_equal(a, b)
            (cx, cy), (fx, fy) = plain_c[stem][0], flip_c[stem][0]
            assert fx == pytest.approx(1 - cx if e["flip_h"] else cx, abs=1e-5)
            assert fy == pytest.approx(1 - cy if e["flip_v"] else cy, abs=1e-5)

    def test_slant_across_axis_runs_end_to_end(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, across_axis="slant", resolution_m=0.002, n_augment=2, random_flip=True)
        assert export.run_chip(args) == 0
        assert len(list((tmp_path / "out" / "chip").glob("target_7_a*.png"))) == 2

    def test_box_size_m_scales_with_resolution(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        widths = {}
        for res in (0.05, 0.1):
            out = tmp_path / f"r{res}"
            args = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=out, resolution_m=res, box_size_m=1.0)
            assert export.run_chip(args) == 0
            line = (out / "chip" / "targets.yolo" / "labels" / "target_7.txt").read_text().split()
            widths[res] = float(line[3]) * 64
        assert widths[0.05] == pytest.approx(20.0, abs=0.5)
        assert widths[0.1] == pytest.approx(10.0, abs=0.5)


class TestRunChipBackground:
    """--n-background: no-target images rendered through the chip path."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip", "dual-swath"], render_level="geometric",
            size_px=16, swath_size_px=16, box_size=4.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None, background_max_nan_frac=1.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _write_targets(self, tmp_path):
        targets_json = tmp_path / "targets.json"
        write_targets_json([TestRunChipAugmentation()._target()], targets_json)
        return targets_json

    @pytest.mark.parametrize("render_level", ["radiometric", "geometric"])
    def test_background_images_have_empty_labels(self, tmp_path, swath_ss_h5_dir, render_level):
        targets_json = self._write_targets(tmp_path)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, n_background=2, render_level=render_level)
        assert export.run_chip(args) == 0
        for style in ("chip", "dual-swath"):
            style_dir = tmp_path / "out" / style
            assert sorted(p.name for p in style_dir.glob("bg_*.png")) == ["bg_0.png", "bg_1.png"]
            labels = style_dir / "targets.yolo" / "labels"
            assert (labels / "bg_0.txt").read_text() == "" and (labels / "bg_1.txt").read_text() == ""
            assert (labels / "target_7.txt").read_text().strip()
            coco = json.loads((style_dir / "targets.coco.json").read_text())
            bg_ids = {im["id"] for im in coco["images"] if im["file_name"].startswith("bg_")}
            assert len(bg_ids) == 2
            assert not any(a["image_id"] in bg_ids for a in coco["annotations"])
            log = json.loads((style_dir / "augmentations.json").read_text())
            bg_log = [e for e in log if e["target_id"] is None]
            assert [e["background"] for e in bg_log] == [0, 1]
            assert all(e["n_targets"] == 0 and abs(e["ping"] - 150) > 10 for e in bg_log)

    def test_targets_unchanged_by_backgrounds(self, tmp_path, swath_ss_h5_dir):
        targets_json = self._write_targets(tmp_path)
        plain = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "plain", jitter_px=3)
        with_bg = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "bg", jitter_px=3, n_background=1)
        assert export.run_chip(plain) == 0
        assert export.run_chip(with_bg) == 0
        for style in ("chip", "dual-swath"):
            a = (tmp_path / "plain" / style / "target_7.png").read_bytes()
            assert a == (tmp_path / "bg" / style / "target_7.png").read_bytes()
        assert not list((tmp_path / "plain" / "chip").glob("bg_*"))

    def test_augmented_background_naming_and_ping_ranges(self, tmp_path, swath_ss_h5_dir):
        targets_json = self._write_targets(tmp_path)
        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, styles=["chip"], n_background=1, n_augment=2, random_flip=True,
            jitter_px=2, background_ping_ranges='{"survey_a.ss.h5": [[0, 60]]}',
        )
        assert export.run_chip(args) == 0
        chip_dir = tmp_path / "out" / "chip"
        assert sorted(p.name for p in chip_dir.glob("bg_*.png")) == ["bg_0_a0.png", "bg_0_a1.png"]
        log = [e for e in json.loads((chip_dir / "augmentations.json").read_text()) if e["target_id"] is None]
        assert all(0 <= e["ping"] < 60 for e in log)
        assert {"flip_h", "flip_v", "nan_fraction", "source_file", "freq_key"} <= set(log[0])

    def test_ping_ranges_from_file(self, tmp_path):
        ranges = tmp_path / "ranges.json"
        ranges.write_text(json.dumps({"a.ss.h5": [[1, 5], [9, 12]]}))
        assert export._load_ping_ranges(ranges) == {"a.ss.h5": [(1, 5), (9, 12)]}
        assert export._load_ping_ranges(str(ranges)) == {"a.ss.h5": [(1, 5), (9, 12)]}
        assert export._load_ping_ranges({"b": [[0, 1]]}) == {"b": [(0, 1)]}
        assert export._load_ping_ranges(None) is None


class TestSwathFill:
    """--swath-fill end to end (unit tests: tests/unit/export/test_windows_geo.py)."""

    def test_fill_resolution_is_not_floored_at_ping_spacing(self, tmp_path, swath_ss_h5_dir, mocker):
        """Fixture swath (~0.1 m) is far narrower than ping spacing
        (~0.37 m): the old frame floors resolution at ping spacing (swath
        a sliver of the image); --swath-fill uses width / size_px."""
        lat = float(np.linspace(40.0000, 40.0010, 300)[150])
        lon = float(np.linspace(-70.0000, -70.0010, 300)[150])
        t = Target(target_id="7", source_format="x", catalog_source="x", latitude=lat, longitude=lon,
                   channel=1, sample=15, position_confidence="verified")
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        res = {}
        for fill in (False, True):
            spy.reset_mock()
            args = argparse.Namespace(
                targets_json=targets_json, ss_h5_dir=swath_ss_h5_dir, max_match_dist_m=25.0,
                output=tmp_path / f"out{fill}", styles=["dual-swath"], render_level="geometric",
                size_px=64, swath_size_px=64, box_size=10.0, with_context=False, jitter_px=0, gamma=1.0,
                pct_low=1.0, pct_high=99.0, normalization_scope="image", min_confidence="verified", seed=0,
                resolution_m=None, swath_fill=fill,
            )
            assert export.run_chip(args) == 0
            res[fill] = spy.call_args.kwargs["resolution_m"]
        assert res[False] > 0.3          # ping-spacing floor
        assert res[True] < 0.01          # ~0.1 m swath / 64 px


class TestChipCategories:
    def test_parse_categories(self):
        from sidescanpy.cli.export import _parse_categories

        assert _parse_categories(None) is None
        assert _parse_categories("") is None
        assert _parse_categories("class_b, class_a") == ["class_b", "class_a"]
        assert _parse_categories(["class_a", "class_b"]) == ["class_a", "class_b"]

    def test_chip_rejects_unlisted_category(self, tmp_path):
        import argparse

        from sidescanpy.cli.export import run_chip
        from sidescanpy.core.target import Target
        from sidescanpy.targets.io import write_targets_json

        tj = tmp_path / "t.json"
        write_targets_json(
            [Target(target_id="1", source_format="x", catalog_source="c", position_confidence="verified", category="class_z")], tj
        )
        args = argparse.Namespace(
            targets_json=tj, output=tmp_path / "out", min_confidence="verified",
            ss_h5_dir=tmp_path, categories="class_a,class_b",
        )
        with pytest.raises(SystemExit, match="class_z"):
            run_chip(args)
