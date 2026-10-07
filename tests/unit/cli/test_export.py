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
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing import run_correction_steps
from sidescanpy.processing.geometric import compute_geometric_ranges
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


@pytest.fixture
def two_file_ss_h5_dir(tmp_path):
    """Two independent 20-ping files, same shape as georef_ss_h5_dir's
    single file -- for testing that waterfall windows/overview span the
    whole survey without restarting at each file boundary."""
    d = tmp_path / "ss_h5"
    d.mkdir()
    write_survey(run_correction_steps(_geo_friendly_survey(), {"tvg", "geometric", "georef"}), d / "survey_a.ss.h5")
    write_survey(run_correction_steps(_geo_friendly_survey(), {"tvg", "geometric", "georef"}), d / "survey_b.ss.h5")
    return d


def _swath_friendly_survey(n_pings=300, n_samples=30):
    """Unlike ``_geo_friendly_survey`` (20 pings x 2000 samples, built for
    geometric/georef range coverage), single-swath/dual-swath's native
    square window needs ``n_samples``/``2*n_samples`` *pings* to fit
    inside the survey (see ``extract_full_channel``'s own docstring) --
    many more pings than samples, the opposite proportion.
    """
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz_start=1_495_000)
    # max slant range at the default 1500m/s, 400kHz fixture sample rate is
    # n_samples * 0.001875m (~0.056m here) -- altitude must be well under
    # that or every sample stays inside the water column (ground_range_m/
    # across_track_m all NaN, exactly _geo_friendly_survey's own docstring
    # warning about its default fixture, scaled down for this one's much
    # smaller n_samples).
    ds["altitude_m"] = ("ping", np.full(n_pings, 0.01, dtype=np.float32))
    ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, n_pings))
    ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, n_pings))
    survey.add_frequency("hz_1495000", ds)
    return survey


@pytest.fixture
def swath_ss_h5_dir(tmp_path):
    d = tmp_path / "ss_h5"
    d.mkdir()
    corrected = run_correction_steps(_swath_friendly_survey(), {"tvg", "geometric", "georef"})
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
        # ping 10's exact lat/lon from _geo_friendly_survey()'s own
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

        raw_args = self._args(tmp_path, georef_ss_h5_dir, value_var="raw", frequency="1495000", output_dir=tmp_path / "raw_out")
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
        """Real field OOM report: this used to call read_survey(path)
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


class TestOrientPixelWindow:
    """Real-data bug: chip/context exports never matched EdgeTech
    Discover's own target thumbnails -- ``orient_for_display`` existed and
    was validated for this, but was never called from the chip export
    path. ``_orient_pixel_window`` is the adapted equivalent (handles
    every co-located target sharing one window, not just a single
    primary target)."""

    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_vertical_flip_always_applies(self):
        window = np.array([[1, 2], [3, 4], [5, 6]])  # 3 rows, 2 cols
        ct = self._co_target(sample=0, window_row=0)
        flipped, (out,) = export._orient_pixel_window(window, [ct], channel_idx=1, style="single-swath")
        np.testing.assert_array_equal(flipped, window[::-1, :])
        assert out.extra_attributes["window_row"] == 2

    def test_port_channel_flips_horizontally_for_single_channel_style(self):
        window = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows, 3 cols
        ct = self._co_target(sample=0, window_row=0)
        _, (out,) = export._orient_pixel_window(window, [ct], channel_idx=0, style="chip")
        assert out.sample == 2

    def test_starboard_channel_does_not_flip_horizontally(self):
        window = np.array([[1, 2, 3], [4, 5, 6]])
        ct = self._co_target(sample=0, window_row=0)
        flipped, (out,) = export._orient_pixel_window(window, [ct], channel_idx=1, style="chip")
        assert out.sample == 0
        np.testing.assert_array_equal(flipped, window[::-1, :])

    def test_dual_swath_never_flips_horizontally_even_for_channel_0(self):
        """dual-swath windows are already port-mirrored by
        _combine_channels_mirrored at combination time -- flipping again
        here would un-mirror them."""
        window = np.array([[1, 2, 3], [4, 5, 6]])
        ct = self._co_target(sample=0, window_row=0)
        _, (out,) = export._orient_pixel_window(window, [ct], channel_idx=0, style="dual-swath")
        assert out.sample == 0

    def test_multiple_co_targets_all_remapped(self):
        window = np.zeros((4, 5))
        cts = [self._co_target(sample=1, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = export._orient_pixel_window(window, cts, channel_idx=0, style="chip")
        assert {(o.sample, o.extra_attributes["window_row"]) for o in out} == {(3, 3), (0, 0)}


def _geo_channel_ds():
    """A single-channel dataset with real across_track_m/along-track
    geometry, for _geo_targets_within_window/_flip_geo_vertical tests --
    mirrors _geo_friendly_survey's altitude/lat/lon overrides but returns
    a bare per-channel Dataset (geometric ranges already computed)."""
    ds = make_minimal_sidescan_dataset(n_pings=20, n_samples=2000, freq_hz_start=1_495_000)
    ds["altitude_m"] = ("ping", np.full(20, 0.5, dtype=np.float32))
    ds["latitude"] = ("ping", np.linspace(40.0000, 40.0010, 20))
    ds["longitude"] = ("ping", np.linspace(-70.0000, -70.0010, 20))
    return compute_geometric_ranges(ds)


class TestGeoTargetsWithinWindow:
    def test_finds_a_co_located_target_and_places_it_locally(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.viz.waterfall import along_track_distance_m

        ds = _geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        co_target = Target(target_id="co", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        candidates = [(co_target, int(ping_idx))]
        found = export._geo_targets_within_window(candidates, ds, 0, along, 16, meta, locate_point)
        assert len(found) == 1
        assert found[0].target_id == "co"
        assert 0 <= found[0].sample < 16
        assert 0 <= found[0].extra_attributes["window_row"] < 16

    def test_target_outside_window_is_excluded(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.viz.waterfall import along_track_distance_m

        ds = _geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        # a target at ping 0 -- far outside a tight 16*0.5m=8m window centered mid-survey
        far_target = Target(target_id="far", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        found = export._geo_targets_within_window([(far_target, 0)], ds, 0, along, 16, meta, locate_point)
        assert found == []

    def test_out_of_range_ping_idx_skipped_not_crashed(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.viz.waterfall import along_track_distance_m

        ds = _geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]
        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        bogus = Target(target_id="bogus", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        # ping_idx way outside this masked slice's own ping axis (lo=0, n_pings=20)
        found = export._geo_targets_within_window([(bogus, 9999)], ds, 0, along, 16, meta, locate_point)
        assert found == []

    def test_along_size_px_bounds_row_independent_of_size_px(self):
        # Real bug regression, 2026-09-29: the row bound used to reuse
        # size_px (the column count) for both axes -- a co-target whose
        # locate_fn-reported row falls inside size_px but outside a
        # smaller real along_size_px must now be excluded. A stub
        # locate_fn pins the row deterministically (row=8), avoiding any
        # dependence on this fixture's own real geometry.
        from sidescanpy.viz.waterfall import along_track_distance_m

        ds = _geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        def stub_locate(*_args, **_kwargs):
            return 8, 2  # row=8: inside size_px=16, outside along_size_px=4

        co_target = Target(target_id="co", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        candidates = [(co_target, int(ping_idx))]
        found_no_limit = export._geo_targets_within_window(candidates, ds, 0, along, 16, {}, stub_locate)
        found_with_limit = export._geo_targets_within_window(candidates, ds, 0, along, 16, {}, stub_locate, along_size_px=4)
        assert len(found_no_limit) == 1  # demonstrates the stale size_px bound over-accepts
        assert found_with_limit == []


class TestFlipGeoVertical:
    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_flips_grid_vertically(self):
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        flipped, _ = export._flip_geo_vertical(grid, [], along_size_px=3)
        np.testing.assert_array_equal(flipped, grid[::-1, :])

    def test_remaps_every_target_window_row(self):
        grid = np.zeros((10, 10))
        targets = [self._co_target(sample=1, window_row=0), self._co_target(sample=4, window_row=9)]
        _, out = export._flip_geo_vertical(grid, targets, along_size_px=10)
        assert {t.extra_attributes["window_row"] for t in out} == {10, 1}

    def test_uses_along_size_px_not_column_count_when_non_square(self):
        # Real bug regression, 2026-09-29: along-track pixel resolution
        # can now differ from across-track, so a grid's row count and
        # column count can genuinely differ -- the flip must mirror
        # against the actual ROW count, not whatever size_px happens to
        # be passed for the (unrelated) column axis.
        grid = np.arange(12).reshape(4, 3)  # 4 rows, 3 cols -- non-square
        targets = [self._co_target(sample=1, window_row=0), self._co_target(sample=1, window_row=4)]
        flipped, out = export._flip_geo_vertical(grid, targets, along_size_px=4)
        np.testing.assert_array_equal(flipped, grid[::-1, :])
        assert {t.extra_attributes["window_row"] for t in out} == {4, 0}


class TestOrientGeoWindow:
    """Real, confirmed finding (2026-09-29): extract_geometric_window's
    ascending-across_track_m column order puts near-range on the RIGHT
    for a port window (negative across_track_m) but on the LEFT for a
    starboard window (positive) -- port and starboard windows were
    genuine mirror images of each other with no correction applied
    anywhere in this codebase, confirmed via real target coordinates
    (0001 port: across_min=-16.28m; 0002 starboard: across_min=+15.55m).
    """

    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_port_gets_horizontal_flip_on_top_of_vertical(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = export._orient_geo_window(grid, [], size_px=2, combine=False, channel_idx=0)
        # vertical flip first (rows reversed), then horizontal (cols reversed)
        expected = grid[::-1, :][:, ::-1]
        np.testing.assert_array_equal(oriented, expected)

    def test_starboard_gets_only_vertical_flip(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = export._orient_geo_window(grid, [], size_px=2, combine=False, channel_idx=1)
        np.testing.assert_array_equal(oriented, grid[::-1, :])

    def test_uses_along_size_px_for_vertical_flip_when_non_square(self):
        # Real bug regression, 2026-09-29: a non-square grid (anisotropic
        # along-track pixel resolution) must flip against its own real
        # row count, not size_px (the across-track/column count).
        grid = np.arange(12).reshape(4, 3)  # 4 rows, 3 cols
        targets = [self._co_target(sample=1, window_row=0)]
        oriented, out = export._orient_geo_window(grid, targets, size_px=3, combine=False, channel_idx=1, along_size_px=4)
        np.testing.assert_array_equal(oriented, grid[::-1, :])
        assert out[0].extra_attributes["window_row"] == 4

    def test_dual_swath_never_gets_horizontal_flip_even_with_channel_idx_zero(self):
        # combine=True always passes channel_idx=None in the real caller
        # (_extract_geo_style_window's geom_channel_idx), but this
        # function is tested directly against combine as the deciding
        # flag, not channel_idx alone -- confirms combine wins.
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = export._orient_geo_window(grid, [], size_px=2, combine=True, channel_idx=None)
        np.testing.assert_array_equal(oriented, grid[::-1, :])

    def test_port_flip_remaps_target_column(self):
        grid = np.zeros((4, 5))
        targets = [self._co_target(sample=0, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = export._orient_geo_window(grid, targets, size_px=4, combine=False, channel_idx=0)
        assert {o.sample for o in out} == {4, 0}

    def test_starboard_does_not_remap_target_column(self):
        grid = np.zeros((4, 5))
        targets = [self._co_target(sample=0, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = export._orient_geo_window(grid, targets, size_px=4, combine=False, channel_idx=1)
        assert {o.sample for o in out} == {0, 4}

    def test_port_near_range_moves_to_column_zero_after_orientation(self):
        """The actual point of this fix, verified directly from real
        window geometry (not just the flip mechanics tested above): a
        real port window's near-range (nadir-ward) edge is at the LAST
        column before orientation (ascending across_track_m puts port's
        least-negative/nearest-zero value last) and must be at column 0
        after -- matching starboard's own, already-correct column-0-is-
        near-range convention (starboard's smallest/nearest-zero value
        is ascending-order-first already, so it needs no flip)."""
        from sidescanpy.export.geometric_window import extract_geometric_window
        from sidescanpy.viz.waterfall import along_track_distance_m

        ds = _geo_channel_ds()
        along = along_track_distance_m(ds)
        across_port = ds["across_track_m"].isel(channel=0).values
        finite_port = np.isfinite(across_port)
        p_ping, p_sample = np.argwhere(finite_port)[len(np.argwhere(finite_port)) // 2]
        port_across_m = float(across_port[p_ping, p_sample])
        size_px = 16
        resolution_m = 0.5
        grid_port, meta_port = extract_geometric_window(
            ds, size_px, resolution_m=resolution_m, center_along_m=float(along[p_ping]),
            center_across_m=port_across_m, channel_idx=0,
        )
        # Pre-flip: column 0's real across_track_m is meta["across_min"]
        # (most negative = far-range for port); the last column is
        # across_min + (size_px-1)*resolution_m (least negative = near-range).
        across_min = meta_port["across_min"]
        pre_flip_col0_across_m = across_min
        pre_flip_last_col_across_m = across_min + (size_px - 1) * resolution_m
        assert abs(pre_flip_last_col_across_m) < abs(pre_flip_col0_across_m), (
            "test setup assumption broken: expected port's near-range (closer to 0) "
            "to be the LAST column before orientation"
        )

        oriented, _ = export._orient_geo_window(grid_port, [], size_px=size_px, combine=False, channel_idx=0)
        # After the horizontal flip, oriented column 0 == pre-flip (vertically-flipped) last column.
        np.testing.assert_array_equal(oriented[:, 0], grid_port[::-1, :][:, -1])


class TestCorrelateTarget:
    """Real field report: lots of "no ss_h5 ping within 25.0m of its
    recorded position" warnings, and some targets missing from chip
    output entirely -- traced to _correlate_target requiring a GPS
    lat/lon match for *every* target, silently dropping any target with
    no recorded GPS position (or one that doesn't match closely enough)
    even when its real, already-known ping position (e.g. from a pixel-
    based source catalog like vendor-format's targets.csv) was sitting
    right there on the object. These test the fallback directly against
    a hand-built index, rather than only indirectly through run_chip."""

    def _index(self, *, lat, lon, n_pings, freq_key="hz_1495000", sample_interval_m=0.05, n_samples=1000, ping_start=0):
        return {
            Path("a.ss.h5"): {
                freq_key: {
                    "lat": np.asarray(lat, dtype=float),
                    "lon": np.asarray(lon, dtype=float),
                    "sample_interval_m": sample_interval_m,
                    "n_samples": n_samples,
                    "n_pings": n_pings,
                    "ping_start": ping_start,
                }
            }
        }

    def test_gps_match_within_range_unchanged(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=float(lat[10]), longitude=float(lon[10]))

        result = export._correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 10
        assert dist_m < 1.0

    def test_no_gps_falls_back_to_known_ping(self):
        """The exact reported bug: no recorded GPS position at all, but a
        real, already-known ping -- must now match instead of being
        silently dropped."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        result = export._correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 7
        assert dist_m == 0.0

    def test_gps_too_far_falls_back_to_known_ping(self):
        """Recorded GPS position exists but doesn't land within
        max_dist_m of the survey's own track (e.g. a less precise/
        independent positioning system) -- ping fallback still rescues
        it rather than dropping it."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=41.0, longitude=-71.0, ping=7)

        result = export._correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        assert result[2] == 7
        assert result[3] == 0.0

    def test_ping_out_of_range_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=500)

        assert export._correlate_target(t, index, max_dist_m=25.0) is None

    def test_no_position_at_all_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=None)

        assert export._correlate_target(t, index, max_dist_m=25.0) is None

    def test_ambiguous_ping_across_multiple_files_returns_none_not_a_guess(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        index[Path("b.ss.h5")] = {
            "hz_1495000": {
                "lat": lat.copy(), "lon": lon.copy(), "sample_interval_m": 0.05, "n_samples": 1000,
                "n_pings": 20, "ping_start": 0,
            },
        }
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        assert export._correlate_target(t, index, max_dist_m=25.0) is None

    def test_ping_fallback_uses_global_ping_start_not_local_zero_based_index(self):
        """Real bug, found 2026-09-27 from a downstream consumer's
        multi-part-converted mission: every target failed to correlate
        (all "no ss_h5 ping within Nm" warnings) once a survey's own
        ping/ping_number coordinate stopped starting at 0 for every file
        -- true for ANY directory of several globally-renumbered .ss.h5
        files (segment's own sectionN.ss.h5 outputs already do this,
        not just the new per-source-file conversion output that
        triggered this report). The old `0 <= target.ping < n_pings`
        check silently assumed every file's own ping axis is locally
        zero-based, which was only ever true for a single, lone file."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        # This file covers mission-global pings [100, 120) -- NOT [0, 20).
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=100)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=107)

        result = export._correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 7  # 107 - ping_start(100) -- the correct LOCAL array index
        assert dist_m == 0.0

    def test_ping_fallback_global_ping_before_this_files_range_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=100)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        assert export._correlate_target(t, index, max_dist_m=25.0) is None

    def test_ping_fallback_across_two_globally_offset_files_picks_the_right_one(self):
        """The real multi-part scenario: two files, each internally
        zero-based-*looking* by n_pings alone, but covering disjoint
        global ping ranges -- exactly one must claim a given global
        ping, at the right LOCAL index within it."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=0)
        index[Path("b.ss.h5")] = {
            "hz_1495000": {
                "lat": lat.copy(), "lon": lon.copy(), "sample_interval_m": 0.05, "n_samples": 1000,
                "n_pings": 20, "ping_start": 20,
            },
        }
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=25)

        result = export._correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert path == Path("b.ss.h5")
        assert ping_idx == 5  # 25 - ping_start(20)


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
            value_var="raw", output=out_raw, seed=0,
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
        for value_var in ("raw", "db"):
            out = tmp_path / f"out_{value_var}"
            args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric", value_var=value_var, output=out, seed=0)
            assert export.run_chip(args) == 0
            imgs[value_var] = np.asarray(Image.open(out / "chip" / "target_7.png"))
        assert imgs["db"].shape == imgs["raw"].shape
        assert not np.array_equal(imgs["db"], imgs["raw"])

    def test_value_var_raw_has_no_effect_and_warns_under_radiometric_render_level(self, tmp_path, georef_ss_h5_dir, caplog):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", value_var="raw")
        assert export.run_chip(args) == 0
        assert any("--value-var=raw has no effect" in r.message for r in caplog.records)

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
        """Real bug found running this against real project-b data: this
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
        spy = mocker.spy(export, "contact_sheet")
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
        spy = mocker.spy(export, "render_target_context")
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
        spy = mocker.spy(export, "extract_geometric_window_griddata")
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
        spy = mocker.spy(export, "extract_geometric_window_griddata")
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
        spy = mocker.spy(export, "extract_geometric_window_griddata")
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
        spy = mocker.spy(export, "extract_georef_window")
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
        spy = mocker.spy(export, "extract_georef_window")
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
        this fixture only if TVG is a no-op there -- run_correction_steps
        with 'tvg' genuinely modifies backscatter_intensity, so raw and
        radiometric renders of the same target must differ."""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        raw_args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="raw", styles=["chip"], output=tmp_path / "raw_out")
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
        for level in ("raw", "db"):
            args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level=level, styles=["chip"], output=tmp_path / f"{level}_out")
            assert export.run_chip(args) == 0
            imgs[level] = np.array(Image.open(tmp_path / f"{level}_out" / "chip" / "target_7.png"))
        assert not np.array_equal(imgs["db"], imgs["raw"])

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
        """Real field report: chip/COCO output had every target's box
        at the same size, regardless of each target's own real, measured
        extent (e.g. vendor-format's targets.csv 4-corner columns). "chip"
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

        sheet = mocker.spy(export, "contact_sheet")
        context = mocker.spy(export, "render_target_context")
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
        downstream-project's own batch script actually uses
        (--render-level geometric), so this path mattered more than the
        raw-pixel one for the report that found this. Exact pixel values
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
        (_match_frequency then picks a frequency arbitrarily) can be
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
        spy = mocker.spy(export, spied_fn)
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, resample_method=resample_method)
        assert export.run_chip(args) == 0
        assert spy.call_count == 1

    def test_omitting_resample_method_defaults_to_griddata(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(export, "extract_geometric_window_griddata")
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
    """Geometric/georef single-swath/dual-swath block-average the source
    neighborhood before resampling (_load_swath_neighborhood) -- the
    fix for ~3GB-per-target loads plus the resample's own working set
    on real survey-a data."""

    def test_factors_keep_two_cells_per_output_pixel(self, georef_ss_h5_dir):
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        mps = export._sample_interval_m(ds)
        half_span, fp, fs = export._geo_decimation(ds, 10, "dual-swath", size_px=64, jitter_px=0, resolution_m=0.1)
        assert fs == int(0.1 / (export._SWATH_OVERSAMPLE * mps))
        assert fs > 1
        assert fp >= 1

    def test_neighborhood_covers_the_window_in_meters(self, georef_ss_h5_dir):
        """Sized from size_px * resolution_m along-track, not from
        size_px pings -- a pings-sized neighborhood left real geometric
        chips short of data at top/bottom (griddata nearest-fill streaks)."""
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        spacing = export._neighborhood_ping_spacing_m(ds, 0, ds.sizes["ping"])
        half_span, _fp, _fs = export._geo_decimation(ds, 10, "chip", size_px=100, jitter_px=10, resolution_m=spacing)
        assert half_span * spacing >= (100 / 2 + 10) * spacing

    def test_loaded_neighborhood_is_averaged_and_rescaled(self, georef_ss_h5_dir):
        ds = read_survey(next(georef_ss_h5_dir.glob("*.ss.h5"))).get_frequency("hz_1495000").dataset
        out = export._load_swath_neighborhood(ds, 0, ds.sizes["ping"], 1, 10, with_db=True)
        assert out.sizes["sample"] == ds.sizes["sample"] // 10
        assert out.sizes["ping"] == ds.sizes["ping"]
        assert "backscatter_db" in out
        assert "slant_range_m" not in out
        assert export._sample_interval_m(out) == pytest.approx(10 * export._sample_interval_m(ds))

    def test_annotation_matches_full_resolution_resample(self, tmp_path, georef_ss_h5_dir, monkeypatch):
        targets_json = tmp_path / "targets.json"
        write_targets_json([TestRunChip()._matched_target()], targets_json)

        def run(out, oversample):
            monkeypatch.setattr(export, "_SWATH_OVERSAMPLE", oversample)
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

    def test_slant_across_axis_neighborhood_uses_signed_slant_range(self, tmp_path, swath_ss_h5_dir):
        """--across-axis slant swaps the loaded neighborhood's across_track_m
        for +/-slant_range_m (port negative), which every downstream
        geometric extraction/co-target lookup reads -- and keeps the
        water-column samples ground projection drops."""
        survey = read_survey(next(swath_ss_h5_dir.glob("*.ss.h5")))
        ds_full = survey.get_frequency("hz_1495000").dataset
        t = self._target()
        views = {
            axis: export._load_geo_neighborhood(
                ds_full, t, 150, {1: [(t, 150)]}, ["chip"], size_px=64, jitter_px=0,
                resolution_m=0.002, with_db=False, across_axis=axis,
            )
            for axis in ("ground", "slant")
        }
        slant = views["slant"].ds
        np.testing.assert_allclose(slant["across_track_m"].isel(channel=1).values, slant["slant_range_m"].isel(channel=1).values)
        np.testing.assert_allclose(slant["across_track_m"].isel(channel=0).values, -slant["slant_range_m"].isel(channel=0).values)
        ground_across = views["ground"].ds["across_track_m"].values
        assert np.isnan(ground_across).any() and np.isfinite(slant["across_track_m"].values).all()
        finite = np.isfinite(ground_across)
        assert (np.abs(slant["across_track_m"].values[finite]) >= np.abs(ground_across[finite]) - 1e-9).all()

    def test_slant_across_axis_runs_end_to_end(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, across_axis="slant", resolution_m=0.002, n_augment=2, random_flip=True)
        assert export.run_chip(args) == 0
        assert len(list((tmp_path / "out" / "chip").glob("target_7_a*.png"))) == 2

    def test_across_values_slant_signs_port_negative(self):
        ds = xr.Dataset({"slant_range_m": (("ping", "channel", "sample"), np.ones((2, 2, 3)))})
        v = export._across_values(ds, "slant").values
        assert (v[:, 0] < 0).all() and (v[:, 1] > 0).all()
        with pytest.raises(ValueError):
            export._across_values(ds, "sideways")

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


class TestSwathFill:
    """--swath-fill: swath frame sized to where the data is, square pixels."""

    def _arrays(self, n_pings=50, n_samples=100, port_reach=40.0, stbd_reach=60.0, near=2.0):
        # across: port negative, starboard positive; reach in meters
        frac = np.linspace(0.0, 1.0, n_samples)
        across = np.empty((n_pings, 2, n_samples))
        across[:, 0, :] = -(near + frac * (port_reach - near))
        across[:, 1, :] = near + frac * (stbd_reach - near)
        values = np.ones_like(across)
        return across, values

    def test_dual_swath_spans_port_far_to_starboard_far(self):
        across, values = self._arrays()
        lo, hi = export._swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert lo == pytest.approx(-40.0) and hi == pytest.approx(60.0)

    def test_short_pings_crop_long_ones_not_pad(self):
        across, values = self._arrays()
        values[:45, 1, 80:] = np.nan  # 90% of pings stop short on starboard
        _lo, hi = export._swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert hi < 60.0 * 0.82

    def test_one_sided_recording_mirrors_other_side(self):
        across, values = self._arrays()
        values[:, 0, :] = np.nan
        lo, hi = export._swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert (lo, hi) == pytest.approx((-60.0, 60.0))

    def test_single_swath_near_to_far_per_channel(self):
        across, values = self._arrays()
        assert export._swath_fill_extent(across, values, combine=False, channel_idx=1) == pytest.approx((2.0, 60.0))
        assert export._swath_fill_extent(across, values, combine=False, channel_idx=0) == pytest.approx((-40.0, -2.0))

    def test_no_data_returns_none(self):
        across, values = self._arrays()
        values[:] = np.nan
        assert export._swath_fill_extent(across, values, combine=True, channel_idx=None) is None

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
        spy = mocker.spy(export, "extract_geometric_window_griddata")
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
        from sidescanpy.targets.core import Target
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
