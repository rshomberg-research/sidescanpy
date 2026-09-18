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
from dataclasses import replace

import numpy as np
import pytest

from sidescanpy.cli import export
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
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

    def test_targets_json_rejected_with_ground_mode(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([Target(target_id="1", source_format="x", catalog_source="x", latitude=40.0, longitude=-70.0)], targets_json)
        args = self._args(tmp_path, georef_ss_h5_dir, range_mode="ground", targets_json=targets_json)
        assert export.run_waterfall(args) == 1

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
        x, y = export._target_waterfall_xy(t, ping_idx=57)
        assert x == -300.0
        assert y == 57.0

    def test_starboard_channel_is_positive_x(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=1, sample=300)
        x, y = export._target_waterfall_xy(t, ping_idx=57)
        assert x == 300.0
        assert y == 57.0

    def test_missing_channel_defaults_to_port(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=None, sample=300)
        x, _y = export._target_waterfall_xy(t, ping_idx=0)
        assert x == -300.0

    def test_y_is_the_global_ping_index_unmodified(self):
        """The whole point of this helper -- plot_waterfall's own
        ping_offset already bakes the segment's start into the y-axis,
        so a target's file-global ping_idx must NOT be re-based to the
        segment here (that was a real bug caught before this shipped)."""
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=0)
        _x, y = export._target_waterfall_xy(t, ping_idx=4321)
        assert y == 4321.0


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


class TestFlipGeoVertical:
    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_flips_grid_vertically(self):
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        flipped, _ = export._flip_geo_vertical(grid, [], size_px=3)
        np.testing.assert_array_equal(flipped, grid[::-1, :])

    def test_remaps_every_target_window_row(self):
        grid = np.zeros((10, 10))
        targets = [self._co_target(sample=1, window_row=0), self._co_target(sample=4, window_row=9)]
        _, out = export._flip_geo_vertical(grid, targets, size_px=10)
        assert {t.extra_attributes["window_row"] for t in out} == {10, 1}


class TestRunChip:
    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="radiometric",
            size_px=200, swath_size_px=200, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
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
        spy = mocker.spy(export, "extract_geometric_window")
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
