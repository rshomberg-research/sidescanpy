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

import numpy as np
import pytest
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
targets_waterfall = importlib.import_module("sidescanpy.targets.waterfall")
targets_correlate = importlib.import_module("sidescanpy.targets.correlate")


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
        spy = mocker.spy(targets_waterfall, "target_waterfall_position")

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
        spy = mocker.spy(targets_correlate, "project_target_to_frequency")

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

        raw_args = self._args(tmp_path, georef_ss_h5_dir, value_var="uncorrected", frequency="1495000", output_dir=tmp_path / "raw_out")
        rad_args = self._args(tmp_path, georef_ss_h5_dir, value_var="radiometric", frequency="1495000", output_dir=tmp_path / "rad_out")
        assert export.run_waterfall(raw_args) == 0
        assert export.run_waterfall(rad_args) == 0

        raw_img = np.array(Image.open(tmp_path / "raw_out" / "hz_1495000" / "segment_001.png"))
        rad_img = np.array(Image.open(tmp_path / "rad_out" / "hz_1495000" / "segment_001.png"))
        assert not np.array_equal(raw_img, rad_img)


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


class TestChipPingRanges:
    def test_ping_ranges_from_file(self, tmp_path):
        ranges = tmp_path / "ranges.json"
        ranges.write_text(json.dumps({"a.ss.h5": [[1, 5], [9, 12]]}))
        assert export._load_ping_ranges(ranges) == {"a.ss.h5": [(1, 5), (9, 12)]}
        assert export._load_ping_ranges(str(ranges)) == {"a.ss.h5": [(1, 5), (9, 12)]}
        assert export._load_ping_ranges({"b": [[0, 1]]}) == {"b": [(0, 1)]}
        assert export._load_ping_ranges(None) is None


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
