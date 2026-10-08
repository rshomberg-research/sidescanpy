"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.windows import (
    WindowExtent,
    combine_channels_mirrored,
    extract_chip,
    extract_full_channel,
    extract_pixel_style_window,
    place_targets_in_file,
    rescale_target_position,
    rescale_targets,
    resize_to,
    sample_negative_window,
    target_combined_col,
    targets_within_window,
    with_bbox_size_px,
)


def _target(target_id, row, col):
    return Target(
        target_id=target_id,
        source_format="test",
        catalog_source="test.csv",
        sample=col,
        extra_attributes={"window_row": row},
    )


class TestExtractChip:
    def test_exact_center_no_jitter(self):
        array = np.arange(100).reshape(10, 10)
        window, extent = extract_chip(array, center_row=5, center_col=5, size_px=4)
        assert window.shape == (4, 4)
        assert extent == WindowExtent(row_start=3, col_start=3, n_rows=4, n_cols=4)

    def test_shifts_inward_near_edge_instead_of_padding(self):
        array = np.arange(100).reshape(10, 10).astype(float)
        window, extent = extract_chip(array, center_row=0, center_col=0, size_px=4)
        assert window.shape == (4, 4)
        assert extent.row_start == 0
        assert extent.col_start == 0
        # shifted to the real slice, not zero-padded around the requested center
        assert np.array_equal(window, array[0:4, 0:4])

    def test_jitter_zero_is_deterministic(self):
        array = np.arange(400).reshape(20, 20)
        rng = np.random.default_rng(1)
        _window, extent1 = extract_chip(array, 10, 10, 6, jitter_px=0, rng=rng)
        _window, extent2 = extract_chip(array, 10, 10, 6, jitter_px=0, rng=rng)
        assert extent1 == extent2

    def test_source_smaller_than_size_returns_full_extent_not_padded(self):
        # confirmed against real data: a source catalog's own decoded
        # window can have a hard ping-count ceiling under the requested
        # chip size — the result should be the full (non-square) extent,
        # not zero-padded to size_px.
        array = np.arange(20 * 50).reshape(20, 50).astype(float)
        window, extent = extract_chip(array, center_row=10, center_col=25, size_px=400)
        assert window.shape == (20, 50)
        assert extent.n_rows == 20
        assert np.array_equal(window, array)

    def test_jitter_stays_within_bounds(self):
        array = np.arange(400).reshape(20, 20)
        rng = np.random.default_rng(2)
        for _ in range(20):
            window, extent = extract_chip(array, 10, 10, 6, jitter_px=5, rng=rng)
            assert window.shape == (6, 6)
            assert 0 <= extent.row_start <= 20 - 6
            assert 0 <= extent.col_start <= 20 - 6


class TestExtractFullChannel:
    def test_square_native_window_uses_sample_count(self):
        # 300 pings x 50 samples: a non-square native array
        array = np.arange(300 * 50).reshape(300, 50).astype(float)
        window, extent = extract_full_channel(array, square=True)
        assert extent.n_rows == 50  # native window height == sample count, not a fixed ping count
        assert extent.n_cols == 50
        assert window.shape == (50, 50)  # native resolution -- not resized here

    def test_non_square_uses_fixed_ping_count(self):
        array = np.arange(300 * 50).reshape(300, 50).astype(float)
        _window, extent = extract_full_channel(array, square=False, fixed_ping_count=120)
        assert extent.n_rows == 120
        assert extent.n_cols == 50

    def test_survey_shorter_than_native_window_raises(self):
        # 10 pings but 50 samples -> square mode needs 50 pings, doesn't have them
        array = np.arange(10 * 50).reshape(10, 50).astype(float)
        with pytest.raises(ValueError):
            extract_full_channel(array, square=True)


class TestResizeTo:
    def test_resizes_to_requested_square_size(self):
        image = np.arange(100 * 60, dtype=np.uint8).reshape(100, 60) % 255
        resized = resize_to(image, 32)
        assert resized.shape == (32, 32)

    def test_is_the_last_step_after_a_photometric_transform(self):
        # regression guard for the washed-out-full-channel bug: stretching
        # a low-variance, already-downsampled array measures percentiles
        # against a blurred distribution and loses real contrast -- the
        # fix is stretch-then-resize, not resize-then-stretch. This proves
        # resize_to itself is a pure resize (no renormalization), so a
        # caller doing stretch-then-resize actually gets what it stretched.
        from sidescanpy.export.image import to_uint8

        rng = np.random.default_rng(0)
        native = rng.integers(0, 65535, size=(200, 200), dtype=np.uint16)
        img_8bit, _meta = to_uint8(native, gamma=2.5)
        resized = resize_to(img_8bit, 20)
        # a real stretch should use much of the 0-255 range; resize_to must
        # not have re-compressed it back toward a narrow band
        assert resized.max() - resized.min() > 100


class TestSampleNegativeWindow:
    def test_respects_min_distance(self):
        rng = np.random.default_rng(3)
        targets = [(50.0, 50.0)]
        for _ in range(20):
            row, col = sample_negative_window(100, 100, targets, size_px=10, min_distance_px=20, rng=rng)
            assert np.hypot(row - 50, col - 50) >= 20

    def test_raises_when_infeasible(self):
        rng = np.random.default_rng(4)
        # a target dead-center of a small array with a huge exclusion radius
        # leaves no valid location anywhere in bounds
        targets = [(5.0, 5.0)]
        with pytest.raises(RuntimeError):
            sample_negative_window(10, 10, targets, size_px=4, min_distance_px=100, rng=rng, max_attempts=10)


class TestTargetsWithinWindow:
    def test_includes_targets_inside_extent_and_translates_coords(self):
        extent = WindowExtent(row_start=10, col_start=20, n_rows=50, n_cols=50)
        inside = _target("in", row=30, col=40)
        outside = _target("out", row=5, col=5)
        result = targets_within_window([inside, outside], extent)
        assert len(result) == 1
        assert result[0].target_id == "in"
        assert result[0].sample == 40 - 20
        assert result[0].extra_attributes["window_row"] == 30 - 10

    def test_skips_targets_missing_position(self):
        extent = WindowExtent(row_start=0, col_start=0, n_rows=10, n_cols=10)
        no_pos = Target(target_id="no_pos", source_format="test", catalog_source="test.csv")
        assert targets_within_window([no_pos], extent) == []


class TestRescaleTargetPosition:
    def test_scales_proportionally(self):
        row, col = rescale_target_position(50, 100, native_shape=(200, 200), output_shape=(50, 50))
        assert row == pytest.approx(12.5)
        assert col == pytest.approx(25.0)


class TestCombinedChannels:
    def test_port_mirrored_then_starboard(self):
        port = np.array([[1, 2, 3]])
        stbd = np.array([[4, 5, 6]])
        np.testing.assert_array_equal(combine_channels_mirrored(port, stbd), [[3, 2, 1, 4, 5, 6]])

    @pytest.mark.parametrize("channel, sample", [(0, 0), (0, 2), (1, 0), (1, 2)])
    def test_target_combined_col_points_at_the_same_value(self, channel, sample):
        port = np.array([[10, 11, 12]])
        stbd = np.array([[20, 21, 22]])
        combined = combine_channels_mirrored(port, stbd)
        col = target_combined_col(sample, 3, channel)
        assert combined[0, col] == (port if channel == 0 else stbd)[0, sample]

    def test_place_targets_in_file(self):
        t0 = Target(target_id="a", source_format="x", catalog_source="x", channel=0, sample=1)
        t1 = Target(target_id="b", source_format="x", catalog_source="x", channel=1, sample=2)
        by_channel, combined = place_targets_in_file({0: [(t0, 7)], 1: [(t1, 9)]}, n_samples_single=5)
        assert by_channel[0][0].extra_attributes["window_row"] == 7
        assert by_channel[1][0].sample == 2
        assert {(c.target_id, c.sample, c.extra_attributes["window_row"]) for c in combined} == {("a", 3, 7), ("b", 7, 9)}


class TestTargetAnnotationHelpers:
    def test_with_bbox_size_px_scales_measured_extent(self):
        t = Target(target_id="a", source_format="x", catalog_source="x", bbox=(0.0, 0.0, 4.0, 10.0))
        assert with_bbox_size_px(t, 2.0, 0.5).extra_attributes["bbox_size_px"] == (8.0, 5.0)

    def test_with_bbox_size_px_no_bbox_is_unchanged(self):
        t = Target(target_id="a", source_format="x", catalog_source="x")
        assert with_bbox_size_px(t, 2.0, 2.0) is t

    def test_rescale_targets(self):
        t = _target("a", row=10, col=20)
        (out,) = rescale_targets([t], (100, 200), (50, 50))
        assert (out.extra_attributes["window_row"], out.sample) == (5.0, 5.0)


def _neighborhood(n_pings=60, n_samples=20):
    values = np.zeros((n_pings, 2, n_samples))
    values[:, 0, :] = np.arange(n_samples)  # port value = sample index
    values[:, 1, :] = 100 + np.arange(n_samples)  # starboard = 100 + sample index
    return xr.Dataset({"backscatter_intensity": (("ping", "channel", "sample"), values)})


class TestExtractPixelStyleWindow:
    def _call(self, style, *, channel=1, sample=5, placed=None, require=True, size_px=8):
        masked = _neighborhood()
        placed = placed or {}
        by_channel, combined = place_targets_in_file(placed, masked.sizes["sample"])
        return extract_pixel_style_window(
            masked, center_sample=sample, channel_idx=channel, local_ping=30, ping_offset=100, style=style,
            level="radiometric", size_px=size_px, jitter_px=0, rng=np.random.default_rng(0),
            placed_by_channel=by_channel, placed_combined=combined, require_targets=require,
        )

    def test_requires_a_target_by_default(self):
        assert self._call("chip") is None

    def test_background_window_has_no_targets_and_a_file_extent(self):
        out = self._call("chip", require=False)
        assert out.targets == []
        assert out.extent == WindowExtent(row_start=126, col_start=1, n_rows=8, n_cols=8)
        assert out.image.shape == (8, 8)

    def test_chip_keeps_target_and_orients(self):
        t = Target(target_id="t", source_format="x", catalog_source="x", channel=1, sample=5)
        out = self._call("chip", placed={1: [(t, 130)]})
        (ct,) = out.targets
        # row 130 - 126 = 4, flipped vertically in an 8-row window -> 3
        assert (ct.sample, ct.extra_attributes["window_row"]) == (4, 3)
        assert out.image[3, 4] == 105

    def test_port_chip_is_mirrored(self):
        t = Target(target_id="t", source_format="x", catalog_source="x", channel=0, sample=5)
        out = self._call("chip", channel=0, placed={0: [(t, 130)]})
        (ct,) = out.targets
        assert out.image[int(ct.extra_attributes["window_row"]), int(ct.sample)] == 5

    def test_dual_swath_is_square_resized_and_spans_both_channels(self):
        t = Target(target_id="t", source_format="x", catalog_source="x", channel=1, sample=5)
        out = self._call("dual-swath", placed={1: [(t, 130)]}, size_px=16)
        assert out.image.shape == (16, 16)
        assert out.extent.n_cols == 40 and out.extent.n_rows == 40
        assert out.image[:, 0].mean() < 50 < out.image[:, -1].mean()

    def test_missing_level_variable_returns_none(self):
        masked = _neighborhood()
        out = extract_pixel_style_window(
            masked, center_sample=5, channel_idx=1, local_ping=30, ping_offset=0, style="chip", level="db",
            size_px=8, jitter_px=0, rng=np.random.default_rng(0), placed_by_channel={}, placed_combined=[],
            require_targets=False,
        )
        assert out is None
