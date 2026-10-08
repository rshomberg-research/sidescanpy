"""
Unit tests for sidescanpy.processing.outliers.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np

from sidescanpy.processing.outliers import mask_bright_ping_rows, reject_range_outliers


class TestRejectRangeOutliers:
    def test_flags_injected_spikes_leaves_normal_values(self):
        rng = np.random.default_rng(0)
        n = 2000
        ranges = rng.uniform(0, 100, size=n)
        values = 10.0 + rng.normal(0, 0.5, size=n)  # flat "sandy bottom" signal

        spike_idx = rng.choice(n, size=20, replace=False)
        values[spike_idx] += 500.0  # far outside the local population

        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5)

        assert np.all(np.isnan(cleaned[spike_idx]))
        non_spike = np.setdiff1d(np.arange(n), spike_idx)
        # Overwhelming majority of the clean population should survive --
        # a small amount of edge-case rejection near bin boundaries is
        # expected and fine, a mass rejection would not be.
        assert np.isfinite(cleaned[non_spike]).mean() > 0.95

    def test_small_bins_left_unfiltered(self):
        ranges = np.array([1.0, 1.0, 1.0])  # below min_bin_count
        values = np.array([1.0, 2.0, 1000.0])
        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5, min_bin_count=5)
        np.testing.assert_array_equal(cleaned, values)

    def test_degenerate_zero_mad_bin_left_unfiltered(self):
        ranges = np.full(10, 5.0)
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0])
        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5, min_bin_count=5)
        # Median of the mostly-identical bin has MAD=0 -- no scale to
        # reject against, so nothing gets flagged even though 1000.0
        # is visibly anomalous. Documents the known limitation rather
        # than silently disagreeing with it.
        np.testing.assert_array_equal(cleaned, values)

    def test_shape_preserved(self):
        rng = np.random.default_rng(1)
        values = rng.uniform(0, 10, size=(4, 2, 50))
        ranges = rng.uniform(0, 60, size=(4, 2, 50))
        cleaned = reject_range_outliers(values, ranges, bin_width_m=5.0)
        assert cleaned.shape == values.shape


class TestMaskBrightPingRows:
    def _array(self):
        rng = np.random.default_rng(0)
        return 10.0 + rng.normal(0, 0.5, size=(40, 30))

    def test_masks_a_gain_jump_row_only(self):
        a = self._array()
        a[17] *= 5.0
        out = mask_bright_ping_rows(a)
        assert np.isnan(out[17]).all()
        assert np.isfinite(np.delete(out, 17, axis=0)).all()
        assert np.isfinite(a).all()  # input untouched

    def test_returns_input_when_nothing_to_mask(self):
        a = self._array()
        assert mask_bright_ping_rows(a) is a

    def test_too_few_rows_or_zero_mad_is_a_no_op(self):
        few = self._array()[:4]
        few[0] *= 10
        assert mask_bright_ping_rows(few) is few
        flat = np.ones((20, 5))
        flat[3] = 100.0
        assert mask_bright_ping_rows(flat) is flat

    def test_lower_threshold_masks_more(self):
        a = self._array()
        a[5] += 0.8  # modified z-score of row 5's median is ~7
        assert not np.isnan(mask_bright_ping_rows(a, mad_k=20.0)[5]).any()
        assert np.isnan(mask_bright_ping_rows(a, mad_k=3.0)[5]).all()
