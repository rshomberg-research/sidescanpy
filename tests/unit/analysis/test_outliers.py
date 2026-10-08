"""
Unit tests for sidescanpy.analysis.outliers.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest

from sidescanpy.analysis.outliers import compare_outlier_rejection
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestCompareOutlierRejection:
    def test_before_after_metrics_reflect_rejection(self):
        n_pings, n_samples = 30, 300
        rng = np.random.default_rng(4)
        # Small baseline noise is required: with a perfectly constant
        # baseline, non-spike bins have MAD=0 and reject_range_outliers
        # deliberately leaves those bins unfiltered (see
        # TestRejectRangeOutliers.test_degenerate_zero_mad_bin_left_unfiltered).
        bs = (5.0 + rng.normal(0, 0.1, size=(n_pings, 2, n_samples))).astype(np.float32)
        spike_idx = rng.choice(bs.size, size=50, replace=False)
        bs.flat[spike_idx] += 500.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = compare_outlier_rejection(ds, bin_width_m=5.0)

        assert result.n_rejected > 0
        # Not every spike is guaranteed rejected (edge bins, spike
        # clustering) -- check the bulk of the population was cleaned
        # rather than requiring a specific residual max.
        assert result.before.maximum > 400.0
        assert result.before.pct_negative == result.after.pct_negative == 0.0
        assert result.n_rejected >= 40  # most of the 50 injected spikes
        assert result.filtered_values.shape == bs.shape

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="ground_range_m"):
            compare_outlier_rejection(ds)
