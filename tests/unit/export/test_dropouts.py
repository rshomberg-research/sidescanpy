"""
Tests for dropout masking in export windows: samples recorded as exactly
zero become NaN before resampling, also after TVG has moved them off the
dB floor.

2026 rshomberg@uri.edu
"""

import numpy as np

from sidescanpy.export.windows.levels import mask_dropouts, mask_nadir_gap_with_uncorrected
from sidescanpy.export.windows.neighborhood import load_swath_neighborhood
from sidescanpy.processing import apply_tvg
from tests.unit.synthetic import make_raw_dataset


def _channel(tvg: bool):
    arr = np.full((40, 2, 400), 0.05, dtype=np.float32)
    arr[..., 60:] = 10.0
    arr[10, 1, 200] = 0.0      # dropouts
    arr[25, 0, 300] = 0.0
    ds = make_raw_dataset(n_pings=40, n_samples=400, backscatter=arr)
    return apply_tvg(ds) if tvg else ds


class TestMaskDropouts:
    def test_zero_samples_masked_without_tvg(self):
        out = mask_dropouts(mask_nadir_gap_with_uncorrected(_channel(tvg=False)))
        bs = out["backscatter_intensity"].values
        assert np.isnan(bs[10, 1, 200]) and np.isnan(bs[25, 0, 300])
        assert np.isfinite(bs[10, 1, 201])

    def test_zero_samples_masked_after_tvg(self):
        ds = _channel(tvg=True)
        assert ds["backscatter_intensity"].values[10, 1, 200] != -120.0  # TVG moved it off the floor
        out = mask_dropouts(mask_nadir_gap_with_uncorrected(ds))
        assert np.isnan(out["backscatter_intensity"].values[10, 1, 200])
        assert np.isnan(out["uncorrected"].values[25, 0, 300])

    def test_unchanged_when_no_dropouts(self):
        ds = mask_nadir_gap_with_uncorrected(_channel(tvg=False).assign(
            backscatter_intensity=lambda d: d["backscatter_intensity"] + 1.0))
        assert mask_dropouts(ds) is ds

    def test_neighborhood_loader_masks_dropouts(self):
        out = load_swath_neighborhood(_channel(tvg=True), 0, 40, 1, 1, with_db=False)
        assert np.isnan(out["backscatter_intensity"].values[10, 1, 200])
