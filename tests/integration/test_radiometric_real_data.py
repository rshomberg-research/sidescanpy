"""Integration tests for radiometric correction on real data."""

import pytest
import xarray as xr
import numpy as np
from sidescanpy.processing.radiometric import apply_tvg


@pytest.mark.integration
@pytest.mark.slow
def test_radiometric_on_real_data():
    """Test TVG on actual survey data."""
    # Load real data
    ds_raw = xr.open_dataset(
        "data/survey-a/sshf/20251015170335.ss.h5",
        group="frequencies/hz_1495000",
        engine="h5netcdf"
    )
    
    # Apply TVG
    ds_tvg = apply_tvg(ds_raw)
    
    # Assertions
    assert ds_tvg.attrs["processing_level"] == "radiometric"
    assert not np.isnan(ds_tvg["backscatter_intensity"]).all()
    assert np.isfinite(ds_tvg["backscatter_intensity"]).any()
    
    # Values should be in reasonable dB range (-120 to +30)
    valid_vals = ds_tvg["backscatter_intensity"].values
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    assert valid_vals.min() > -150
    assert valid_vals.max() < 50


@pytest.mark.integration
@pytest.mark.slow
def test_radiometric_increases_far_range():
    """TVG should increase values at far range."""
    ds_raw = xr.open_dataset(
        "data/survey-a/sshf/20251015170335.ss.h5",
        group="frequencies/hz_1495000",
        engine="h5netcdf"
    )
    
    ds_tvg = apply_tvg(ds_raw)
    
    # Get a single ping
    raw_ping = ds_raw["backscatter_intensity"].isel(ping=0, channel=0).values
    tvg_ping = ds_tvg["backscatter_intensity"].isel(ping=0, channel=0).values
    
    # Near range: raw and corrected should be similar
    # Far range: corrected should be higher (TVG gain added)
    
    # Skip first 10 samples (too close to transducer)
    mid = len(tvg_ping) // 2
    far_start = mid + 100
    
    if far_start < len(tvg_ping):
        near_diff = np.median(tvg_ping[10:mid] - raw_ping[10:mid])
        far_diff = np.median(tvg_ping[far_start:] - raw_ping[far_start:])
        
        # Far range should have more gain than near range
        assert far_diff > near_diff
