"""
Regression tests for the absolute ``sample`` coordinate: slicing leading
samples must not change slant range, ground range or TVG.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import apply_tvg, apply_tvg_to_amplitude
from sidescanpy.processing.sample_axis import sample_index
from tests.conftest import make_minimal_sidescan_dataset

pytestmark = pytest.mark.unit

N_SAMPLES = 100
CUT = 30


def _ds(*, with_coord: bool = True) -> xr.Dataset:
    ds = make_minimal_sidescan_dataset(n_pings=4, n_samples=N_SAMPLES)
    ds["backscatter_intensity"] = ds["backscatter_intensity"] + 1.0  # strictly positive
    ds["altitude_m"] = ds["altitude_m"] * 0 + 5.0
    ds.attrs["sample_rate_hz"] = 1000.0
    if with_coord:
        ds = ds.assign_coords(sample=np.arange(N_SAMPLES))
    else:
        ds = ds.drop_vars("sample", errors="ignore")
    return ds


def test_sample_index_uses_coordinate():
    ds = _ds().isel(sample=slice(CUT, None))
    idx = sample_index(ds)
    assert idx.dtype == np.float64
    assert idx[0] == CUT


def test_sample_index_fallback_without_coordinate():
    idx = sample_index(_ds(with_coord=False))
    np.testing.assert_array_equal(idx, np.arange(N_SAMPLES))


def test_geometric_ranges_commute_with_leading_slice():
    ds = _ds()
    full = compute_geometric_ranges(ds).isel(sample=slice(CUT, None))
    cropped = compute_geometric_ranges(ds.isel(sample=slice(CUT, None)))
    for var in ("slant_range_m", "ground_range_m", "across_track_m"):
        np.testing.assert_array_equal(full[var].values, cropped[var].values)


def test_apply_tvg_commutes_with_leading_slice():
    ds = _ds()
    full = apply_tvg(ds).isel(sample=slice(CUT, None))
    cropped = apply_tvg(ds.isel(sample=slice(CUT, None)))
    np.testing.assert_allclose(
        full["backscatter_intensity"].values,
        cropped["backscatter_intensity"].values,
        rtol=1e-6,
    )


def test_apply_tvg_preserves_dtype():
    ds = _ds()
    assert ds["backscatter_intensity"].dtype == np.float32
    out = apply_tvg(ds)
    assert out["backscatter_intensity"].dtype == np.float32


def test_apply_tvg_without_coordinate_still_works():
    out = apply_tvg(_ds(with_coord=False))
    assert out["backscatter_intensity"].shape[-1] == N_SAMPLES
    geo = compute_geometric_ranges(_ds(with_coord=False))
    assert geo["slant_range_m"].shape[-1] == N_SAMPLES


def test_apply_tvg_to_amplitude_sample_idx_offset():
    amp = np.full((2, N_SAMPLES), 2.0, dtype=np.float32)
    full = apply_tvg_to_amplitude(amp, 0.75, freq_hz=100e3)[:, CUT:]
    cropped = apply_tvg_to_amplitude(
        amp[:, CUT:], 0.75, freq_hz=100e3, sample_idx=np.arange(CUT, N_SAMPLES)
    )
    np.testing.assert_allclose(full, cropped)
