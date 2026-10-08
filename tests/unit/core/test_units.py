"""core.units: amplitude (20*log10) conversion, floor handling, floor-clip masking."""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.units import (
    FLOOR_DB,
    MIN_LINEAR,
    db_to_linear,
    linear_to_db,
    mask_floor_clip,
    mask_floor_db,
)


def test_constants_consistent():
    assert FLOOR_DB == -120.0
    assert MIN_LINEAR == 1e-6
    assert 20.0 * np.log10(MIN_LINEAR) == pytest.approx(FLOOR_DB)


def test_amplitude_convention():
    assert linear_to_db(10.0) == pytest.approx(20.0)
    assert linear_to_db(1.0) == pytest.approx(0.0)


def test_round_trip():
    x = np.array([1e-3, 0.5, 1.0, 40.0, 81.0])
    np.testing.assert_allclose(db_to_linear(linear_to_db(x)), x)


def test_zero_and_negative_hit_floor():
    out = linear_to_db(np.array([0.0, -5.0, MIN_LINEAR / 10]))
    np.testing.assert_allclose(out, FLOOR_DB)


def test_nan_preserved():
    out = linear_to_db(np.array([np.nan, 1.0]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.0)


def test_custom_floor():
    assert linear_to_db(np.array([0.0]), floor_db=-60.0)[0] == pytest.approx(-60.0)


def test_dtype_preserved_float32():
    assert linear_to_db(np.array([1.0, 2.0], dtype=np.float32)).dtype == np.float32


def test_dataarray_input_keeps_type_and_coords():
    da = xr.DataArray([0.0, 1.0, np.nan], dims="x", coords={"x": [0, 1, 2]})
    out = linear_to_db(da)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_allclose(out.values, [FLOOR_DB, 0.0, np.nan])
    assert list(out.x.values) == [0, 1, 2]


def test_mask_floor_clip_masks_both_artifact_values():
    out = mask_floor_clip(np.array([0.0, -120.0, 3.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 3.0


def test_mask_floor_clip_returns_input_when_clean():
    a = np.array([1.0, 2.0])
    assert mask_floor_clip(a) is a


def test_mask_floor_db_masks_at_or_below_floor():
    out = mask_floor_db(np.array([-130.0, -120.0, -119.9, 0.0]))
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [-119.9, 0.0])
