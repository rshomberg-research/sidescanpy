"""
Unit tests for sidescanpy.analysis.navigation.
"""

import numpy as np
import xarray as xr

from sidescanpy.analysis.navigation import summarize_navigation


def _make_ds(*, n_pings: int = 10, **overrides) -> xr.Dataset:
    data_vars = {
        "heading_deg": (["ping"], np.linspace(10.0, 100.0, n_pings, dtype=np.float64)),
        "pitch_deg": (["ping"], np.zeros(n_pings)),
        "roll_deg": (["ping"], np.zeros(n_pings)),
        "altitude_m": (["ping"], np.full(n_pings, 5.0)),
        "depth_m": (["ping"], np.full(n_pings, 20.0)),
        "heave_m": (["ping"], np.zeros(n_pings)),
        "latitude": (["ping"], np.linspace(40.0, 40.01, n_pings)),
        "longitude": (["ping"], np.linspace(-70.0, -69.99, n_pings)),
    }
    data_vars.update(overrides)
    return xr.Dataset(data_vars)


class TestSummarizeNavigation:
    def test_reports_min_max_mean_per_var(self):
        ds = _make_ds()
        result = summarize_navigation(ds)
        assert result.field_stats["altitude_m"].minimum == 5.0
        assert result.field_stats["altitude_m"].maximum == 5.0
        assert result.field_stats["depth_m"].mean == 20.0
        assert result.field_stats["altitude_m"].n_nan == 0

    def test_compass_bearing_classification(self):
        ds = _make_ds(n_pings=5, heading_deg=(["ping"], np.array([0.0, 90.0, 180.0, 270.0, 359.9])))
        result = summarize_navigation(ds)
        assert result.heading_convention == "compass_bearing"
        assert result.heading_frac_compass_range == 1.0
        assert result.heading_frac_out_of_range == 0.0

    def test_signed_classification(self):
        ds = _make_ds(n_pings=5, heading_deg=(["ping"], np.array([-170.0, -10.0, 0.0, 10.0, 170.0])))
        result = summarize_navigation(ds)
        assert result.heading_convention == "signed"
        assert result.heading_frac_signed_range == 1.0

    def test_ambiguous_classification_and_note(self):
        ds = _make_ds(n_pings=3, heading_deg=(["ping"], np.array([-400.0, -300.0, -250.0])))
        result = summarize_navigation(ds)
        assert result.heading_convention == "ambiguous"
        assert result.heading_frac_out_of_range == 1.0
        assert any("outside both" in note for note in result.notes)

    def test_mixed_range_is_ambiguous_not_a_silent_convention_pick(self):
        # some pings look like a compass bearing (>=180), others like a
        # signed heading with no compass-range equivalent (negative) --
        # neither bucket holds for every value, so this must not silently
        # report either "compass_bearing" or "signed".
        ds = _make_ds(n_pings=4, heading_deg=(["ping"], np.array([-90.0, 45.0, 270.0, -45.0])))
        result = summarize_navigation(ds)
        assert result.heading_convention == "ambiguous"

    def test_nan_values_counted_and_noted(self):
        ds = _make_ds(n_pings=3, heading_deg=(["ping"], np.array([10.0, np.nan, 30.0])))
        result = summarize_navigation(ds)
        assert result.field_stats["heading_deg"].n_nan == 1
        assert any("NaN" in note for note in result.notes)

    def test_missing_var_skipped_not_raised(self):
        ds = _make_ds().drop_vars("heave_m")
        result = summarize_navigation(ds)
        assert "heave_m" not in result.field_stats

    def test_missing_heading_var_gives_none_classification(self):
        ds = _make_ds().drop_vars("heading_deg")
        result = summarize_navigation(ds)
        assert result.heading_convention is None
        assert result.heading_frac_out_of_range is None

    def test_all_nan_heading_gives_none_classification(self):
        ds = _make_ds(n_pings=3, heading_deg=(["ping"], np.full(3, np.nan)))
        result = summarize_navigation(ds)
        assert result.heading_convention is None

    def test_custom_heading_var(self):
        ds = _make_ds()
        ds["yaw_deg"] = (["ping"], np.full(ds.sizes["ping"], -500.0))
        result = summarize_navigation(ds, heading_var="yaw_deg")
        assert result.heading_convention == "ambiguous"
        # the default heading_deg field (in-range here) must not be what drove the result
        assert "yaw_deg" in "".join(result.notes)
