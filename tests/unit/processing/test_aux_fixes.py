"""
Unit tests for sidescanpy.processing.aux_fixes (held-value reduction) and
its use by the interpolate_aux step.

Sample-and-hold logs repeat the latest sensor value at every ping or
packet; the scenario in ``tests.unit.synthetic.make_held_aux_scenario``
has a known true track to measure the result against.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.history import read_history
from sidescanpy.core.schema import make_empty_attitude_record, make_empty_navigation_record
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing import correct_file, correct_survey
from sidescanpy.processing.aux_fixes import (
    POSITION_GROUP,
    AuxUpdateStats,
    aux_update_events,
    held_value_fixes,
)
from sidescanpy.processing.interpolate import (
    DEFAULT_FIELD_SOURCES,
    interpolate_aux_onto_dataset,
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
)
from tests.unit.synthetic import (
    _M_PER_DEG_LAT,
    make_held_aux_scenario,
    make_raw_dataset,
    make_split_update_scenario,
)

T0 = np.datetime64("2026-01-01T00:00:00", "ns")


def _record(seconds, **vars_) -> xr.Dataset:
    times = T0 + (np.asarray(seconds, dtype=float) * 1e9).round().astype("timedelta64[ns]")
    return xr.Dataset(
        {k: (("obs",), np.asarray(v, dtype=float)) for k, v in vars_.items()},
        coords={"time": (("obs",), times)},
        attrs={"source": "test"},
    )


def _secs(ds: xr.Dataset) -> np.ndarray:
    return (ds["time"].values - T0) / np.timedelta64(1, "s")


def _interp_error_m(ds: xr.Dataset, sc: dict, mask: np.ndarray) -> np.ndarray:
    dn = (ds["latitude"].values - sc["lat"]) * _M_PER_DEG_LAT
    de = (ds["longitude"].values - sc["lon"]) * sc["m_per_deg_lon"]
    return np.hypot(dn, de)[mask]


def _repeat_fraction(ds: xr.Dataset, mask: np.ndarray) -> float:
    lat, lon = ds["latitude"].values, ds["longitude"].values
    same = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])
    pair = mask[1:] & mask[:-1]
    return float(same[pair].mean())


def _hdg_err(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope="module")
def scenario():
    return make_held_aux_scenario()


@pytest.fixture(scope="module")
def fixed_and_raw(scenario):
    s = scenario["survey"]
    fixed = interpolate_aux_onto_pings(s).get_frequency(scenario["freq_key"]).dataset
    raw = interpolate_aux_onto_pings(s, held_values="raw").get_frequency(scenario["freq_key"]).dataset
    return fixed, raw


# ============================================================
# held_value_fixes ON SMALL RECORDS
# ============================================================

@pytest.mark.unit
class TestHeldValueFixes:
    def test_keeps_changes_at_midpoints_when_held(self):
        rec = _record([0, 1, 2, 3, 4, 5], v=[1, 1, 1, 2, 2, 3])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["v"].values, [1, 2, 3])
        np.testing.assert_allclose(_secs(out), [0.0, 2.5, 4.5])

    def test_unique_values_unchanged(self):
        rec = _record([0, 1, 2, 3], v=[1, 2, 3, 4])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["v"].values, rec["v"].values)
        np.testing.assert_array_equal(out["time"].values, rec["time"].values)

    def test_fix_after_single_entry_keeps_own_time(self):
        rec = _record([0, 1, 2, 3], v=[1, 1, 2, 3])
        out = held_value_fixes(rec)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5, 3.0])

    def test_lat_lon_are_separate_groups_by_default(self):
        # Only longitude changes at t=2: a longitude fix there, no latitude fix.
        rec = _record([0, 1, 2, 3], latitude=[1, 1, 1, 1], longitude=[5, 5, 6, 6])
        out = held_value_fixes(rec, max_hold_s=np.inf)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])
        np.testing.assert_array_equal(out["latitude"].values, [1, np.nan])
        np.testing.assert_array_equal(out["longitude"].values, [5, 6])

    def test_position_group_joins_lat_lon_when_asked(self):
        rec = _record([0, 1, 2, 3], latitude=[1, 1, 1, 1], longitude=[5, 5, 6, 6])
        out = held_value_fixes(rec, groups=[POSITION_GROUP], max_hold_s=np.inf)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])
        np.testing.assert_array_equal(out["latitude"].values, [1, 1])
        np.testing.assert_array_equal(out["longitude"].values, [5, 6])

    def test_interleaved_lat_lon_updates_give_even_track_by_default(self):
        # A 1 Hz GPS logged at 20 Hz whose latitude reaches the log 0.1 s
        # before longitude: a joint group turns the latitude-only update into
        # a fix with a stale longitude, a stair-step once per second.
        t_log = np.arange(0, 60, 0.05)
        fix_t = np.floor(t_log)                        # latest 1 Hz fix time
        lon_t = np.where(t_log - fix_t >= 0.1, fix_t, fix_t - 1)
        speed = 1e-5                                   # deg/s on each axis
        rec = _record(t_log, latitude=fix_t * speed, longitude=np.maximum(lon_t, 0) * speed)
        ping_t = np.arange(5, 55, 0.1)

        def spread(out):
            secs = _secs(out)
            pos = []
            for v in ("latitude", "longitude"):
                ok = np.isfinite(out[v].values)
                pos.append(np.interp(ping_t, secs[ok], out[v].values[ok]))
            step = np.hypot(np.diff(pos[0]), np.diff(pos[1]))
            return np.percentile(step, 95) / np.median(step)

        assert spread(held_value_fixes(rec)) < 1.2
        assert spread(held_value_fixes(rec, groups=[POSITION_GROUP])) > 1.5

    def test_other_variables_are_own_groups_with_nan_elsewhere(self):
        rec = _record([0, 1, 2, 3], roll=[0, 0, 1, 1], pitch=[5, 6, 6, 6])
        out = held_value_fixes(rec)
        np.testing.assert_allclose(_secs(out), [0.0, 1.0, 1.5])
        np.testing.assert_array_equal(out["roll"].values, [0, np.nan, 1])
        np.testing.assert_array_equal(out["pitch"].values, [5, 6, np.nan])

    def test_explicit_groups(self):
        rec = _record([0, 1, 2, 3], roll=[0, 0, 1, 1], pitch=[5, 6, 6, 6])
        out = held_value_fixes(rec, groups=[("roll", "pitch")])
        # As one group, every entry up to t=2 changes something: own times.
        np.testing.assert_allclose(_secs(out), [0.0, 1.0, 2.0])
        assert np.isfinite(out["roll"].values).all()

    def test_nan_entries_are_never_fixes(self):
        rec = _record([0, 1, 2, 3, 4], v=[1, np.nan, 1, 2, np.nan])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["v"].values, [1, 2])
        np.testing.assert_allclose(_secs(out), [0.0, 2.5])

    def test_duplicate_timestamps_last_entry_wins(self):
        rec = _record([0, 1, 2, 2, 3], v=[1, 1, 1, 2, 2])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["v"].values, [1, 2])
        # Old then new value logged at t=2: the update was seen at t=2.
        np.testing.assert_allclose(_secs(out), [0.0, 2.0])

    def test_duplicate_timestamps_agreeing_collapse_to_one(self):
        rec = _record([0, 1, 2, 2, 3], v=[1, 1, 2, 2, 2])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["v"].values, [1, 2])
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])

    def test_unsorted_input_sorted(self):
        rec = _record([2, 0, 1, 3], v=[2, 1, 1, 2])
        out = held_value_fixes(rec)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])

    def test_long_hold_adds_anchor_at_last_held_entry(self):
        rec = _record(np.arange(8.0), v=[1, 2, 2, 2, 2, 2, 3, 4])
        out = held_value_fixes(rec, max_hold_s=2.0)
        # 2 held from t=1 to t=5 (4 s) -> anchor at t=5.
        np.testing.assert_allclose(_secs(out), [0.0, 1.0, 5.0, 5.5, 7.0])
        np.testing.assert_array_equal(out["v"].values, [1, 2, 2, 3, 4])

    def test_inf_max_hold_disables_anchors(self):
        rec = _record(np.arange(8.0), v=[1, 2, 2, 2, 2, 2, 3, 4])
        out = held_value_fixes(rec, max_hold_s=np.inf)
        np.testing.assert_array_equal(out["v"].values, [1, 2, 3, 4])

    def test_attrs_kept_and_counts_added(self):
        rec = _record([0, 1, 2, 3, 4, 5], v=[1, 1, 1, 2, 2, 3])
        out = held_value_fixes(rec)
        assert out.attrs["source"] == "test"
        assert out.attrs["held_value_fixes_entries_in"] == 6
        assert out.attrs["held_value_fixes_entries_out"] == 3
        assert out.attrs["held_value_fixes_dropped"] == 3
        assert out.attrs["held_value_fixes_max_hold_s"] == 2.0

    def test_empty_record(self):
        out = held_value_fixes(make_empty_navigation_record())
        assert out.sizes["obs"] == 0
        assert set(out.data_vars) == {"latitude", "longitude"}

    def test_unknown_group_variable_raises(self):
        with pytest.raises(KeyError):
            held_value_fixes(_record([0, 1], v=[1, 2]), groups=[("v", "w")])

    def test_negative_max_hold_raises(self):
        with pytest.raises(ValueError):
            held_value_fixes(_record([0, 1], v=[1, 2]), max_hold_s=-1.0)

    def test_heading_change_across_north_is_a_fix(self):
        rec = _record([0, 1, 2, 3], heading_deg=[359.0, 359.0, 1.0, 1.0])
        out = held_value_fixes(rec)
        np.testing.assert_array_equal(out["heading_deg"].values, [359.0, 1.0])


# ============================================================
# interpolate_aux ON THE HELD-LOG SCENARIO
# ============================================================

@pytest.mark.unit
class TestInterpolateHeldScenario:
    def test_scenario_input_is_held(self, scenario):
        ds = scenario["survey"].get_frequency(scenario["freq_key"]).dataset
        assert _repeat_fraction(ds, scenario["moving"]) > 0.8

    def test_moving_pings_do_not_repeat(self, scenario, fixed_and_raw):
        fixed, raw = fixed_and_raw
        assert _repeat_fraction(fixed, scenario["moving"]) == 0.0
        assert _repeat_fraction(raw, scenario["moving"]) > 0.5

    def test_position_error_much_smaller_than_raw(self, scenario, fixed_and_raw):
        fixed, raw = fixed_and_raw
        err_fixed = _interp_error_m(fixed, scenario, scenario["moving"])
        err_raw = _interp_error_m(raw, scenario, scenario["moving"])
        rms_fixed = np.sqrt(np.mean(err_fixed**2))
        rms_raw = np.sqrt(np.mean(err_raw**2))
        assert rms_fixed < 0.15
        assert err_fixed.max() < 0.5
        assert rms_fixed < rms_raw / 5

    def test_flat_during_stop(self, scenario, fixed_and_raw):
        fixed, _ = fixed_and_raw
        t = scenario["t"]
        inner = scenario["stopped"] & (t > 72.0) & (t < 83.0)
        lat, lon = fixed["latitude"].values[inner], fixed["longitude"].values[inner]
        assert np.ptp(lat) == 0.0 and np.ptp(lon) == 0.0
        assert _interp_error_m(fixed, scenario, inner).max() < 0.05

    def test_heading_across_north_stays_near_north(self, scenario, fixed_and_raw):
        fixed, raw = fixed_and_raw
        t = scenario["t"]
        turn = (t >= 42.0) & (t < 48.0)
        hdg = fixed["heading_deg"].values
        assert _hdg_err(hdg[turn], 0.0).max() < 25.0
        err = _hdg_err(hdg, scenario["heading"])[scenario["moving"]]
        err_raw = _hdg_err(raw["heading_deg"].values, scenario["heading"])[scenario["moving"]]
        # Largest error is at the turn's start/end corners (4 deg/s, ~2 Hz updates).
        assert err.max() < 2.0
        assert err.mean() < err_raw.mean() / 3

    def test_attitude_error_smaller_than_raw(self, scenario, fixed_and_raw):
        fixed, raw = fixed_and_raw
        m = scenario["moving"]
        for var, truth in (("roll_deg", "roll"), ("pitch_deg", "pitch")):
            e_fix = np.abs(fixed[var].values - scenario[truth])[m]
            e_raw = np.abs(raw[var].values - scenario[truth])[m]
            assert e_fix.mean() < e_raw.mean() / 2, var

    def test_raw_mode_matches_plain_interpolation(self, scenario, fixed_and_raw):
        _, raw = fixed_and_raw
        s = scenario["survey"]
        ch = s.get_frequency(scenario["freq_key"])
        ref = interpolate_onto_channel(ch, s.navigation, field_map={"latitude": "latitude", "longitude": "longitude"})
        ref = interpolate_onto_channel(
            ref, s.attitude,
            field_map={f: v for f, (a, v) in DEFAULT_FIELD_SOURCES.items() if a == "attitude"},
        )
        for var in DEFAULT_FIELD_SOURCES:
            np.testing.assert_array_equal(raw[var].values, ref.dataset[var].values, err_msg=var)

    def test_history_records_params(self, fixed_and_raw):
        fixed, raw = fixed_and_raw
        rec = [r for r in read_history(fixed) if r["step"] == "interpolate_aux"][-1]
        assert rec["params"]["held_values"] == "fixes"
        assert rec["params"]["max_hold_s"] == 2.0
        assert rec["params"]["per_ping_fallback"] is True
        assert rec["params"]["per_ping_fallback_fields"] == []
        rec_raw = [r for r in read_history(raw) if r["step"] == "interpolate_aux"][-1]
        assert rec_raw["params"]["held_values"] == "raw"

    def test_invalid_held_values_raises(self, scenario):
        with pytest.raises(ValueError, match="held_values"):
            interpolate_aux_onto_pings(scenario["survey"], held_values="bogus")

    def test_correct_survey_params(self, scenario):
        out = correct_survey(
            scenario["survey"], ["interpolate_aux"],
            {"interpolate_aux": {"held_values": "raw", "max_hold_s": 5.0}},
        )
        rec = read_history(out.get_frequency(scenario["freq_key"]).dataset)[-1]
        assert rec["params"]["held_values"] == "raw"
        assert rec["params"]["max_hold_s"] == 5.0


# ============================================================
# PER-PING FALLBACK (no aux record)
# ============================================================

def _no_aux(scenario) -> SideScanSurvey:
    s = scenario["survey"].copy()
    s.navigation = make_empty_navigation_record()
    s.attitude = make_empty_attitude_record()
    return s


@pytest.mark.unit
class TestPerPingFallback:
    def test_held_per_ping_values_interpolated(self, scenario, caplog):
        caplog.set_level(logging.DEBUG, logger="sidescanpy.processing.interpolate")
        out = interpolate_aux_onto_pings(_no_aux(scenario)).get_frequency(scenario["freq_key"]).dataset
        before = scenario["survey"].get_frequency(scenario["freq_key"]).dataset
        m = scenario["moving"]
        assert _repeat_fraction(out, m) == 0.0
        rms_fixed = np.sqrt(np.mean(_interp_error_m(out, scenario, m) ** 2))
        rms_held = np.sqrt(np.mean(_interp_error_m(before, scenario, m) ** 2))
        assert rms_fixed < rms_held / 5
        assert "per-ping" in caplog.text
        assert "is empty" not in caplog.text

    def test_history_records_fallback_fields(self, scenario):
        out = interpolate_aux_onto_pings(_no_aux(scenario)).get_frequency(scenario["freq_key"]).dataset
        rec = read_history(out)[-1]
        assert rec["params"]["per_ping_fallback_fields"] == sorted(
            ["latitude", "longitude", "heading_deg", "roll_deg", "pitch_deg", "heave_m"]
        )

    def test_unique_per_ping_values_unchanged(self):
        n = 50
        ds = make_raw_dataset(n_pings=n)
        ds = ds.assign(
            latitude=("ping", 41.5 + 1e-5 * np.arange(n)),
            longitude=("ping", -71.5 - 1e-5 * np.arange(n)),
            heading_deg=("ping", (355.0 + 0.3 * np.arange(n)) % 360.0),
            roll_deg=("ping", np.sin(np.arange(n) / 3.0)),
            pitch_deg=("ping", np.cos(np.arange(n) / 4.0)),
        )
        s = SideScanSurvey(source_files=[Path("unique.sdf")], source_format="SDF", survey_name="unique")
        s.add_frequency("hz_200000", ds)
        out = interpolate_aux_onto_pings(s).get_frequency("hz_200000").dataset
        for var in DEFAULT_FIELD_SOURCES:
            np.testing.assert_array_equal(out[var].values, ds[var].values, err_msg=var)
        assert set(read_history(out)[-1]["params"]["per_ping_fallback_fields"]) == set(DEFAULT_FIELD_SOURCES)

    def test_fallback_disabled_keeps_old_behavior(self, scenario, caplog):
        caplog.set_level(logging.WARNING, logger="sidescanpy.processing.interpolate")
        s = _no_aux(scenario)
        out = interpolate_aux_onto_pings(s, per_ping_fallback=False).get_frequency(scenario["freq_key"]).dataset
        before = s.get_frequency(scenario["freq_key"]).dataset
        for var in DEFAULT_FIELD_SOURCES:
            np.testing.assert_array_equal(out[var].values, before[var].values)
        assert "is empty" in caplog.text
        assert read_history(out)[-1]["params"]["per_ping_fallback_fields"] == []

    def test_fallback_on_ping_slice_uses_whole_channel(self, scenario):
        s = _no_aux(scenario)
        full = interpolate_aux_onto_pings(s).get_frequency(scenario["freq_key"]).dataset
        ds = s.get_frequency(scenario["freq_key"]).dataset
        part = interpolate_aux_onto_dataset(ds.isel(ping=slice(333, 777)), s)
        for var in DEFAULT_FIELD_SOURCES:
            np.testing.assert_array_equal(part[var].values, full[var].values[333:777], err_msg=var)

    def test_chunked_correct_file_matches_unchunked(self, scenario, tmp_path):
        s = _no_aux(scenario)
        whole, chunked = tmp_path / "whole.ss.h5", tmp_path / "chunked.ss.h5"
        write_survey(s, whole)
        write_survey(s, chunked)
        correct_file(whole, ["interpolate_aux"])
        correct_file(chunked, ["interpolate_aux"], chunk_pings=250)
        a = read_survey(whole).get_frequency(scenario["freq_key"]).dataset
        b = read_survey(chunked).get_frequency(scenario["freq_key"]).dataset
        for var in DEFAULT_FIELD_SOURCES:
            np.testing.assert_allclose(a[var].values, b[var].values, rtol=0, atol=0, err_msg=var)
        assert _repeat_fraction(b, scenario["moving"]) == 0.0


# ============================================================
# MERGING AN UPDATE SPREAD OVER CONSECUTIVE ENTRIES
# ============================================================

def _speed_spread(ds_or_fixes, sc: dict, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """(p95-p5)/mean of per-ping speed, and per-ping position error (m)."""
    if "ping" in ds_or_fixes.dims:
        lat, lon = ds_or_fixes["latitude"].values, ds_or_fixes["longitude"].values
    else:
        secs = _secs(ds_or_fixes)
        pos = []
        for v in ("latitude", "longitude"):
            ok = np.isfinite(ds_or_fixes[v].values)
            pos.append(np.interp(sc["t"], secs[ok], ds_or_fixes[v].values[ok]))
        lat, lon = pos
    north = (lat - sc["lat"][0]) * _M_PER_DEG_LAT
    east = (lon - sc["lon"][0]) * sc["m_per_deg_lon"]
    speed = np.hypot(np.diff(north), np.diff(east)) / np.diff(sc["t"])
    speed = speed[mask[1:] & mask[:-1]]
    err = np.hypot((lat - sc["lat"]) * _M_PER_DEG_LAT, (lon - sc["lon"]) * sc["m_per_deg_lon"])[mask]
    p5, p95 = np.percentile(speed, [5, 95])
    return float((p95 - p5) / speed.mean()), err


@pytest.fixture(scope="module")
def split():
    return make_split_update_scenario()


@pytest.mark.unit
class TestMergeWindow:
    def test_default_none_is_unchanged(self):
        rec = _record(np.arange(10.0), v=[1, 1, 2, 2.1, 2.1, 2.1, 3, 3.1, 3.1, 3.1])
        a = held_value_fixes(rec)
        b = held_value_fixes(rec, merge_window_s=None)
        xr.testing.assert_identical(a, b)
        assert "held_value_fixes_merge_window_s" not in a.attrs

    def test_close_changes_merge_into_one_event(self):
        # 1 held, then 2 arrives at t=2 and is refined to 2.1 at t=3.
        rec = _record(np.arange(8.0), v=[1, 1, 2, 2.1, 2.1, 2.1, 2.1, 2.1])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        np.testing.assert_array_equal(out["v"].values, [1, 2.1])
        # Timed at the midpoint before the FIRST change of the cluster.
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])
        assert out.attrs["held_value_fixes_merge_window_s"] == 1.0

    def test_changes_further_apart_than_window_stay_separate(self):
        rec = _record(np.arange(8.0), v=[1, 1, 2, 2, 2.1, 2.1, 2.1, 2.1])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        np.testing.assert_array_equal(out["v"].values, [1, 2, 2.1])
        np.testing.assert_allclose(_secs(out), [0.0, 1.5, 3.5])

    def test_chain_of_close_changes_is_one_event(self):
        rec = _record(np.arange(8.0), v=[1, 1, 2, 3, 4, 4, 4, 4])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        np.testing.assert_array_equal(out["v"].values, [1, 4])
        np.testing.assert_allclose(_secs(out), [0.0, 1.5])

    def test_first_entry_is_not_merged(self):
        # The record's first entry is a start, not an observed change.
        rec = _record(np.arange(5.0), v=[1, 2, 2, 2, 2])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        np.testing.assert_array_equal(out["v"].values, [1, 2])
        np.testing.assert_allclose(_secs(out), [0.0, 1.0])

    def test_first_change_after_change_keeps_own_time(self):
        # 2 follows another change at t=1, so it is not after a hold.
        rec = _record(np.arange(7.0), v=[1, 5, 2, 2.1, 2.1, 2.1, 2.1])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        # t=1 and t=2 are 1 s apart, so 5, 2 and 2.1 chain into one event at t=1.
        np.testing.assert_array_equal(out["v"].values, [1, 2.1])
        np.testing.assert_allclose(_secs(out), [0.0, 1.0])

    def test_max_hold_anchor_measured_from_last_change(self):
        # Event: first change t=2, last change t=3; 2.1 held to t=6 (3 s from
        # the last change, 4 s from the first).
        rec = _record(np.arange(8.0), v=[1, 1, 2, 2.1, 2.1, 2.1, 2.1, 3])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=3.5)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5, 6.5])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=2.5)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5, 6.0, 6.5])
        np.testing.assert_array_equal(out["v"].values, [1, 2.1, 2.1, 3])

    def test_groups_merge_independently(self):
        rec = _record(np.arange(6.0), a=[0, 0, 1, 1.1, 1.1, 1.1], b=[0, 0, 0, 5, 5, 5])
        out = held_value_fixes(rec, merge_window_s=1.0, max_hold_s=np.inf)
        np.testing.assert_allclose(_secs(out), [0.0, 1.5, 2.5])
        np.testing.assert_array_equal(out["a"].values, [0, 1.1, np.nan])
        np.testing.assert_array_equal(out["b"].values, [0, np.nan, 5])

    def test_mapping_merges_only_named_groups(self):
        rec = _record(np.arange(6.0), a=[0, 0, 1, 1.1, 1.1, 1.1], b=[0, 0, 1, 1.1, 1.1, 1.1])
        out = held_value_fixes(rec, merge_window_s={"a": 1.0}, max_hold_s=np.inf)
        ok_a, ok_b = np.isfinite(out["a"].values), np.isfinite(out["b"].values)
        np.testing.assert_array_equal(out["a"].values[ok_a], [0, 1.1])
        np.testing.assert_array_equal(out["b"].values[ok_b], [0, 1, 1.1])
        np.testing.assert_allclose(_secs(out)[ok_a], [0.0, 1.5])
        np.testing.assert_allclose(_secs(out)[ok_b], [0.0, 1.5, 3.0])
        assert out.attrs["held_value_fixes_merge_window_s"] == '{"a": 1.0}'

    def test_mapping_group_takes_smallest_window(self):
        rec = _record(np.arange(6.0), a=[0, 0, 1, 1.1, 1.1, 1.1], b=[0, 0, 0, 0, 0, 0])
        out = held_value_fixes(
            rec, groups=[("a", "b")], merge_window_s={"a": 1.0, "b": 0.5}, max_hold_s=np.inf,
        )
        np.testing.assert_array_equal(out["a"].values, [0, 1, 1.1])

    def test_negative_merge_window_raises(self):
        with pytest.raises(ValueError, match="merge_window_s"):
            held_value_fixes(_record([0, 1], v=[1, 2]), merge_window_s=-0.1)


@pytest.mark.unit
class TestSplitUpdateScenario:
    def test_one_event_per_real_update(self, split):
        out = held_value_fixes(split["nav"], groups=[POSITION_GROUP], merge_window_s=0.15)
        # Every update is seen by exactly one log entry first; the record's
        # first entry is its own fix.
        n_updates = int(np.sum(split["update_t"] > 0))
        assert out.sizes["obs"] == n_updates + 1
        # Each event is timed within one log interval of the real update.
        lag = _secs(out)[1:] - split["update_t"][1:]
        assert np.all(np.abs(lag) <= 1.0 / 14.5)

    def test_merging_removes_speed_spikes(self, split):
        m = split["inner"]
        sp_none, err_none = _speed_spread(held_value_fixes(split["nav"], groups=[POSITION_GROUP]), split, m)
        sp_merge, err_merge = _speed_spread(
            held_value_fixes(split["nav"], groups=[POSITION_GROUP], merge_window_s=0.15), split, m,
        )
        assert sp_none > 1.0
        assert sp_merge < 0.5
        assert np.sqrt(np.mean(err_merge**2)) < 0.05
        assert err_merge.max() < 0.1

    def test_interpolate_aux_threads_merge_and_groups(self, split):
        s = split["survey"]
        out = interpolate_aux_onto_pings(
            s, groups=[POSITION_GROUP], merge_window_s=0.15,
        ).get_frequency(split["freq_key"]).dataset
        ref = held_value_fixes(split["nav"], groups=[POSITION_GROUP], merge_window_s=0.15)
        sp_step, _ = _speed_spread(out, split, split["inner"])
        sp_ref, _ = _speed_spread(ref, split, split["inner"])
        assert sp_step == pytest.approx(sp_ref)
        rec = read_history(out)[-1]
        assert rec["params"]["merge_window_s"] == 0.15
        assert rec["params"]["groups"] == [["latitude", "longitude"]]

    def test_history_defaults(self, split):
        out = interpolate_aux_onto_pings(split["survey"]).get_frequency(split["freq_key"]).dataset
        rec = read_history(out)[-1]
        assert rec["params"]["merge_window_s"] is None
        assert rec["params"]["groups"] is None

    def test_correct_survey_step_params(self, split):
        out = correct_survey(
            split["survey"], ["interpolate_aux"],
            {"interpolate_aux": {"groups": [list(POSITION_GROUP)], "merge_window_s": 0.15}},
        )
        rec = read_history(out.get_frequency(split["freq_key"]).dataset)[-1]
        assert rec["params"]["merge_window_s"] == 0.15
        assert rec["params"]["groups"] == [["latitude", "longitude"]]

    def test_mapping_leaves_attitude_unmerged(self, split):
        s = split["survey"].copy()
        att = s.attitude.copy()
        # A fast roll sensor: a new value at every entry (~14 Hz).
        att["roll_deg"] = (("obs",), np.sin(np.arange(att.sizes["obs"]) / 5.0).astype(np.float32))
        s.attitude = att
        window = {"latitude": 0.15, "longitude": 0.15}
        out = interpolate_aux_onto_pings(
            s, groups=[POSITION_GROUP], merge_window_s=window,
        ).get_frequency(split["freq_key"]).dataset
        np.testing.assert_allclose(out["roll_deg"].values, att["roll_deg"].values, atol=1e-6)
        sp, _ = _speed_spread(out, split, split["inner"])
        assert sp < 0.5
        assert read_history(out)[-1]["params"]["merge_window_s"] == window

    def test_groups_missing_from_a_record_are_ignored(self, split):
        # POSITION_GROUP is not in the attitude record: no error there.
        out = interpolate_aux_onto_pings(
            split["survey"], groups=[POSITION_GROUP, ("roll_deg", "pitch_deg")],
        ).get_frequency(split["freq_key"]).dataset
        assert np.isfinite(out["latitude"].values).all()

    def test_invalid_merge_window_raises(self, split):
        with pytest.raises(ValueError, match="merge_window_s"):
            interpolate_aux_onto_pings(split["survey"], merge_window_s=-1.0)


@pytest.mark.unit
class TestAuxUpdateEvents:
    def test_bimodal_changes_unimodal_events(self, split):
        stats = aux_update_events(split["nav"], groups=[POSITION_GROUP], merge_window_s=0.15)
        st = stats["latitude+longitude"]
        assert isinstance(st, AuxUpdateStats)
        assert st.group == ("latitude", "longitude")
        n_updates = int(np.sum(split["update_t"] > 0))
        assert st.entries == split["nav"].sizes["obs"]
        assert st.changes == 2 * n_updates
        assert st.events == n_updates
        # Raw gaps: half are 1 entry (follow-ups), half 3-4 (to the next update).
        assert st.change_gap_entries["p5"] == 1
        assert st.change_gap_entries["p25"] == 1
        assert st.change_gap_entries["p75"] >= 3
        assert st.change_gap_entries["p95"] <= 4
        assert st.change_gap_s["p25"] == pytest.approx(1 / 14.5, abs=1e-6)
        # Events: every gap is 4-5 entries.
        assert st.event_gap_entries["p5"] >= 4
        assert st.event_gap_entries["p95"] <= 5
        assert st.event_gap_s["p50"] == pytest.approx(1 / 3.5, rel=0.1)
        assert st.merged_fraction == pytest.approx(0.5, abs=0.01)

    def test_default_groups_and_no_merge(self):
        rec = _record(np.arange(6.0), v=[1, 1, 2, 2, 3, 3])
        stats = aux_update_events(rec)
        st = stats["v"]
        assert st.entries == 6 and st.changes == 2 and st.events == 2
        assert st.merge_window_s is None
        assert np.isnan(st.merged_fraction)
        assert st.change_gap_entries["p50"] == 2
        assert st.change_gap_s["p50"] == 2.0
        assert st.as_dict()["group"] == ["v"]

    def test_mapping_window_per_group(self, split):
        stats = aux_update_events(split["nav"], merge_window_s={"latitude": 0.15})
        assert stats["latitude"].merge_window_s == pytest.approx(0.15)
        assert stats["longitude"].merge_window_s is None

    def test_no_changes_gives_nan_gaps(self):
        st = aux_update_events(_record(np.arange(4.0), v=[1, 1, 1, 1]))["v"]
        assert st.changes == 0 and st.events == 0
        assert np.isnan(st.change_gap_s["p50"])
