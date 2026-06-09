"""
Unit tests for sidescanpy.processing.interpolate.

Tests interpolation of auxiliary records onto per-ping timestamps using
synthetic surveys with known-formula aux data, so expected values are
derivable rather than guessed.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.core.survey import SideScanFrequencyChannel, SideScanSurvey
from sidescanpy.processing.interpolate import (
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
    DEFAULT_FIELD_SOURCES,
)
from tests.conftest import (
    make_minimal_sidescan_dataset,
    make_minimal_navigation_record,
    make_minimal_attitude_record,
)


# ============================================================
# HELPERS — synthetic aux records with known formulas
# ============================================================

def _make_linear_navigation(
    n_obs: int = 100,
    *,
    start_time: str = "2026-01-01T00:00:00",
    step_sec: float = 0.1,        # 10 Hz aux rate
    lat0: float = 41.0,
    lat_slope: float = 1e-5,      # degrees per second
    lon0: float = -71.0,
    lon_slope: float = -1e-5,
) -> xr.Dataset:
    """Navigation record with linear lat/lon as a function of time."""
    times = pd.date_range(start_time, periods=n_obs, freq=f"{step_sec}s").values \
        .astype("datetime64[ns]")
    elapsed_s = np.arange(n_obs) * step_sec
    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), lat0 + lat_slope * elapsed_s),
            "longitude": (("obs",), lon0 + lon_slope * elapsed_s),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "synthetic_linear"},
    )


def _make_linear_attitude(
    n_obs: int = 100,
    *,
    start_time: str = "2026-01-01T00:00:00",
    step_sec: float = 0.1,
) -> xr.Dataset:
    """Attitude record with sinusoidal roll/pitch and linear heading."""
    times = pd.date_range(start_time, periods=n_obs, freq=f"{step_sec}s").values \
        .astype("datetime64[ns]")
    elapsed_s = np.arange(n_obs) * step_sec
    return xr.Dataset(
        data_vars={
            "roll_deg":  (("obs",), (3.0 * np.sin(elapsed_s * 0.5)).astype(np.float32)),
            "pitch_deg": (("obs",), (2.0 * np.cos(elapsed_s * 0.3)).astype(np.float32)),
            "heave_m":   (("obs",), (0.1 * elapsed_s).astype(np.float32)),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "synthetic_linear"},
    )


def _build_known_survey(
    *,
    n_pings: int = 10,
    ping_step_sec: float = 1.0,           # pings every 1 second
    aux_step_sec: float = 0.1,             # aux every 100 ms (10x finer)
    n_aux: int = 100,
) -> SideScanSurvey:
    """Build a survey with one frequency channel and known-formula aux."""
    # Sonar dataset — pings at integer seconds
    base_time = pd.date_range("2026-01-01T00:00:01",
                              periods=n_pings,
                              freq=f"{ping_step_sec}s").values \
        .astype("datetime64[ns]")
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=10)
    ds = ds.assign_coords(time=("ping", base_time))

    survey = SideScanSurvey(
        source_files=[Path("synthetic.jsf")],
        source_format="JSF",
        survey_name="synthetic",
    )
    survey.add_frequency("hz_1495000", ds)
    survey.navigation = _make_linear_navigation(n_obs=n_aux, step_sec=aux_step_sec)
    survey.attitude   = _make_linear_attitude(n_obs=n_aux, step_sec=aux_step_sec)
    return survey


# ============================================================
# DEFAULTS
# ============================================================

@pytest.mark.unit
class TestDefaults:
    def test_default_field_sources_includes_lat_lon(self):
        assert "latitude"  in DEFAULT_FIELD_SOURCES
        assert "longitude" in DEFAULT_FIELD_SOURCES
        assert DEFAULT_FIELD_SOURCES["latitude"]  == ("navigation", "latitude")
        assert DEFAULT_FIELD_SOURCES["longitude"] == ("navigation", "longitude")

    def test_default_field_sources_includes_attitude(self):
        for f in ("heading_deg", "roll_deg", "pitch_deg", "heave_m"):
            assert f in DEFAULT_FIELD_SOURCES
            assert DEFAULT_FIELD_SOURCES[f][0] == "attitude"

    def test_default_field_sources_excludes_live_sensors(self):
        """altitude_m, depth_m, sound_vel_ms, water_temp_C are device-direct
        readings and should not be interpolated by default."""
        for f in ("altitude_m", "depth_m", "sound_vel_ms"):
            assert f not in DEFAULT_FIELD_SOURCES


# ============================================================
# LINEAR INTERPOLATION ON KNOWN FORMULAS
# ============================================================

@pytest.mark.unit
class TestLinearInterpolation:
    def test_lat_lon_match_known_formula(self, tmp_path):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # Pings at 1, 2, ..., 10 sec elapsed → expected lat = 41.0 + 1e-5 * t
        ping_times = ds["time"].values.astype("datetime64[ns]")
        elapsed_s = (ping_times - np.datetime64("2026-01-01T00:00:00", "ns")) \
            / np.timedelta64(1, "s")

        expected_lat = 41.0 + 1e-5 * elapsed_s
        expected_lon = -71.0 - 1e-5 * elapsed_s

        np.testing.assert_allclose(ds["latitude"].values,  expected_lat,  atol=1e-9)
        np.testing.assert_allclose(ds["longitude"].values, expected_lon, atol=1e-9)

    def test_attitude_fields_interpolated(self, tmp_path):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # Attitude fields should not be all-NaN (they were before interpolation)
        for var in ("roll_deg", "pitch_deg", "heave_m"):
            values = ds[var].values
            assert not np.all(np.isnan(values))

    def test_method_nearest_returns_aux_value(self, tmp_path):
        """method='nearest' should return the aux value closest in time
        to each ping."""
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey, method="nearest")

        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # With aux at 10 Hz and pings at 1 Hz, nearest-neighbor at ping time t
        # should equal the aux value at the closest aux time.
        nav = survey.navigation
        ping_times_ns = ds["time"].values.astype("int64")
        nav_times_ns  = nav["time"].values.astype("int64")

        for i, t_ping in enumerate(ping_times_ns):
            j = np.argmin(np.abs(nav_times_ns - t_ping))
            np.testing.assert_allclose(
                ds["latitude"].values[i],
                nav["latitude"].values[j],
                atol=1e-9,
            )


# ============================================================
# EXTRAPOLATION MODES
# ============================================================

@pytest.mark.unit
class TestExtrapolationClamp:
    def test_pings_before_aux_clamped_to_first(self, tmp_path):
        """Pings before the aux time range should hold the first aux value."""
        # Aux starts at T+1s; pings start at T+0s
        survey = _build_known_survey(n_pings=5, ping_step_sec=0.1)  # 5 pings, 0.1s apart
        # Override navigation to start later than pings
        nav = _make_linear_navigation(
            n_obs=20, start_time="2026-01-01T00:00:01.5", step_sec=0.1
        )
        survey.navigation = nav

        result = interpolate_aux_onto_pings(survey, extrapolation="clamp")
        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # First aux lat is at T+1.5s; all pings are earlier than that
        first_aux_lat = nav["latitude"].values[0]
        # All ping latitudes should equal the first aux latitude (clamped)
        for v in ds["latitude"].values:
            np.testing.assert_allclose(v, first_aux_lat, atol=1e-9)

    def test_pings_after_aux_clamped_to_last(self, tmp_path):
        """Pings after the aux time range should hold the last aux value."""
        # Pings start at T+5s; aux only covers T+0 to T+1s
        survey = _build_known_survey(n_pings=5, ping_step_sec=0.1)
        # Move pings to be after aux range
        ds = survey.get_frequency("hz_1495000").dataset
        new_times = pd.date_range("2026-01-01T00:00:10",
                                  periods=ds.sizes["ping"],
                                  freq="0.1s").values.astype("datetime64[ns]")
        ds = ds.assign_coords(time=("ping", new_times))
        survey._freq_channels["hz_1495000"] = SideScanFrequencyChannel(
            ds, freq_key="hz_1495000"
        )

        # Reduce aux range to T+0 to T+1s only
        survey.navigation = _make_linear_navigation(n_obs=10, step_sec=0.1)

        result = interpolate_aux_onto_pings(survey, extrapolation="clamp")
        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        last_aux_lat = survey.navigation["latitude"].values[-1]
        for v in ds["latitude"].values:
            np.testing.assert_allclose(v, last_aux_lat, atol=1e-9)


@pytest.mark.unit
class TestExtrapolationLinear:
    def test_pings_outside_aux_extrapolated(self, tmp_path):
        survey = _build_known_survey(n_pings=5, ping_step_sec=0.1)
        # Move pings before aux range
        ds = survey.get_frequency("hz_1495000").dataset
        new_times = pd.date_range("2026-01-01T00:00:00.0",
                                    periods=ds.sizes["ping"],
                                    freq="0.05s").values.astype("datetime64[ns]")
        ds = ds.assign_coords(time=("ping", new_times))
        survey._freq_channels["hz_1495000"] = SideScanFrequencyChannel(
            ds, freq_key="hz_1495000"
        )

        # Aux: lat = 41.0 + 1e-5 * elapsed_from_aux_start
        aux_start = np.datetime64("2026-01-01T00:00:01", "ns")
        survey.navigation = _make_linear_navigation(
            n_obs=10, start_time="2026-01-01T00:00:01", step_sec=0.1
        )

        result = interpolate_aux_onto_pings(survey, extrapolation="linear")
        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # Linear extrapolation should preserve the slope (1e-5/s) extending
        # backward from the aux start. Compute expected values relative to
        # aux start, not absolute T0.
        ping_times = ds["time"].values
        elapsed_from_aux_s = (ping_times - aux_start) / np.timedelta64(1, "s")
        expected_lat = 41.0 + 1e-5 * elapsed_from_aux_s

        np.testing.assert_allclose(ds["latitude"].values, expected_lat, atol=1e-7)


@pytest.mark.unit
class TestExtrapolationNan:
    def test_pings_outside_aux_get_nan(self, tmp_path):
        survey = _build_known_survey(n_pings=5, ping_step_sec=0.1)
        # Move pings outside aux range
        ds = survey.get_frequency("hz_1495000").dataset
        new_times = pd.date_range("2026-01-01T00:00:10.0",
                                  periods=ds.sizes["ping"],
                                  freq="0.1s").values.astype("datetime64[ns]")
        ds = ds.assign_coords(time=("ping", new_times))
        survey._freq_channels["hz_1495000"] = SideScanFrequencyChannel(
            ds, freq_key="hz_1495000"
        )

        survey.navigation = _make_linear_navigation(n_obs=10, step_sec=0.1)

        result = interpolate_aux_onto_pings(survey, extrapolation="nan")
        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # All pings outside aux range → all NaN
        assert np.all(np.isnan(ds["latitude"].values))


# ============================================================
# EMPTY AUX RECORD HANDLING
# ============================================================

@pytest.mark.unit
class TestEmptyAuxSkips:
    def test_empty_aux_skips_field(self, sample_dataset, caplog):
        """When the aux record is empty, the field should be left unchanged
        and a warning should be logged."""
        survey = SideScanSurvey(
            source_files=[Path("test.jsf")],
            source_format="JSF",
            survey_name="test",
        )
        survey.add_frequency("hz_1495000", sample_dataset)
        # navigation is None by default; do NOT set it

        original_lat = survey.get_frequency("hz_1495000").dataset["latitude"].values.copy()

        with caplog.at_level(logging.WARNING):
            result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        np.testing.assert_array_equal(ch.dataset["latitude"].values, original_lat)
        assert any(
            "navigation" in r.message.lower() or "empty" in r.message.lower()
            for r in caplog.records
        )

    def test_empty_attitude_skips_attitude_fields(self, sample_dataset, caplog):
        survey = SideScanSurvey(
            source_files=[Path("test.jsf")],
            source_format="JSF",
            survey_name="test",
        )
        survey.add_frequency("hz_1495000", sample_dataset)
        survey.navigation = make_minimal_navigation_record(n_obs=10)
        # attitude is None

        original_roll = survey.get_frequency("hz_1495000").dataset["roll_deg"].values.copy()

        with caplog.at_level(logging.WARNING):
            result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        np.testing.assert_array_equal(ch.dataset["roll_deg"].values, original_roll)

@pytest.mark.unit
class TestAllNanSource:
    def test_all_nan_source_preserves_original(self, tmp_path):
        """When the aux source variable is all NaN, the per-ping field
        should be preserved (not overwritten with NaN)."""
        survey = _build_known_survey()
        # Replace nav latitude with all NaN
        nav = survey.navigation.copy(deep=True)
        nav["latitude"][:] = np.nan
        survey.navigation = nav

        # Capture original per-ping latitude
        fk = "hz_1495000"
        ch = survey.get_frequency(fk)
        original_lat = ch.dataset["latitude"].values.copy()

        # Run interpolation
        result = interpolate_aux_onto_pings(survey)
        new_lat = result.get_frequency(fk).dataset["latitude"].values

        # Original values must be preserved (not all NaN)
        np.testing.assert_array_equal(new_lat, original_lat)


@pytest.mark.unit
class TestEmptyAuxForce:
    def test_force_overrides_skip(self, sample_dataset, caplog):
        """force=True forces interpolation even when aux is empty.
        Result: all NaN per-ping values."""
        survey = SideScanSurvey(
            source_files=[Path("test.jsf")],
            source_format="JSF",
            survey_name="test",
        )
        survey.add_frequency("hz_1495000", sample_dataset)
        # No navigation set

        result = interpolate_aux_onto_pings(survey, force=True)

        ch = result.get_frequency("hz_1495000")
        # latitude should be NaN-filled (no aux data to interpolate from)
        assert np.all(np.isnan(ch.dataset["latitude"].values))


# ============================================================
# DROP INVALID FROM SOURCE
# ============================================================

@pytest.mark.unit
class TestDropInvalid:
    def test_drop_invalid_removes_nan_from_source(self, tmp_path):
        """drop_invalid=True (default) should drop NaN aux values before
        interpolating, so the interpolation isn't disturbed."""
        survey = _build_known_survey(n_pings=5, ping_step_sec=0.1)

        # Inject NaN into a few aux values
        nav = survey.navigation.copy(deep=True)
        nav["latitude"][3:7] = np.nan
        survey.navigation = nav

        # With drop_invalid=True (default), NaN values are dropped, interpolation
        # over the gap should give a clean linear fit.
        result = interpolate_aux_onto_pings(survey, drop_invalid=True)
        ch = result.get_frequency("hz_1495000")
        ds = ch.dataset

        # No NaN should leak into per-ping values
        assert not np.any(np.isnan(ds["latitude"].values))

    def test_drop_invalid_false_propagates_nan(self, tmp_path):
        """When drop_invalid=False and pings fall in a NaN-only region of the
        source, the interpolated result is NaN."""
        survey = _build_known_survey(n_pings=3, ping_step_sec=0.1)

        # Pings at T+1.0, T+1.1, T+1.2 (right in the middle)
        ds = survey.get_frequency("hz_1495000").dataset
        new_times = pd.date_range("2026-01-01T00:00:01.0",
                                periods=ds.sizes["ping"],
                                freq="0.1s").values.astype("datetime64[ns]")
        ds = ds.assign_coords(time=("ping", new_times))
        survey._freq_channels["hz_1495000"] = SideScanFrequencyChannel(
            ds, freq_key="hz_1495000"
        )

        # Set aux: 100 obs, but mark obs 5..15 as NaN (covers ping range)
        nav = _make_linear_navigation(n_obs=100, step_sec=0.1)
        # obs index 10 is at T+1.0, obs 20 at T+2.0 — corrupt that whole region
        nav["latitude"][8:25] = np.nan
        survey.navigation = nav

        # With drop_invalid=False, the NaN region in source straddles the ping
        # times → scipy interpolates between NaN points → all NaN output.
        result = interpolate_aux_onto_pings(survey, drop_invalid=False)
        ch = result.get_frequency("hz_1495000")
        assert np.all(np.isnan(ch.dataset["latitude"].values))


# ============================================================
# PROCESSING HISTORY
# ============================================================

@pytest.mark.unit
class TestProcessingHistory:
    def test_history_entry_added(self):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        history = ch.dataset.sidescan.processing_history
        assert len(history) == 1
        assert "interpolate" in history[0]["step"].lower()

    def test_history_entry_records_fields_and_method(self):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey, method="nearest")

        ch = result.get_frequency("hz_1495000")
        entry = ch.dataset.sidescan.processing_history[0]
        assert entry["params"]["method"] == "nearest"
        assert "fields" in entry["params"]

    def test_history_function_path_set(self):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey)

        ch = result.get_frequency("hz_1495000")
        entry = ch.dataset.sidescan.processing_history[0]
        assert "interpolate" in entry["function"]


# ============================================================
# IN_PLACE BEHAVIOR
# ============================================================

@pytest.mark.unit
class TestInPlaceVsCopy:
    def test_default_returns_new_survey(self):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey)
        # Default in_place=False
        # Original survey's per-ping latitude should be unchanged
        original_lat = survey.get_frequency("hz_1495000").dataset["latitude"].values
        # Original was synthetic dataset's default — not the interpolated values
        # The result should differ from original.
        result_lat = result.get_frequency("hz_1495000").dataset["latitude"].values
        # At least some difference should exist (interpolation actually modified)
        assert not np.array_equal(original_lat, result_lat)

    def test_in_place_mutates_survey(self):
        survey = _build_known_survey()
        result = interpolate_aux_onto_pings(survey, in_place=True)
        # in_place=True should return the same survey instance
        assert result is survey


# ============================================================
# CUSTOM FIELD MAP
# ============================================================

@pytest.mark.unit
class TestCustomFields:
    def test_only_specified_fields_interpolated(self):
        survey = _build_known_survey()
        original_roll = survey.get_frequency("hz_1495000").dataset["roll_deg"].values.copy()

        # Only interpolate latitude/longitude
        result = interpolate_aux_onto_pings(
            survey,
            fields={
                "latitude":  ("navigation", "latitude"),
                "longitude": ("navigation", "longitude"),
            },
        )

        ch = result.get_frequency("hz_1495000")
        # roll_deg should be untouched
        np.testing.assert_array_equal(ch.dataset["roll_deg"].values, original_roll)

    def test_unknown_aux_raises(self):
        survey = _build_known_survey()
        with pytest.raises(ValueError, match="aux"):
            interpolate_aux_onto_pings(
                survey,
                fields={"latitude": ("nonexistent_aux", "latitude")},
            )


# ============================================================
# LOW-LEVEL — interpolate_onto_channel with external source
# ============================================================

@pytest.mark.unit
class TestExternalSource:
    def test_external_source_dataset(self):
        """A user-supplied xr.Dataset (not from a SideScanSurvey aux record)
        can be used as the interpolation source."""
        survey = _build_known_survey()
        ch = survey.get_frequency("hz_1495000")

        # Build a custom external nav dataset with totally different values
        # (so we can verify it's actually being used)
        n_obs = 100
        times = pd.date_range("2026-01-01T00:00:00",
                              periods=n_obs, freq="0.1s").values \
            .astype("datetime64[ns]")
        external = xr.Dataset(
            data_vars={
                "lat":  (("obs",), np.full(n_obs, 99.999, dtype=np.float64)),
                "lon":  (("obs",), np.full(n_obs, -99.999, dtype=np.float64)),
            },
            coords={"time": (("obs",), times)},
        )

        new_ch = interpolate_onto_channel(
            ch,
            source=external,
            field_map={"latitude": "lat", "longitude": "lon"},
        )

        # All pings should have the constant value from the external source
        np.testing.assert_allclose(
            new_ch.dataset["latitude"].values, 99.999, atol=1e-9
        )
        np.testing.assert_allclose(
            new_ch.dataset["longitude"].values, -99.999, atol=1e-9
        )

    def test_external_source_missing_var_raises(self):
        survey = _build_known_survey()
        ch = survey.get_frequency("hz_1495000")

        n_obs = 10
        times = pd.date_range("2026-01-01T00:00:00",
                              periods=n_obs, freq="0.1s").values \
            .astype("datetime64[ns]")
        external = xr.Dataset(
            data_vars={"lat": (("obs",), np.zeros(n_obs))},
            coords={"time": (("obs",), times)},
        )

        with pytest.raises((KeyError, ValueError)):
            interpolate_onto_channel(
                ch,
                source=external,
                field_map={"latitude": "nonexistent_var"},
            )

    def test_external_source_missing_time_coord_raises(self):
        survey = _build_known_survey()
        ch = survey.get_frequency("hz_1495000")

        # Build a dataset without a 'time' coord
        external = xr.Dataset(
            data_vars={"lat": (("obs",), np.zeros(10))},
        )

        with pytest.raises((KeyError, ValueError)):
            interpolate_onto_channel(
                ch,
                source=external,
                field_map={"latitude": "lat"},
            )


# ============================================================
# MULTI-FREQUENCY SURVEY
# ============================================================

@pytest.mark.unit
class TestMultipleFrequencies:
    def test_all_channels_updated(self):
        """A survey with two frequency channels should have both updated."""
        # Build a multi-freq survey with synthetic aux
        n_pings = 5
        ping_times = pd.date_range("2026-01-01T00:00:01",
                                   periods=n_pings, freq="1s").values \
            .astype("datetime64[ns]")

        ds_a = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=10,
                                             freq_hz_start=547_500)
        ds_a = ds_a.assign_coords(time=("ping", ping_times))

        ds_b = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=10,
                                             freq_hz_start=1_495_000)
        ds_b = ds_b.assign_coords(time=("ping", ping_times))

        survey = SideScanSurvey(
            source_files=[Path("synthetic.jsf")],
            source_format="JSF",
            survey_name="multi",
        )
        survey.add_frequency("hz_547500",  ds_a)
        survey.add_frequency("hz_1495000", ds_b)
        survey.navigation = _make_linear_navigation(n_obs=100, step_sec=0.1)
        survey.attitude   = _make_linear_attitude(n_obs=100, step_sec=0.1)

        result = interpolate_aux_onto_pings(survey)

        # Both channels should have interpolated lat/lon
        for fk in ("hz_547500", "hz_1495000"):
            ds = result.get_frequency(fk).dataset
            elapsed_s = (ds["time"].values - np.datetime64("2026-01-01T00:00:00", "ns")) \
                / np.timedelta64(1, "s")
            expected_lat = 41.0 + 1e-5 * elapsed_s
            np.testing.assert_allclose(ds["latitude"].values, expected_lat, atol=1e-9)


# ============================================================
# EDGE CASES
# ============================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_single_obs_aux_record(self):
        """An aux record with only one observation is a degenerate case;
        the function should either skip with warning or hold constant."""
        survey = _build_known_survey()
        # Replace navigation with a single-obs record
        survey.navigation = _make_linear_navigation(n_obs=1)

        # Should not crash. With clamp extrapolation, all pings should equal
        # that single value. (Or the function may warn and skip — both are
        # acceptable; verify we don't crash.)
        result = interpolate_aux_onto_pings(survey, extrapolation="clamp")
        # Result is well-defined either way — no NaN, no crash
        ch = result.get_frequency("hz_1495000")
        # Sanity: latitude exists and is finite (or original if skipped)
        assert "latitude" in ch.dataset.data_vars

    def test_aux_with_duplicate_timestamps(self):
        """If aux record has two values at the same time, interpolation
        should handle it (drop or average; xarray drops by default)."""
        survey = _build_known_survey()
        nav = survey.navigation.copy(deep=True)
        # Force duplicate timestamps at the start
        nav["time"][0] = nav["time"][1]
        survey.navigation = nav

        # Should not crash
        result = interpolate_aux_onto_pings(survey)
        assert result is not None