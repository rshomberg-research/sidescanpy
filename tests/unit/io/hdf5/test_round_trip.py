"""
Round-trip identity tests for sidescanpy.io.hdf5.

These tests prove that write+read produces a survey numerically identical to
the original — frequency channels, aux records, attrs, and metadata all
survive serialization. This is the load-bearing guarantee of the HDF5 module.

2026 rshomberg@uri.edu
"""


import numpy as np
import pytest
import xarray as xr

from sidescanpy.io.hdf5.reader import read_survey
from sidescanpy.io.hdf5.writer import write_survey

# ============================================================
# FULL SURVEY ROUND-TRIP
# ============================================================

@pytest.mark.unit
class TestRoundTripFullSurvey:
    def test_frequency_datasets_identical(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)

        for fk in sample_full_survey.frequencies:
            a = sample_full_survey.get_frequency(fk).dataset
            b = loaded.get_frequency(fk).dataset
            xr.testing.assert_identical(a, b)

    def test_navigation_identical(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        xr.testing.assert_identical(
            loaded.navigation, sample_full_survey.navigation
        )

    def test_attitude_identical(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        xr.testing.assert_identical(
            loaded.attitude, sample_full_survey.attitude
        )

    def test_dvl_identical(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        xr.testing.assert_identical(
            loaded.dvl, sample_full_survey.dvl
        )

    def test_sensor_records_identical(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)

        assert sorted(loaded.sensor_types) == sorted(sample_full_survey.sensor_types)
        for sensor_type in sample_full_survey.sensor_types:
            xr.testing.assert_identical(
                loaded.get_sensor(sensor_type),
                sample_full_survey.get_sensor(sensor_type),
            )


# ============================================================
# SURVEY WITHOUT AUX RECORDS
# ============================================================

@pytest.mark.unit
class TestRoundTripNoAux:
    def test_frequencies_round_trip(self, sample_multi_frequency_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        for fk in sample_multi_frequency_survey.frequencies:
            a = sample_multi_frequency_survey.get_frequency(fk).dataset
            b = loaded.get_frequency(fk).dataset
            xr.testing.assert_identical(a, b)

    def test_aux_records_loaded_as_empty(self, sample_multi_frequency_survey, tmp_path):
        """Survey written without aux records should still produce empty-but-
        valid aux records on read (since writer always writes empty datasets
        for navigation/attitude/dvl)."""
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        # Aux records exist but are empty
        assert loaded.navigation is not None
        assert loaded.attitude   is not None
        assert loaded.dvl        is not None
        assert loaded.has_navigation is False
        assert loaded.has_attitude   is False
        assert loaded.has_dvl        is False

    def test_no_sensors(self, sample_multi_frequency_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)
        assert loaded.n_sensors == 0
        assert loaded.sensor_types == []


# ============================================================
# PARTIAL SURVEY (some aux populated, others not)
# ============================================================

@pytest.mark.unit
class TestRoundTripPartialAux:
    def test_only_navigation_set(
        self, sample_multi_frequency_survey, sample_navigation, tmp_path
    ):
        sample_multi_frequency_survey.navigation = sample_navigation
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        assert loaded.has_navigation is True
        assert loaded.has_attitude   is False
        assert loaded.has_dvl        is False
        xr.testing.assert_identical(loaded.navigation, sample_navigation)

    def test_only_attitude_set(
        self, sample_multi_frequency_survey, sample_attitude, tmp_path
    ):
        sample_multi_frequency_survey.attitude = sample_attitude
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        assert loaded.has_navigation is False
        assert loaded.has_attitude   is True
        assert loaded.has_dvl        is False
        xr.testing.assert_identical(loaded.attitude, sample_attitude)

    def test_only_dvl_set(
        self, sample_multi_frequency_survey, sample_dvl, tmp_path
    ):
        sample_multi_frequency_survey.dvl = sample_dvl
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        assert loaded.has_navigation is False
        assert loaded.has_attitude   is False
        assert loaded.has_dvl        is True
        xr.testing.assert_identical(loaded.dvl, sample_dvl)

    def test_only_sensor_set(
        self, sample_multi_frequency_survey, sample_sensor, tmp_path
    ):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        loaded = read_survey(p)

        assert loaded.n_sensors == 1
        assert "pressure" in loaded.sensor_types
        xr.testing.assert_identical(
            loaded.get_sensor("pressure"), sample_sensor
        )


# ============================================================
# DTYPE PRESERVATION
# ============================================================

@pytest.mark.unit
class TestRoundTripDtypes:
    def test_time_coord_is_datetime64_ns_per_frequency(
        self, sample_full_survey, tmp_path
    ):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        for fk in loaded.frequencies:
            assert loaded.get_frequency(fk).dataset["time"].dtype == \
                np.dtype("datetime64[ns]")

    def test_time_coord_is_datetime64_ns_in_aux(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded.navigation["time"].dtype == np.dtype("datetime64[ns]")
        assert loaded.attitude["time"].dtype   == np.dtype("datetime64[ns]")
        assert loaded.dvl["time"].dtype        == np.dtype("datetime64[ns]")

    def test_channel_id_returns_strings_not_bytes(
        self, sample_full_survey, tmp_path
    ):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        fk = loaded.frequencies[0]
        ids = list(loaded.get_frequency(fk).dataset.channel_id.values)
        assert ids == ["port", "starboard"]
        for v in ids:
            assert isinstance(v, str)

    def test_channel_id_selectable_by_name(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        fk = loaded.frequencies[0]
        ds = loaded.get_frequency(fk).dataset
        port = ds["backscatter_intensity"].sel(channel_id="port")
        stbd = ds["backscatter_intensity"].sel(channel_id="starboard")
        assert port.shape == (loaded.get_frequency(fk).n_pings,
                              loaded.get_frequency(fk).n_samples)
        assert stbd.shape == port.shape


# ============================================================
# NAN PRESERVATION (heave_m is NaN-filled by the converter)
# ============================================================

@pytest.mark.unit
class TestRoundTripNaN:
    def test_nan_preserved_in_per_ping_field(
        self, sample_full_survey, tmp_path
    ):
        # Force NaN into heave_m to test round-trip
        from sidescanpy.core.survey import SideScanFrequencyChannel

        # Mutate the dataset of one frequency channel
        fk = sample_full_survey.frequencies[0]
        ch = sample_full_survey.get_frequency(fk)
        ds = ch.dataset.copy(deep=True)
        ds["heave_m"][:] = np.nan

        # Wrap as a fresh channel and replace in the survey
        fresh_channel = SideScanFrequencyChannel(ds, freq_key=fk)
        sample_full_survey._freq_channels[fk] = fresh_channel   # FIXME: private attribute

        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)

        loaded_heave = loaded.get_frequency(fk).dataset["heave_m"].values
        assert np.all(np.isnan(loaded_heave))


# ============================================================
# ATTRS PRESERVATION
# ============================================================

@pytest.mark.unit
class TestRoundTripAttrs:
    def test_per_frequency_attrs_preserved(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)

        for fk in sample_full_survey.frequencies:
            orig = sample_full_survey.get_frequency(fk).dataset.attrs
            new  = loaded.get_frequency(fk).dataset.attrs
            for key in (
                "freq_hz_start", "freq_hz_end",
                "pulse_length_us", "sample_rate_hz",
                "processing_level", "processing_history",
            ):
                assert new[key] == orig[key]

    def test_processing_history_parses_after_load(
        self, sample_full_survey, tmp_path
    ):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        for fk in loaded.frequencies:
            history = loaded.get_frequency(fk).dataset.sidescan.processing_history
            assert isinstance(history, list)

    def test_aux_source_attrs_preserved(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded.navigation.attrs["source"] == \
               sample_full_survey.navigation.attrs["source"]
        assert loaded.attitude.attrs["source"] == \
               sample_full_survey.attitude.attrs["source"]
        assert loaded.dvl.attrs["source"] == \
               sample_full_survey.dvl.attrs["source"]
