"""
Unit tests for sidescanpy.core.survey.

Tests SideScanFrequencyChannel and SideScanSurvey including aux record
management (navigation, attitude, dvl, sensors).

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.survey import (
    SideScanFrequencyChannel,
    SideScanSurvey,
)

# ============================================================
# SideScanFrequencyChannel
# ============================================================

@pytest.mark.unit
class TestFrequencyChannelInit:
    def test_initialization_valid(self, sample_dataset):
        ch = SideScanFrequencyChannel(sample_dataset, freq_key="freq_test")
        assert ch.freq_key == "freq_test"
        assert ch.dataset is sample_dataset

    def test_default_freq_key_derived_from_attrs(self, sample_dataset):
        ch = SideScanFrequencyChannel(sample_dataset)
        assert ch.freq_key == f"hz_{sample_dataset.attrs['freq_hz_start']}"

    def test_invalid_dataset_missing_dim_raises(self, sample_dataset):
        bad = sample_dataset.drop_dims("channel")
        with pytest.raises(ValueError, match="Missing dimensions"):
            SideScanFrequencyChannel(bad)

    @pytest.mark.parametrize(
        "missing_var",
        ["latitude", "longitude", "heading_deg", "depth_m",
         "roll_deg", "pitch_deg", "heave_m", "altitude_m",
         "sound_vel_ms", "backscatter_raw", "backscatter_intensity"],
    )
    def test_missing_required_variable_raises(self, sample_dataset, missing_var):
        bad = sample_dataset.drop_vars(missing_var)
        with pytest.raises(ValueError, match="Missing variables"):
            SideScanFrequencyChannel(bad)

    @pytest.mark.parametrize(
        "missing_attr",
        ["freq_hz_start", "freq_hz_end", "pulse_length_us",
         "sample_rate_hz", "processing_level", "processing_history"],
    )
    def test_missing_required_attr_raises(self, sample_dataset, missing_attr):
        bad = sample_dataset.copy()
        del bad.attrs[missing_attr]
        with pytest.raises(ValueError, match="Missing attributes"):
            SideScanFrequencyChannel(bad)


@pytest.mark.unit
class TestFrequencyChannelProperties:
    def test_frequency(self, sample_survey_channel):
        assert sample_survey_channel.frequency == 1_495_000

    def test_n_pings(self, sample_survey_channel):
        assert sample_survey_channel.n_pings == 10

    def test_n_channels(self, sample_survey_channel):
        assert sample_survey_channel.n_channels == 2

    def test_n_samples(self, sample_survey_channel):
        assert sample_survey_channel.n_samples == 100

    def test_dataset_property_returns_original(self, sample_dataset):
        ch = SideScanFrequencyChannel(sample_dataset)
        assert ch.dataset is sample_dataset


@pytest.mark.unit
class TestFrequencyChannelRepr:
    def test_repr_contains_freq_key(self, sample_survey_channel):
        r = repr(sample_survey_channel)
        assert "SideScanFrequencyChannel" in r
        assert sample_survey_channel.freq_key in r

    def test_repr_contains_counts(self, sample_survey_channel):
        r = repr(sample_survey_channel)
        assert f"pings={sample_survey_channel.n_pings}"     in r
        assert f"channels={sample_survey_channel.n_channels}" in r
        assert f"samples={sample_survey_channel.n_samples}"   in r


# ============================================================
# SideScanSurvey — construction + frequency channels
# ============================================================

@pytest.mark.unit
class TestSurveyInit:
    def test_initialization(self, tmp_dir):
        f = tmp_dir / "test.jsf"
        f.touch()

        survey = SideScanSurvey(
            source_files=[f],
            source_format="JSF",
            survey_name="test_survey",
        )
        assert survey.source_files == [f]
        assert survey.source_format == "JSF"
        assert survey.survey_name == "test_survey"
        assert survey.n_frequencies == 0
        assert survey.n_sensors == 0

    def test_aux_records_start_unset(self, tmp_dir):
        survey = SideScanSurvey(
            source_files=[tmp_dir / "test.jsf"],
            source_format="JSF",
            survey_name="t",
        )
        assert survey.navigation is None
        assert survey.attitude   is None
        assert survey.dvl        is None
        assert survey.sensors    == {}

    def test_has_aux_returns_false_when_unset(self, tmp_dir):
        survey = SideScanSurvey(
            source_files=[tmp_dir / "test.jsf"],
            source_format="JSF",
            survey_name="t",
        )
        assert survey.has_navigation is False
        assert survey.has_attitude   is False
        assert survey.has_dvl        is False


@pytest.mark.unit
class TestSurveyFrequencyChannels:
    def test_add_frequency(self, sample_multi_frequency_survey):
        assert sample_multi_frequency_survey.n_frequencies == 2
        assert "hz_1495000" in sample_multi_frequency_survey.frequencies
        assert "hz_547500"  in sample_multi_frequency_survey.frequencies

    def test_get_frequency(self, sample_multi_frequency_survey):
        ch = sample_multi_frequency_survey.get_frequency("hz_1495000")
        assert isinstance(ch, SideScanFrequencyChannel)
        assert ch.freq_key == "hz_1495000"

    def test_get_unknown_frequency_raises(self, sample_multi_frequency_survey):
        with pytest.raises(KeyError, match="not found"):
            sample_multi_frequency_survey.get_frequency("freq_does_not_exist")

    def test_freq_channels_returns_dict_copy(self, sample_multi_frequency_survey):
        d = sample_multi_frequency_survey.freq_channels
        d["freq_999_hz"] = "tampering"
        # Original dict should be unchanged
        assert "freq_999_hz" not in sample_multi_frequency_survey.frequencies

    def test_frequency_hz_list_sorted(self, sample_multi_frequency_survey):
        freqs = sample_multi_frequency_survey.frequency_hz_list
        assert freqs == sorted(freqs)
        assert 547_500 in freqs
        assert 1_495_000 in freqs

    def test_add_frequency_overwrite_logs(
        self, sample_multi_frequency_survey, sample_dataset, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            sample_multi_frequency_survey.add_frequency(
                "hz_1495000", sample_dataset
            )
        assert any("Overwriting" in rec.message for rec in caplog.records)
        assert sample_multi_frequency_survey.n_frequencies == 2  # not 3


# ============================================================
# SideScanSurvey — navigation
# ============================================================

@pytest.mark.unit
class TestSurveyNavigation:
    def test_set_navigation(self, sample_multi_frequency_survey, sample_navigation):
        sample_multi_frequency_survey.navigation = sample_navigation
        assert sample_multi_frequency_survey.navigation is sample_navigation

    def test_has_navigation_when_set(
        self, sample_multi_frequency_survey, sample_navigation
    ):
        sample_multi_frequency_survey.navigation = sample_navigation
        assert sample_multi_frequency_survey.has_navigation is True

    def test_has_navigation_false_when_empty(
        self, sample_multi_frequency_survey, empty_navigation
    ):
        sample_multi_frequency_survey.navigation = empty_navigation
        assert sample_multi_frequency_survey.has_navigation is False

    def test_set_navigation_to_none_clears(
        self, sample_multi_frequency_survey, sample_navigation
    ):
        sample_multi_frequency_survey.navigation = sample_navigation
        sample_multi_frequency_survey.navigation = None
        assert sample_multi_frequency_survey.navigation is None
        assert sample_multi_frequency_survey.has_navigation is False

    def test_set_invalid_navigation_raises(self, sample_multi_frequency_survey):
        bad = xr.Dataset()  # missing everything
        with pytest.raises(ValueError, match="Navigation"):
            sample_multi_frequency_survey.navigation = bad

    def test_overwrite_navigation_logs(
        self, sample_multi_frequency_survey, sample_navigation, caplog
    ):
        sample_multi_frequency_survey.navigation = sample_navigation
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            sample_multi_frequency_survey.navigation = sample_navigation
        assert any("Overwriting" in rec.message for rec in caplog.records)


# ============================================================
# SideScanSurvey — attitude
# ============================================================

@pytest.mark.unit
class TestSurveyAttitude:
    def test_set_attitude(self, sample_multi_frequency_survey, sample_attitude):
        sample_multi_frequency_survey.attitude = sample_attitude
        assert sample_multi_frequency_survey.attitude is sample_attitude

    def test_has_attitude_when_set(
        self, sample_multi_frequency_survey, sample_attitude
    ):
        sample_multi_frequency_survey.attitude = sample_attitude
        assert sample_multi_frequency_survey.has_attitude is True

    def test_has_attitude_false_when_empty(
        self, sample_multi_frequency_survey, empty_attitude
    ):
        sample_multi_frequency_survey.attitude = empty_attitude
        assert sample_multi_frequency_survey.has_attitude is False

    def test_set_attitude_to_none_clears(
        self, sample_multi_frequency_survey, sample_attitude
    ):
        sample_multi_frequency_survey.attitude = sample_attitude
        sample_multi_frequency_survey.attitude = None
        assert sample_multi_frequency_survey.attitude is None
        assert sample_multi_frequency_survey.has_attitude is False

    def test_set_invalid_attitude_raises(self, sample_multi_frequency_survey):
        bad = xr.Dataset()
        with pytest.raises(ValueError, match="Attitude"):
            sample_multi_frequency_survey.attitude = bad

    def test_overwrite_attitude_logs(
        self, sample_multi_frequency_survey, sample_attitude, caplog
    ):
        sample_multi_frequency_survey.attitude = sample_attitude
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            sample_multi_frequency_survey.attitude = sample_attitude
        assert any("Overwriting" in rec.message for rec in caplog.records)


# ============================================================
# SideScanSurvey — dvl
# ============================================================

@pytest.mark.unit
class TestSurveyDVL:
    def test_set_dvl(self, sample_multi_frequency_survey, sample_dvl):
        sample_multi_frequency_survey.dvl = sample_dvl
        assert sample_multi_frequency_survey.dvl is sample_dvl

    def test_has_dvl_when_set(self, sample_multi_frequency_survey, sample_dvl):
        sample_multi_frequency_survey.dvl = sample_dvl
        assert sample_multi_frequency_survey.has_dvl is True

    def test_has_dvl_false_when_empty(
        self, sample_multi_frequency_survey, empty_dvl
    ):
        sample_multi_frequency_survey.dvl = empty_dvl
        assert sample_multi_frequency_survey.has_dvl is False

    def test_set_dvl_to_none_clears(
        self, sample_multi_frequency_survey, sample_dvl
    ):
        sample_multi_frequency_survey.dvl = sample_dvl
        sample_multi_frequency_survey.dvl = None
        assert sample_multi_frequency_survey.dvl is None
        assert sample_multi_frequency_survey.has_dvl is False

    def test_set_invalid_dvl_raises(self, sample_multi_frequency_survey):
        bad = xr.Dataset()
        with pytest.raises(ValueError, match="DVL"):
            sample_multi_frequency_survey.dvl = bad

    def test_overwrite_dvl_logs(
        self, sample_multi_frequency_survey, sample_dvl, caplog
    ):
        sample_multi_frequency_survey.dvl = sample_dvl
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            sample_multi_frequency_survey.dvl = sample_dvl
        assert any("Overwriting" in rec.message for rec in caplog.records)


# ============================================================
# SideScanSurvey — sensors
# ============================================================

@pytest.mark.unit
class TestSurveySensors:
    def test_add_sensor(self, sample_multi_frequency_survey, sample_sensor):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        assert sample_multi_frequency_survey.n_sensors == 1
        assert "pressure" in sample_multi_frequency_survey.sensor_types

    def test_get_sensor(self, sample_multi_frequency_survey, sample_sensor):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        ds = sample_multi_frequency_survey.get_sensor("pressure")
        assert ds is sample_sensor

    def test_get_unknown_sensor_raises(self, sample_multi_frequency_survey):
        with pytest.raises(KeyError, match="not found"):
            sample_multi_frequency_survey.get_sensor("does_not_exist")

    def test_sensors_returns_dict_copy(
        self, sample_multi_frequency_survey, sample_sensor
    ):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        d = sample_multi_frequency_survey.sensors
        d["bogus"] = "tampering"
        assert "bogus" not in sample_multi_frequency_survey.sensor_types

    def test_sensor_types_sorted(
        self, sample_multi_frequency_survey, sample_sensor
    ):
        from tests.conftest import make_minimal_sensor_record
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        sample_multi_frequency_survey.add_sensor(
            "temperature",
            make_minimal_sensor_record(sensor_type="temperature", var_name="temp_c"),
        )
        assert sample_multi_frequency_survey.sensor_types == ["pressure", "temperature"]

    def test_add_invalid_sensor_raises(self, sample_multi_frequency_survey):
        bad = xr.Dataset()
        with pytest.raises(ValueError, match="Sensor"):
            sample_multi_frequency_survey.add_sensor("pressure", bad)

    def test_overwrite_sensor_logs(
        self, sample_multi_frequency_survey, sample_sensor, caplog
    ):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        assert any("Overwriting" in rec.message for rec in caplog.records)
        assert sample_multi_frequency_survey.n_sensors == 1


# ============================================================
# SideScanSurvey — derived properties
# ============================================================

@pytest.mark.unit
class TestSurveyTimeRange:
    def test_time_range_from_freq_channels(self, sample_multi_frequency_survey):
        tr = sample_multi_frequency_survey.time_range
        assert tr is not None
        t_min, t_max = tr
        assert t_min < t_max

    def test_time_range_none_when_empty(self, tmp_dir):
        survey = SideScanSurvey(
            source_files=[tmp_dir / "test.jsf"],
            source_format="JSF",
            survey_name="empty",
        )
        assert survey.time_range is None

    def test_time_range_includes_aux(
        self, sample_multi_frequency_survey, sample_navigation
    ):
        # Build a nav stream with timestamps far outside the freq channels
        far_future = np.datetime64("2099-01-01T00:00:00", "ns") + \
                     np.arange(5) * np.timedelta64(1, "s")
        ds = sample_navigation.copy()
        ds = ds.isel(obs=slice(0, 5))
        ds = ds.assign_coords(time=("obs", far_future))
        sample_multi_frequency_survey.navigation = ds

        _, t_max = sample_multi_frequency_survey.time_range
        assert t_max == far_future.max()


@pytest.mark.unit
class TestSurveyBoundingBox:
    def test_bbox_from_freq_channels(self, sample_multi_frequency_survey):
        bbox = sample_multi_frequency_survey.bounding_box
        assert bbox is not None
        lat_min, lat_max, lon_min, lon_max = bbox
        assert lat_min <= lat_max
        assert lon_min <= lon_max

    def test_bbox_prefers_navigation(
        self, sample_multi_frequency_survey, sample_navigation
    ):
        # Make navigation cover a clearly different range than per-ping coords
        ds = sample_navigation.copy()
        ds["latitude"][:]  = np.linspace(60.0, 60.5, ds.sizes["obs"])
        ds["longitude"][:] = np.linspace(10.0, 10.5, ds.sizes["obs"])
        sample_multi_frequency_survey.navigation = ds

        lat_min, lat_max, lon_min, lon_max = sample_multi_frequency_survey.bounding_box
        assert lat_min >= 60.0
        assert lat_max <= 60.5
        assert lon_min >= 10.0
        assert lon_max <= 10.5

    def test_bbox_falls_back_when_nav_empty(
        self, sample_multi_frequency_survey, empty_navigation
    ):
        sample_multi_frequency_survey.navigation = empty_navigation
        bbox = sample_multi_frequency_survey.bounding_box
        # Should still produce a bbox from per-ping nav
        assert bbox is not None

    def test_bbox_none_when_empty(self, tmp_dir):
        survey = SideScanSurvey(
            source_files=[tmp_dir / "test.jsf"],
            source_format="JSF",
            survey_name="empty",
        )
        assert survey.bounding_box is None


# ============================================================
# SideScanSurvey — repr
# ============================================================

@pytest.mark.unit
class TestSurveyRepr:
    def test_repr_contains_name(self, sample_multi_frequency_survey):
        r = repr(sample_multi_frequency_survey)
        assert "SideScanSurvey" in r
        assert "sample_survey" in r

    def test_repr_shows_n_frequencies(self, sample_multi_frequency_survey):
        r = repr(sample_multi_frequency_survey)
        assert "n_frequencies=2" in r

    def test_repr_shows_aux_dashes_when_unset(self, sample_multi_frequency_survey):
        r = repr(sample_multi_frequency_survey)
        assert "nav=-" in r
        assert "att=-" in r
        assert "dvl=-" in r

    def test_repr_shows_aux_counts_when_set(
        self, sample_multi_frequency_survey, sample_navigation
    ):
        sample_multi_frequency_survey.navigation = sample_navigation
        r = repr(sample_multi_frequency_survey)
        assert "nav=10" in r

    def test_repr_shows_sensor_count(
        self, sample_multi_frequency_survey, sample_sensor
    ):
        sample_multi_frequency_survey.add_sensor("pressure", sample_sensor)
        r = repr(sample_multi_frequency_survey)
        assert "sensors=1" in r
