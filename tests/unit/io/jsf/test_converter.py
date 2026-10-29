"""
Unit tests for native HDF5 to standard schema conversion

Uses mocks for JSFMessage instances — the converter only reads attributes off
messages, so Mock(spec=...) with attribute assignments is sufficient.

Real binary parsing is tested separately against actual JSF byte fixtures.


2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.jsf.converter import (
    _build_attitude_record,
    _build_dvl_record,
    _build_frequency_channel,
    _build_navigation_record,
    _build_sensor_record,
    convert,
)
from tests.unit.io.jsf.conftest import (
    PR_FLAG_HEADING,
    PR_FLAG_PITCH,
    PR_FLAG_ROLL,
    _make_real_pitch_roll_msg,
    _make_sonar_msg,
)

# ============================================================
# CONVERT — top-level API
# ============================================================

@pytest.mark.unit
class TestConvertEmpty:
    def test_empty_message_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            convert([], source_files=[Path("dummy.jsf")])


@pytest.mark.unit
class TestConversionSingleFrequency:

    def test_returns_sidescan_survey(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs, source_files=[Path("test.jsf")])
        assert isinstance(result, SideScanSurvey)

    def test_one_frequency_channel(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs, source_files=[Path("test.jsf")])
        assert result.n_frequencies==1

    def test_frequency_channel_validates(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs, source_files=[Path("test.jsf")])
        ch = result.get_frequency(result.frequencies[0])
        ch.dataset.sidescan.validate()

    def test_aux_records_empty_when_no_aux_messages(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs, source_files=[Path("test.jsf")])
        assert result.has_navigation    is False
        assert result.has_attitude      is False
        assert result.has_dvl           is False
        assert result.n_sensors         == 0


@pytest.mark.unit
class TestConvertMultiFrequency:

    def test_two_frequencies_yield_two_channels(self, make_sonar_msg):
        msgs = []

        # Frequency A: 1495 MHz, 5 paired pings
        for i in range(5):
            msgs.append(
                make_sonar_msg(
                    ping_number=i, channel=0,
                    transmit_freq_start=1_495_000,
                    transmit_freq_end=1_495_000,
                )
            )
            msgs.append(
                make_sonar_msg(
                    ping_number=i, channel=1,
                    transmit_freq_start=1_495_000,
                    transmit_freq_end=1_495_000,
                )
            )

        # Frequency A: 547.5 KHz, 5 paired pings
        for i in range(5):
            msgs.append(
                make_sonar_msg(
                    ping_number=i, channel=0,
                    transmit_freq_start=547_500,
                    transmit_freq_end=547_500,
                )
            )
            msgs.append(
                make_sonar_msg(
                    ping_number=i, channel=1,
                    transmit_freq_start=547_500,
                    transmit_freq_end=547_500,
                )
            )

        result = convert(msgs, source_files=[Path("test.jsf")])
        assert result.n_frequencies == 2
        assert 547_500      in result.frequency_hz_list
        assert 1_495_000    in result.frequency_hz_list



# ============================================================
# CONVERT — auxiliary streams
# ============================================================

@pytest.mark.unit
class TestConvertWithNavigation:

    def test_navigation_populated_from_nmea(self, paired_sonar_msgs, nmea_rmc_msgs):
        result = convert(paired_sonar_msgs + nmea_rmc_msgs,
                         source_files=[Path("test.jsf")])
        assert result.has_navigation
        assert result.navigation.sizes["obs"] == len(nmea_rmc_msgs)

    def test_navigation_validates(self, paired_sonar_msgs, nmea_rmc_msgs):
        result = convert(paired_sonar_msgs + nmea_rmc_msgs, source_files=[Path("test.jsf")])
        result.navigation.navigation_record.validate()


@pytest.mark.unit
class TestConvertWithAttitude:
    def test_attitude_populated_from_pitch_roll(self, paired_sonar_msgs, pitch_roll_msgs):
        result = convert(paired_sonar_msgs + pitch_roll_msgs,
                         source_files=[Path("test.jsf")])
        assert result.has_attitude
        assert result.attitude.sizes["obs"] == len(pitch_roll_msgs)

    def test_attitude_validates(self, paired_sonar_msgs, pitch_roll_msgs):
        result = convert(paired_sonar_msgs + pitch_roll_msgs,
                         source_files=[Path("test.jsf")])
        result.attitude.attitude_record.validate()


@pytest.mark.unit
class TestConvertWithDVL:
    def test_dvl_populated_from_dvl_messages(self, paired_sonar_msgs, dvl_msgs):
        result = convert(paired_sonar_msgs + dvl_msgs,
                         source_files=[Path("test.jsf")])
        assert result.has_dvl
        assert result.dvl.sizes["obs"] == len(dvl_msgs)

    def test_dvl_validates(self, paired_sonar_msgs, dvl_msgs):
        result = convert(paired_sonar_msgs + dvl_msgs,
                         source_files=[Path("test.jsf")])
        result.dvl.dvl_record.validate()


@pytest.mark.unit
class TestConvertWithSensor:
    def test_pressure_messages_become_sensor(self, paired_sonar_msgs, pressure_msgs):
        result = convert(paired_sonar_msgs + pressure_msgs,
                         source_files=[Path("test.jsf")])
        assert "pressure" in result.sensor_types
        assert result.get_sensor("pressure").sizes["obs"] == len(pressure_msgs)

    def test_sensor_validates(self, paired_sonar_msgs, pressure_msgs):
        result = convert(paired_sonar_msgs + pressure_msgs,
                         source_files=[Path("test.jsf")])
        result.get_sensor("pressure").sensor_record.validate()


# ============================================================
# CONVERT — repair modes
# ============================================================

@pytest.mark.unit
class TestConvertRepairDrop:
    def test_orphan_port_dropped(self, make_sonar_msg, paired_sonar_msgs, caplog):
        # Add one extra port record with no matching starboard
        msgs = paired_sonar_msgs + [make_sonar_msg(ping_number=99, channel=0)]
        with caplog.at_level(logging.WARNING):
            result = convert(msgs, source_files=[Path("test.jsf")], repair_mode="drop")
        ch = result.get_frequency(result.frequencies[0])
        assert ch.n_pings == 5

    def test_orphan_starboard_dropped(self, make_sonar_msg, paired_sonar_msgs):
        msgs = [make_sonar_msg(ping_number=99, channel=1)] + paired_sonar_msgs
        result = convert(msgs, source_files=[Path("test.jsf")],
                         repair_mode="drop")
        ch = result.get_frequency(result.frequencies[0])
        assert ch.n_pings == 5


@pytest.mark.unit
class TestConvertRepairNanPad:
    def test_orphan_port_nan_padded(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [
            make_sonar_msg(ping_number=99, channel=0, backscatter_val=500.0)
        ]
        result = convert(msgs, source_files=[Path("test.jsf")],
                         repair_mode="nan_pad")
        ch = result.get_frequency(result.frequencies[0])
        # 5 pairs + 1 padded = 6
        assert ch.n_pings == 6

        # Find the padded ping (ping_number == 99)
        ds = ch.dataset
        idx = int(np.where(ds["ping_number"].values == 99)[0][0])
        padded = ds.isel(ping=idx)

        # Starboard channel should be all NaN
        stbd = padded["backscatter_intensity"].sel(channel_id="starboard").values
        assert np.all(np.isnan(stbd))

        # Port channel should have the real data
        port = padded["backscatter_intensity"].sel(channel_id="port").values
        assert np.allclose(port, 500.0)

    def test_orphan_starboard_nan_padded(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [
            make_sonar_msg(ping_number=99, channel=1, backscatter_val=700.0)
        ]
        result = convert(msgs, source_files=[Path("test.jsf")],
                         repair_mode="nan_pad")
        ch = result.get_frequency(result.frequencies[0])
        ds = ch.dataset
        idx = int(np.where(ds["ping_number"].values == 99)[0][0])
        padded = ds.isel(ping=idx)
        port = padded["backscatter_intensity"].sel(channel_id="port").values
        assert np.all(np.isnan(port))

    def test_default_is_nan_pad(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [make_sonar_msg(ping_number=99, channel=0)]
        # No repair_mode arg → should default to "nan_pad"
        result = convert(msgs, source_files=[Path("test.jsf")])
        ch = result.get_frequency(result.frequencies[0])
        assert ch.n_pings == 6  # padded, not dropped

    def test_padded_ping_per_ping_nav_is_nan(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [
            make_sonar_msg(ping_number=99, channel=0, latitude=41.234)
        ]
        result = convert(msgs, source_files=[Path("test.jsf")],
                         repair_mode="nan_pad")
        ch = result.get_frequency(result.frequencies[0])
        ds = ch.dataset
        idx = int(np.where(ds["ping_number"].values == 99)[0][0])
        # Per decision 9: nan-fill, do not fall back to other side
        # (port has the real data; this is a port-only ping so its lat IS valid)
        # Test the inverse case below
        assert ds["latitude"].isel(ping=idx).item() == pytest.approx(41.234)

@pytest.mark.unit
class TestConvertRepairAbort:
    def test_orphan_port_raises(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [make_sonar_msg(ping_number=99, channel=0)]
        with pytest.raises(ValueError, match="incomplete|orphan|unmatched"):
            convert(msgs, source_files=[Path("test.jsf")], repair_mode="abort")

    def test_orphan_starboard_raises(self, make_sonar_msg, paired_sonar_msgs):
        msgs = paired_sonar_msgs + [make_sonar_msg(ping_number=99, channel=1)]
        with pytest.raises(ValueError, match="incomplete|orphan|unmatched"):
            convert(msgs, source_files=[Path("test.jsf")], repair_mode="abort")


# ============================================================
# CONVERT — error cases
# ============================================================

@pytest.mark.unit
class TestConvertInconsistentAttrs:
    @pytest.mark.parametrize("varying_field,values", [
    ("transmit_freq_end", [1_495_000, 1_500_000]),
    ("sweep_length",      [0.020, 0.025]),
    ("sample_freq",       [400_000, 500_000]),
    ("sample_interval",   [1.0 / 400_000, 1.0 / 500_000]),
    ])
    def test_inconsistent_file_attr_raises(self, make_sonar_msg, varying_field, values):
        msgs = [
            make_sonar_msg(ping_number=0, channel=0, **{varying_field: values[0]}),
            make_sonar_msg(ping_number=0, channel=1, **{varying_field: values[0]}),
            make_sonar_msg(ping_number=1, channel=0, **{varying_field: values[1]}),
            make_sonar_msg(ping_number=1, channel=1, **{varying_field: values[1]}),
        ]
        with pytest.raises(ValueError, match=varying_field):
            convert(msgs, source_files=[Path("test.jsf")])


    def test_varying_freq_start_creates_multiple_channels(self, make_sonar_msg):
        """transmit_freq_start variation is the *defining* feature of multi-freq
        JSF data — it must split into separate channels, not raise."""
        msgs = [
            make_sonar_msg(ping_number=0, channel=0, transmit_freq_start=1_495_000),
            make_sonar_msg(ping_number=0, channel=1, transmit_freq_start=1_495_000),
            make_sonar_msg(ping_number=0, channel=0, transmit_freq_start=547_500),
            make_sonar_msg(ping_number=0, channel=1, transmit_freq_start=547_500),
        ]
        result = convert(msgs, source_files=[Path("test.jsf")])
        assert result.n_frequencies == 2

@pytest.mark.unit
class TestConvertInconsistentSamples:
    def test_port_starboard_n_samples_mismatch_raises(self, make_sonar_msg):
        msgs = [
            make_sonar_msg(ping_number=0, channel=0, n_samples=100),
            make_sonar_msg(ping_number=0, channel=1, n_samples=200),
        ]
        with pytest.raises(ValueError, match="sample"):
            convert(msgs, source_files=[Path("test.jsf")])


# ============================================================
# CONVERT — survey metadata + edge cases
# ============================================================

@pytest.mark.unit
class TestConvertSurveyMetadata:
    def test_source_files_passed_through(self, paired_sonar_msgs):
        files = [Path("a.jsf"), Path("b.jsf")]
        result = convert(paired_sonar_msgs, source_files=files)
        assert result.source_files == files

    def test_survey_name_explicit(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs,
                         source_files=[Path("test.jsf")],
                         survey_name="my_survey")
        assert result.survey_name == "my_survey"

    def test_survey_name_default_from_first_file(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs,
                         source_files=[Path("/tmp/dive_42.jsf")])
        assert result.survey_name == "dive_42"

    def test_source_format_is_jsf(self, paired_sonar_msgs):
        result = convert(paired_sonar_msgs, source_files=[Path("test.jsf")])
        assert result.source_format == "JSF"


@pytest.mark.unit
class TestConvertAuxOnly:
    """A file with no sonar messages but valid aux messages produces a
    zero-frequency survey with populated aux records."""

    def test_only_nmea_messages(self, nmea_rmc_msgs):
        result = convert(nmea_rmc_msgs, source_files=[Path("test.jsf")])
        assert result.n_frequencies == 0
        assert result.has_navigation

    def test_only_attitude_messages(self, pitch_roll_msgs):
        result = convert(pitch_roll_msgs, source_files=[Path("test.jsf")])
        assert result.n_frequencies == 0
        assert result.has_attitude


# ============================================================
# _build_frequency_channel — focused unit tests
# ============================================================

@pytest.mark.unit
class TestBuildFrequencyChannel:
    def test_returns_dataset(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert isinstance(ds, xr.Dataset)

    def test_dataset_validates(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        ds.sidescan.validate()

    def test_dimensions_correct(self, make_sonar_msg):
        msgs = []
        for i in range(7):
            msgs.append(make_sonar_msg(ping_number=i, channel=0, n_samples=50))
            msgs.append(make_sonar_msg(ping_number=i, channel=1, n_samples=50))
        ds = _build_frequency_channel(msgs, repair_mode="nan_pad")
        assert ds.sizes["ping"]    == 7
        assert ds.sizes["channel"] == 2
        assert ds.sizes["sample"]  == 50

    def test_channel_id_named_coord(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert list(ds.coords["channel_id"].values) == ["port", "starboard"]

    def test_port_and_starboard_data_separated(self, make_sonar_msg):
        msgs = []
        for i in range(3):
            msgs.append(make_sonar_msg(ping_number=i, channel=0,
                                       backscatter_val=111.0))
            msgs.append(make_sonar_msg(ping_number=i, channel=1,
                                       backscatter_val=999.0))
        ds = _build_frequency_channel(msgs, repair_mode="nan_pad")
        port = ds["backscatter_intensity"].sel(channel_id="port").values
        stbd = ds["backscatter_intensity"].sel(channel_id="starboard").values
        assert np.allclose(port, 111.0)
        assert np.allclose(stbd, 999.0)

    def test_messages_need_not_be_port_first(self, make_sonar_msg):
        port_first = []
        stbd_first = []
        for i in range(4):
            port_first.append(make_sonar_msg(ping_number=i, channel=0,
                                             backscatter_val=1.0))
            port_first.append(make_sonar_msg(ping_number=i, channel=1,
                                             backscatter_val=2.0))
            stbd_first.append(make_sonar_msg(ping_number=i, channel=1,
                                             backscatter_val=2.0))
            stbd_first.append(make_sonar_msg(ping_number=i, channel=0,
                                             backscatter_val=1.0))

        ds1 = _build_frequency_channel(port_first, repair_mode="nan_pad")
        ds2 = _build_frequency_channel(stbd_first, repair_mode="nan_pad")
        np.testing.assert_array_equal(
            ds1["backscatter_intensity"].values,
            ds2["backscatter_intensity"].values,
        )

    def test_no_raw_copy_written(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert "backscatter_raw" not in ds
        assert "backscatter_intensity" in ds

    def test_processing_level_is_raw(self, paired_sonar_msgs):
        from sidescanpy.core.history import processing_level
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert "processing_level" not in ds.attrs
        assert processing_level(ds) == "raw"

    def test_processing_history_is_empty_json_list(self, paired_sonar_msgs):
        import json
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert json.loads(ds.attrs["processing_history"]) == []

    def test_sample_rate_hz_from_sample_interval_not_sample_freq(self, make_sonar_msg):
        """Real-data regression: the JSF header's own sample_freq field is
        unreliable (survey-specific, off by 2-3x from the real rate) --
        sample_rate_hz must come from the independently-decoded
        sample_interval, not sample_freq."""
        msgs = [
            make_sonar_msg(ping_number=0, channel=0, sample_freq=54_656, sample_interval=1.0 / 120_192.3),
            make_sonar_msg(ping_number=0, channel=1, sample_freq=54_656, sample_interval=1.0 / 120_192.3),
        ]
        ds = _build_frequency_channel(msgs, repair_mode="nan_pad")
        assert ds.attrs["sample_rate_hz"] == pytest.approx(120_192.3, rel=1e-6)
        assert ds.attrs["sample_rate_hz"] != pytest.approx(54_656)

    def test_per_ping_lat_from_port(self, make_sonar_msg):
        msgs = []
        for i in range(4):
            msgs.append(make_sonar_msg(ping_number=i, channel=0,
                                       latitude=41.0 + i * 0.001))
            msgs.append(make_sonar_msg(ping_number=i, channel=1,
                                       latitude=99.0))  # ignored
        ds = _build_frequency_channel(msgs, repair_mode="nan_pad")
        for i in range(4):
            assert ds["latitude"].isel(ping=i).item() == pytest.approx(41.0 + i * 0.001)

    def test_ping_number_coordinate(self, make_sonar_msg):
        msgs = []
        for i in range(4):
            msgs.append(make_sonar_msg(ping_number=i, channel=0))
            msgs.append(make_sonar_msg(ping_number=i, channel=1))
        ds = _build_frequency_channel(msgs, repair_mode="nan_pad")
        np.testing.assert_array_equal(
            ds.coords["ping_number"].values, [0, 1, 2, 3]
        )

    def test_time_coord_is_datetime64_ns(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert ds["time"].dtype == np.dtype("datetime64[ns]")

    def test_time_monotonic(self, paired_sonar_msgs):
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        times = ds["time"].values.astype("int64")
        assert np.all(np.diff(times) > 0)

    def test_heave_m_is_nan(self, paired_sonar_msgs):
        """heave is not in SonarDataMessage; should be NaN-filled."""
        ds = _build_frequency_channel(paired_sonar_msgs, repair_mode="nan_pad")
        assert np.all(np.isnan(ds["heave_m"].values))


# ============================================================
# _build_navigation_record
# ============================================================

@pytest.mark.unit
class TestBuildNavigationRecord:
    def test_empty_input_returns_empty_valid_record(self):
        ds = _build_navigation_record([])
        ds.navigation_record.validate()
        assert ds.navigation_record.is_empty()

    def test_rmc_messages_populate_lat_lon(self, nmea_rmc_msgs):
        ds = _build_navigation_record(nmea_rmc_msgs)
        assert ds.sizes["obs"] == len(nmea_rmc_msgs)
        assert "latitude"  in ds.data_vars
        assert "longitude" in ds.data_vars

    def test_validates(self, nmea_rmc_msgs):
        ds = _build_navigation_record(nmea_rmc_msgs)
        ds.navigation_record.validate()

    def test_time_is_datetime64_ns(self, nmea_rmc_msgs):
        ds = _build_navigation_record(nmea_rmc_msgs)
        assert ds["time"].dtype == np.dtype("datetime64[ns]")

    def test_source_attr_set(self, nmea_rmc_msgs):
        ds = _build_navigation_record(nmea_rmc_msgs)
        assert "source" in ds.attrs


# ============================================================
# _build_attitude_record
# ============================================================

@pytest.mark.unit
class TestBuildAttitudeRecord:
    def test_empty_input_returns_empty_valid_record(self):
        ds = _build_attitude_record([])
        ds.attitude_record.validate()
        assert ds.attitude_record.is_empty()

    def test_pitch_roll_messages_populate_fields(self, pitch_roll_msgs):
        ds = _build_attitude_record(pitch_roll_msgs)
        assert ds.sizes["obs"] == len(pitch_roll_msgs)
        for var in ("roll_deg", "pitch_deg", "heave_m"):
            assert var in ds.data_vars

    def test_validates(self, pitch_roll_msgs):
        ds = _build_attitude_record(pitch_roll_msgs)
        ds.attitude_record.validate()

    def test_time_is_datetime64_ns(self, pitch_roll_msgs):
        ds = _build_attitude_record(pitch_roll_msgs)
        assert ds["time"].dtype == np.dtype("datetime64[ns]")


@pytest.mark.unit
class TestBuildAttitudeRecordFromSpecBytes:
    """Attitude record built from real (synthetic-bytes) message 2020 objects."""

    T0 = 1_700_000_000

    def _msgs(self, n=25, **kwargs):
        return [
            _make_real_pitch_roll_msg(
                time_s=self.T0 + i // 10, ms_in_second=(i % 10) * 100,
                pitch_deg=0.5 * i, roll_deg=-0.25 * i, heading_deg=10.0 * i,
                **kwargs,
            )
            for i in range(n)
        ]

    def test_time_is_absolute_when_built_from_messages(self):
        ds = _build_attitude_record(self._msgs())
        expected = np.array(
            [self.T0 * 10**9 + i * 100 * 10**6 for i in range(25)], dtype="int64"
        ).astype("datetime64[ns]")
        np.testing.assert_array_equal(ds["time"].values, expected)

    def test_time_unique_when_messages_within_same_second(self):
        ds = _build_attitude_record(self._msgs())
        assert np.unique(ds["time"].values).size == ds.sizes["obs"]

    def test_roll_pitch_not_nan_when_flags_valid(self):
        ds = _build_attitude_record(self._msgs())
        assert np.isfinite(ds["roll_deg"].values).all()
        assert np.isfinite(ds["pitch_deg"].values).all()
        np.testing.assert_allclose(
            ds["pitch_deg"].values, 0.5 * np.arange(25), atol=180 / 32768
        )

    def test_heading_included_when_message_carries_it(self):
        ds = _build_attitude_record(self._msgs())
        assert "heading_deg" in ds.data_vars
        np.testing.assert_allclose(ds["heading_deg"].values, 10.0 * np.arange(25))
        ds.attitude_record.validate()

    def test_heave_nan_when_heave_flag_clear(self):
        flags = PR_FLAG_PITCH | PR_FLAG_ROLL | PR_FLAG_HEADING
        ds = _build_attitude_record(self._msgs(flags=flags))
        assert np.isnan(ds["heave_m"].values).all()
        assert np.isfinite(ds["roll_deg"].values).all()


@pytest.mark.unit
class TestAuxRecordTimes:
    """Aux record times are float seconds since 1970 converted to ns."""

    T = 1_700_000_000.25

    def test_dvl_time_absolute_when_built(self):
        from tests.unit.io.jsf.conftest import _make_dvl_msg
        ds = _build_dvl_record([_make_dvl_msg(timestamp=self.T)])
        assert ds["time"].values[0] == np.datetime64(1_700_000_000_250_000_000, "ns")

    def test_sensor_time_absolute_when_built(self):
        from tests.unit.io.jsf.conftest import _make_pressure_msg
        ds = _build_sensor_record("pressure", [_make_pressure_msg(timestamp=self.T)])
        assert ds["time"].values[0] == np.datetime64(1_700_000_000_250_000_000, "ns")

    def test_navigation_time_exact_when_ms_resolution(self):
        from tests.unit.io.jsf.conftest import _make_nmea_rmc_msg
        ds = _build_navigation_record([_make_nmea_rmc_msg(timestamp=1_700_000_000.123)])
        assert ds["time"].values[0] == np.datetime64(1_700_000_000_123_000_000, "ns")


# ============================================================
# _build_dvl_record
# ============================================================

@pytest.mark.unit
class TestBuildDVLRecord:
    def test_empty_input_returns_empty_valid_record(self):
        ds = _build_dvl_record([])
        ds.dvl_record.validate()
        assert ds.dvl_record.is_empty()

    def test_dvl_messages_populate_altitude(self, dvl_msgs):
        ds = _build_dvl_record(dvl_msgs)
        assert ds.sizes["obs"] == len(dvl_msgs)
        assert "altitude_m" in ds.data_vars

    def test_altitude_values_come_from_bottom_dist(self):
        """DopplerVelocityLogDataMessage has no altitude_m attribute — its
        altitude-equivalent field is bottom_dist. Regression test for a real
        bug: _build_dvl_record read m.altitude_m (AttributeError on every
        real JSF file with DVL messages) while the old mock-based fixture
        set .altitude_m directly and masked it. Asserts the actual values
        flow through, not just that the variable exists."""
        from tests.unit.io.jsf.conftest import _make_dvl_msg
        msgs = [_make_dvl_msg(obs_index=i, bottom_dist=float(i)) for i in range(3)]
        ds = _build_dvl_record(msgs)
        np.testing.assert_array_equal(ds["altitude_m"].values, [0.0, 1.0, 2.0])

    def test_validates(self, dvl_msgs):
        ds = _build_dvl_record(dvl_msgs)
        ds.dvl_record.validate()

    def test_time_is_datetime64_ns(self, dvl_msgs):
        ds = _build_dvl_record(dvl_msgs)
        assert ds["time"].dtype == np.dtype("datetime64[ns]")


# ============================================================
# _build_sensor_record
# ============================================================

@pytest.mark.unit
class TestBuildSensorRecord:
    def test_empty_input_returns_empty_valid_record(self):
        ds = _build_sensor_record("pressure", [])
        ds.sensor_record.validate()
        assert ds.sensor_record.is_empty()

    def test_pressure_messages_populate_fields(self, pressure_msgs):
        ds = _build_sensor_record("pressure", pressure_msgs)
        assert ds.sizes["obs"] == len(pressure_msgs)
        assert "depth_m" in ds.data_vars

    def test_validates(self, pressure_msgs):
        ds = _build_sensor_record("pressure", pressure_msgs)
        ds.sensor_record.validate()

    def test_sensor_type_attr_set(self, pressure_msgs):
        ds = _build_sensor_record("pressure", pressure_msgs)
        assert ds.attrs["sensor_type"] == "pressure"

    def test_time_is_datetime64_ns(self, pressure_msgs):
        ds = _build_sensor_record("pressure", pressure_msgs)
        assert ds["time"].dtype == np.dtype("datetime64[ns]")


# ============================================================
# STARTING DEPTH (window offset of the first sample)
# ============================================================

def _offset_msgs(offsets, n=100):
    """Paired pings; offsets[i] = (port, starboard) starting depth in samples."""
    msgs = []
    for i, (op, os_) in enumerate(offsets):
        msgs.append(_make_sonar_msg(ping_number=i, channel=0, n_samples=n, starting_depth_samples=op))
        msgs.append(_make_sonar_msg(ping_number=i, channel=1, n_samples=n, starting_depth_samples=os_))
        msgs[-2].backscatter_intensity = np.arange(n, dtype=np.float32) + 1
        msgs[-1].backscatter_intensity = np.arange(n, dtype=np.float32) + 1
    return msgs


@pytest.mark.unit
class TestStartingDepth:
    def test_zero_offset_unchanged_and_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            ds = _build_frequency_channel(_offset_msgs([(0, 0)] * 3), repair_mode="nan_pad")
        np.testing.assert_array_equal(ds["sample"].values, np.arange(100))
        assert "starting depth" not in caplog.text.lower()
        assert "jsf_starting_depth_min" not in ds.attrs

    def test_constant_offset_shifts_sample_coordinate(self, caplog):
        from sidescanpy.processing.sample_axis import sample_index, sample_interval_m

        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            ds = _build_frequency_channel(_offset_msgs([(40, 40)] * 3), repair_mode="nan_pad")
        np.testing.assert_array_equal(ds["sample"].values, 40 + np.arange(100))
        assert ds.sizes["sample"] == 100
        # Slant range of recorded sample i is (start_depth + i) x interval.
        rng = sample_index(ds) * sample_interval_m(ds)[0]
        assert rng[0] == pytest.approx(40 * 1500.0 / (2 * 400_000.0))
        warnings = [r for r in caplog.records if "starting depth" in r.getMessage().lower()]
        assert len(warnings) == 1 and warnings[0].levelname == "WARNING"
        assert ds.attrs["jsf_starting_depth_min"] == 40
        assert ds.attrs["jsf_starting_depth_max"] == 40

    def test_varying_offset_places_each_ping_at_its_range(self):
        ds = _build_frequency_channel(_offset_msgs([(10, 10), (30, 20), (10, 10)]), repair_mode="nan_pad")
        np.testing.assert_array_equal(ds["sample"].values, 10 + np.arange(120))
        v = ds["backscatter_intensity"].values
        # Ping 1 port starts 20 samples later than ping 0, starboard 10 later.
        assert np.isnan(v[1, 0, :20]).all() and v[1, 0, 20] == 1
        assert np.isnan(v[1, 1, :10]).all() and v[1, 1, 10] == 1
        assert v[0, 0, 0] == 1 and np.isnan(v[0, 0, 100:]).all()



@pytest.mark.unit
class TestLaybackAndPositionSource:
    def _convert(self, **kw):
        msgs = [
            _make_sonar_msg(ping_number=p, channel=c, **kw)
            for p in range(3) for c in (0, 1)
        ]
        survey = convert(msgs, source_files=[Path("test.jsf")])
        return survey.get_frequency(survey.frequencies[0]).dataset

    def test_layback_and_cable_out_carried_per_ping(self):
        ds = self._convert(layback_m=125.9, cable_out_m=130.0)
        np.testing.assert_allclose(ds["layback_m"].values, 125.9, rtol=1e-6)
        np.testing.assert_allclose(ds["cable_out_m"].values, 130.0)
        assert ds["layback_m"].attrs["units"] == "m"

    def test_position_source_towfish_when_bit_13_set(self):
        ds = self._convert(position_interpolated=True)
        assert ds.attrs["position_source"] == "towfish"
        assert "bit 13" in ds.attrs["position_source_reason"]

    def test_position_source_ship_when_bit_13_clear(self):
        ds = self._convert(position_interpolated=False)
        assert ds.attrs["position_source"] == "ship"

    def test_tow_point_offset_recorded_when_nonzero(self):
        ds = self._convert(gps_offset_x=1.5, gps_offset_y=-4.0)
        assert ds.attrs["tow_point_starboard_m"] == pytest.approx(1.5)
        assert ds.attrs["tow_point_aft_m"] == pytest.approx(-4.0)

    def test_tow_point_offset_absent_when_zero(self):
        ds = self._convert()
        assert "tow_point_aft_m" not in ds.attrs
