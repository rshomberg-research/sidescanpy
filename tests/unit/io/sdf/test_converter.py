"""
Unit tests for sidescanpy.io.sdf.converter.

Validates SDFSonarPage list → SideScanSurvey conversion. Tests use
synthetic page objects (built directly via make_sdf_sonar_page) so the
converter is decoupled from the byte-level reader.

2026 rshomberg@uri.edu
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.sdf.converter import convert
from tests.unit.io.sdf.conftest import make_sdf_sonar_page

# ============================================================
# BASIC CONVERSION
# ============================================================

@pytest.mark.unit
class TestBasicConversion:
    def test_returns_survey_instance(self):
        pages = [make_sdf_sonar_page(ping_number=1)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert isinstance(survey, SideScanSurvey)

    def test_single_page_creates_one_channel(self):
        pages = [make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert survey.n_frequencies == 1

    def test_two_frequencies_create_two_channels(self):
        pages = [
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=1),
            make_sdf_sonar_page(page_version=3502, center_freq_hz=900_000,
                               ping_number=2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert survey.n_frequencies == 2

    def test_survey_name_from_kwarg(self):
        pages = [make_sdf_sonar_page()]
        survey = convert(
            pages,
            source_files=[Path("test.sdf")],
            survey_name="custom_name",
        )
        assert survey.survey_name == "custom_name"   # was .name

    def test_survey_name_from_first_source_file(self):
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("/path/to/sample.sdf")])
        assert survey.survey_name == "sample"   # was .name

    def test_source_files_attached(self):
        paths = [Path("a.sdf"), Path("b.sdf")]
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=paths)
        assert survey.source_files == paths


# ============================================================
# FREQUENCY GROUPING
# ============================================================

@pytest.mark.unit
class TestFrequencyGrouping:
    def test_pages_grouped_by_center_freq(self):
        pages = [
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=1),
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=2),
            make_sdf_sonar_page(page_version=3502, center_freq_hz=900_000,
                               ping_number=3),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch_low = survey.get_frequency("hz_455000")
        ch_high = survey.get_frequency("hz_900000")
        assert ch_low.n_pings == 2
        assert ch_high.n_pings == 1

    def test_3501_and_3502_separated(self):
        pages = [
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=1),
            make_sdf_sonar_page(page_version=3502, center_freq_hz=900_000,
                               ping_number=1),  # same ping num, different freq
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert survey.n_frequencies == 2

    def test_freq_keys_are_hz_prefixed(self):
        pages = [
            make_sdf_sonar_page(center_freq_hz=455_000),
            make_sdf_sonar_page(center_freq_hz=900_000, ping_number=2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        keys = list(survey.frequencies)
        assert "hz_455000" in keys
        assert "hz_900000" in keys

    def test_pages_within_frequency_sorted_by_ping_number(self):
        """Pages are sorted by ping_number even if input order is jumbled."""
        pages = [
            make_sdf_sonar_page(ping_number=10),
            make_sdf_sonar_page(ping_number=5),
            make_sdf_sonar_page(ping_number=20),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        ping_nums = ch.dataset["ping_number"].values
        assert list(ping_nums) == [5, 10, 20]


# ============================================================
# PER-PING FIELDS
# ============================================================

@pytest.mark.unit
class TestPerPingFields:
    def test_latitude_populated(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, latitude_deg=44.1),
            make_sdf_sonar_page(ping_number=2, latitude_deg=44.2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["latitude"].values, [44.1, 44.2], atol=1e-6,
        )

    def test_longitude_populated(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, longitude_deg=-68.1),
            make_sdf_sonar_page(ping_number=2, longitude_deg=-68.2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["longitude"].values, [-68.1, -68.2], atol=1e-6,
        )

    def test_heading_populated(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, heading_deg=10.0),
            make_sdf_sonar_page(ping_number=2, heading_deg=20.0),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["heading_deg"].values, [10.0, 20.0], atol=1e-4,
        )

    def test_pitch_populated(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, pitch_deg=-3.0),
            make_sdf_sonar_page(ping_number=2, pitch_deg=2.5),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["pitch_deg"].values, [-3.0, 2.5], atol=1e-4,
        )

    def test_roll_populated(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, roll_deg=1.5),
            make_sdf_sonar_page(ping_number=2, roll_deg=-0.5),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["roll_deg"].values, [1.5, -0.5], atol=1e-4,
        )

    # FIXME: depth should not be nan
    def test_depth_is_nan(self):
        """Klein stores depth in Volts; without INI calibration we set NaN."""
        pages = [make_sdf_sonar_page(ping_number=1)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert np.all(np.isnan(ch.dataset["depth_m"].values))

    # FIXME: depth should not be nan
    # def test_depth_populated(self):
    #     pages = [
    #         make_sdf_sonar_page(ping_number=1, depth_m=10.0),
    #         make_sdf_sonar_page(ping_number=2, depth_m=12.0),
    #     ]
    #     survey = convert(pages, source_files=[Path("test.sdf")])
    #     ch = survey.get_frequency("hz_455000")
    #     np.testing.assert_allclose(
    #         ch.dataset["depth_m"].values, [10.0, 12.0], atol=1e-4,
    #     )

    def test_altitude_equals_fathometer_reading(self):
        """For now, altitude_m equals fathometer_reading_m (no attitude correction)."""
        pages = [make_sdf_sonar_page(altitude_m=4.5)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        np.testing.assert_allclose(
            ch.dataset["altitude_m"].values, [4.5], atol=1e-4,
        )
        np.testing.assert_allclose(
            ch.dataset["fathometer_reading_m"].values, [4.5], atol=1e-4,
        )

    def test_sound_vel_is_nan(self):
        """Klein doesn't provide sound velocity; field is NaN per ping."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        sv = ch.dataset["sound_vel_ms"].values
        assert np.all(np.isnan(sv))

    def test_heave_is_nan(self):
        """Klein doesn't provide heave; field is NaN per ping."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        heave = ch.dataset["heave_m"].values
        assert np.all(np.isnan(heave))


# ============================================================
# BACKSCATTER SHAPE
# ============================================================

@pytest.mark.unit
class TestBackscatterShape:
    def test_backscatter_shape(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=50),
            make_sdf_sonar_page(ping_number=2, num_samples=50),
            make_sdf_sonar_page(ping_number=3, num_samples=50),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert ch.dataset["backscatter_intensity"].shape == (3, 2, 50)

    def test_backscatter_dtype(self):
        pages = [make_sdf_sonar_page(num_samples=50)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert ch.dataset["backscatter_intensity"].dtype == np.float32

    def test_port_at_channel_0(self):
        """Channel 0 is port; the data should match port_data values."""
        pages = [make_sdf_sonar_page(num_samples=20)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        port_intensities = ch.dataset["backscatter_intensity"].values[0, 0, :]
        # Synthetic page port_data is np.arange(num_samples) → 0..19
        np.testing.assert_array_equal(port_intensities[:20], np.arange(20))

    def test_stbd_at_channel_1(self):
        pages = [make_sdf_sonar_page(num_samples=20)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        stbd_intensities = ch.dataset["backscatter_intensity"].values[0, 1, :]
        # Synthetic page stbd_data is np.arange(num_samples) + 10000
        np.testing.assert_array_equal(stbd_intensities[:20], np.arange(20) + 10000)

    def test_no_raw_copy_written(self):
        pages = [make_sdf_sonar_page(num_samples=20)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert "backscatter_raw" not in ch.dataset

    def test_pad_to_max_with_nan(self):
        """Pages with fewer samples should be padded to max with NaN."""
        pages = [
            make_sdf_sonar_page(ping_number=1, n_port_samples=30, n_stbd_samples=30),
            make_sdf_sonar_page(ping_number=2, n_port_samples=50, n_stbd_samples=50),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        bs = ch.dataset["backscatter_intensity"].values
        assert bs.shape == (2, 2, 50)

        # Ping 0 (30 samples), positions 30-49 should be NaN
        assert np.all(np.isnan(bs[0, 0, 30:]))
        assert np.all(np.isnan(bs[0, 1, 30:]))

        # Ping 0 first 30 are real values
        assert np.all(np.isfinite(bs[0, 0, :30]))

        # Ping 1 (50 samples), all finite
        assert np.all(np.isfinite(bs[1, 0, :]))

    def test_uniform_samples_no_nan_padding(self):
        """When all pages have the same num_samples, no NaN appears."""
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=20),
            make_sdf_sonar_page(ping_number=2, num_samples=20),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        bs = ch.dataset["backscatter_intensity"].values
        assert not np.any(np.isnan(bs))

    def test_valid_samples_tracks_true_length_when_padded(self):
        """valid_samples preserves each ping's true length after padding."""
        pages = [
            make_sdf_sonar_page(ping_number=1, n_port_samples=30, n_stbd_samples=30),
            make_sdf_sonar_page(ping_number=2, n_port_samples=50, n_stbd_samples=50),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        valid = ch.dataset["valid_samples"].values
        assert valid.shape == (2, 2)
        np.testing.assert_array_equal(valid[0], [30, 30])
        np.testing.assert_array_equal(valid[1], [50, 50])

    def test_valid_samples_equals_max_when_uniform(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=20),
            make_sdf_sonar_page(ping_number=2, num_samples=20),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        valid = ch.dataset["valid_samples"].values
        assert np.all(valid == 20)

    def test_warns_when_padding_occurs(self, caplog):
        pages = [
            make_sdf_sonar_page(ping_number=1, n_port_samples=30, n_stbd_samples=30),
            make_sdf_sonar_page(ping_number=2, n_port_samples=50, n_stbd_samples=50),
        ]
        with caplog.at_level(logging.WARNING):
            convert(pages, source_files=[Path("test.sdf")])
        assert any("NaN-padded" in r.message for r in caplog.records)

    def test_no_warning_when_uniform(self, caplog):
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=20),
            make_sdf_sonar_page(ping_number=2, num_samples=20),
        ]
        with caplog.at_level(logging.WARNING):
            convert(pages, source_files=[Path("test.sdf")])
        assert not any("NaN-padded" in r.message for r in caplog.records)


# ============================================================
# SAMPLE RATE
# ============================================================

@pytest.mark.unit
class TestSampleRate:
    def test_sample_rate_computed_from_first_page(self):
        """sample_rate_hz = num_samples × 1500 / (2 × range_m)."""
        pages = [make_sdf_sonar_page(num_samples=2500, range_m=60.0)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        # 2500 * 1500 / (2 * 60) = 31250 Hz
        expected = 2500 * 1500 / (2 * 60.0)
        assert ch.dataset.attrs["sample_rate_hz"] == pytest.approx(expected, abs=1.0)

    def test_warning_on_different_num_samples(self, caplog):
        """If pages within a frequency have different num_samples, warn."""
        import logging
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=2500, range_m=60.0),
            make_sdf_sonar_page(ping_number=2, num_samples=3000, range_m=60.0),
        ]
        with caplog.at_level(logging.WARNING):
            convert(pages, source_files=[Path("test.sdf")])
        assert any("num_samples" in rec.message or "sample_rate" in rec.message
                   for rec in caplog.records)

    def test_warning_on_different_range(self, caplog):
        import logging
        pages = [
            make_sdf_sonar_page(ping_number=1, num_samples=2500, range_m=60.0),
            make_sdf_sonar_page(ping_number=2, num_samples=2500, range_m=80.0),
        ]
        with caplog.at_level(logging.WARNING):
            convert(pages, source_files=[Path("test.sdf")])
        assert any("range" in rec.message or "sample_rate" in rec.message
                   for rec in caplog.records)


# ============================================================
# NAVIGATION AUX RECORD
# ============================================================

@pytest.mark.unit
class TestNavigationAuxRecord:
    def test_nav_record_built_from_pages(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, latitude_deg=44.1, longitude_deg=-68.1,
                               timestamp=1746136200.0),
            make_sdf_sonar_page(ping_number=2, latitude_deg=44.2, longitude_deg=-68.2,
                               timestamp=1746136201.0),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        nav = survey.navigation
        assert nav.sizes["obs"] == 2

    def test_nav_lat_lon_per_obs(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, latitude_deg=44.1, longitude_deg=-68.1),
            make_sdf_sonar_page(ping_number=2, latitude_deg=44.2, longitude_deg=-68.2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        nav = survey.navigation
        np.testing.assert_allclose(nav["latitude"].values, [44.1, 44.2], atol=1e-6)
        np.testing.assert_allclose(nav["longitude"].values, [-68.1, -68.2], atol=1e-6)


    def test_nav_time_monotonic(self):
        """Navigation time coord should be sorted."""
        pages = [
            make_sdf_sonar_page(ping_number=1, timestamp=1746136200.0),
            make_sdf_sonar_page(ping_number=2, timestamp=1746136201.0),
            make_sdf_sonar_page(ping_number=3, timestamp=1746136202.0),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        nav = survey.navigation
        times = nav["time"].values
        assert np.all(np.diff(times) >= np.timedelta64(0, "ns"))

    def test_nav_record_built_from_pages_deg_kwargs(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, latitude_deg=44.1, longitude_deg=-68.1,
                            timestamp=1746136200.0),
            make_sdf_sonar_page(ping_number=2, latitude_deg=44.2, longitude_deg=-68.2,
                            timestamp=1746136201.0),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        nav = survey.navigation
        assert nav.sizes["obs"] == 2

# ============================================================
# ATTITUDE AUX RECORD
# ============================================================

@pytest.mark.unit
class TestAttitudeAuxRecord:
    def test_attitude_record_built_from_pages(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, heading_deg=10, pitch_deg=1, roll_deg=2),
            make_sdf_sonar_page(ping_number=2, heading_deg=20, pitch_deg=2, roll_deg=3),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        att = survey.attitude
        assert att.sizes["obs"] == 2

    def test_attitude_heading_pitch_roll_per_obs(self):
        pages = [
            make_sdf_sonar_page(ping_number=1, heading_deg=10, pitch_deg=1, roll_deg=2),
            make_sdf_sonar_page(ping_number=2, heading_deg=20, pitch_deg=2, roll_deg=3),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        att = survey.attitude
        np.testing.assert_allclose(att["heading_deg"].values, [10, 20], atol=1e-4)
        np.testing.assert_allclose(att["pitch_deg"].values, [1, 2], atol=1e-4)
        np.testing.assert_allclose(att["roll_deg"].values, [2, 3], atol=1e-4)


# ============================================================
# EMPTY AUX RECORDS
# ============================================================

@pytest.mark.unit
class TestEmptyAuxRecords:
    def test_dvl_record_empty(self):
        """Klein doesn't have DVL; the DVL aux record should be empty."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        # If has_dvl is False, the record should be empty
        if survey.has_dvl:
            assert survey.dvl.sizes["obs"] == 0

    def test_no_sensor_records_or_empty(self):
        """Klein doesn't have pressure/sensor streams; sensors empty or absent."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert len(survey.sensors) == 0


# ============================================================
# SCHEMA VALIDATION
# ============================================================

@pytest.mark.unit
class TestSchemaValidation:
    def test_each_channel_validates(self):
        pages = [
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=1),
            make_sdf_sonar_page(page_version=3502, center_freq_hz=900_000,
                               ping_number=1),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        for _fk, channel in survey.freq_channels.items():
            channel.dataset.sidescan.validate()

    def test_processing_level_is_raw(self):
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert ch.dataset.sidescan.processing_level == "raw"

    def test_processing_history_is_empty_list(self):
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        history = json.loads(ch.dataset.attrs["processing_history"])
        assert history == []


# ============================================================
# EDGE CASES
# ============================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_empty_page_list_raises(self):
        with pytest.raises(ValueError, match="empty|no pages|no sonar"):
            convert([], source_files=[Path("test.sdf")])

    def test_single_frequency_only(self):
        pages = [
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=1),
            make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000,
                               ping_number=2),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        assert survey.n_frequencies == 1

    def test_pages_with_different_num_samples_per_channel(self):
        """Pages where port and starboard have different sample counts."""
        pages = [
            make_sdf_sonar_page(ping_number=1, n_port_samples=20, n_stbd_samples=30),
        ]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        bs = ch.dataset["backscatter_intensity"].values
        # Should pad to max(port, stbd) = 30
        assert bs.shape[2] == 30
        assert np.all(np.isnan(bs[0, 0, 20:]))   # port shorter, NaN tail
        assert np.all(np.isfinite(bs[0, 1, :30]))   # stbd full length


# ============================================================
# ATTRS / METADATA
# ============================================================

@pytest.mark.unit
class TestProcessingMetadata:
    def test_freq_hz_start_and_end_set(self):
        """For Klein single-frequency: freq_hz_start == freq_hz_end == center."""
        pages = [make_sdf_sonar_page(center_freq_hz=455_000)]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert ch.dataset.attrs["freq_hz_start"] == 455_000
        assert ch.dataset.attrs["freq_hz_end"] == 455_000

    def test_pulse_length_us_set(self):
        """Klein doesn't expose pulse length; field is 0.0 by convention."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        assert ch.dataset.attrs["pulse_length_us"] == 0.0

    def test_required_attrs_present(self):
        """All schema-required attrs must be present after conversion."""
        pages = [make_sdf_sonar_page()]
        survey = convert(pages, source_files=[Path("test.sdf")])
        ch = survey.get_frequency("hz_455000")
        required = {
            "freq_hz_start", "freq_hz_end", "sample_rate_hz",
            "pulse_length_us", "processing_history",
        }
        attrs_keys = set(ch.dataset.attrs.keys())
        missing = required - attrs_keys
        assert not missing, f"Missing required attrs: {missing}"

class TestFilterUnknownPages:
    def test_filter_logs_count(self, caplog):
        """Filter unknown pages, log at debug about how many were dropped."""
        import logging

        from sidescanpy.io.sdf.pages import SDFPageHeader, SDFUnknownPage

        # Build mixed input: 2 sonar + 1 unknown
        sonar1 = make_sdf_sonar_page(ping_number=1)
        sonar2 = make_sdf_sonar_page(ping_number=2)

        # Build an unknown page (header for non-sonar version)
        from tests.unit.io.sdf.conftest import make_sdf_header_bytes
        raw = make_sdf_header_bytes(page_version=9999)
        header = SDFPageHeader.from_bytes(raw)
        unknown = SDFUnknownPage(header=header, body_bytes=b"")

        with caplog.at_level(logging.DEBUG):
            survey = convert(
                [sonar1, unknown, sonar2],
                source_files=[Path("test.sdf")],
            )

        # Survey was built (sonar pages preserved)
        ch = survey.get_frequency("hz_455000")
        assert ch.n_pings == 2

        # Filter was logged
        assert any("filtered" in r.message.lower()
                   for r in caplog.records)

    def test_only_unknown_pages_raises(self):
        """If input has only unknown pages, should raise ValueError."""
        from sidescanpy.io.sdf.pages import SDFPageHeader, SDFUnknownPage
        from tests.unit.io.sdf.conftest import make_sdf_header_bytes
        raw = make_sdf_header_bytes(page_version=9999)
        header = SDFPageHeader.from_bytes(raw)
        unknown = SDFUnknownPage(header=header, body_bytes=b"")

        with pytest.raises(ValueError, match="no sonar"):
            convert([unknown], source_files=[Path("test.sdf")])
