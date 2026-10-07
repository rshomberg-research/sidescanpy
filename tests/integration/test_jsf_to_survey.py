"""
Integration tests for the JSF → SideScanSurvey pipeline.

Reads a real JSF file from tests/data/jsf/sample.jsf and verifies the full
parsing + conversion stack produces a schema-valid survey. These tests are
loose by design — they assert structural correctness, not specific values
tied to the fixture file. Tighten as needed when fixture bugs surface.

Marked @pytest.mark.integration; skipped by default. Run with `pytest --slow`.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import pytest

from sidescanpy.io.jsf import read_jsf
from sidescanpy.io.jsf.converter import convert
from sidescanpy.core.survey import SideScanSurvey


SAMPLE_JSF = Path(__file__).resolve().parents[1] / "data" / "jsf" / "sample.jsf"


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def jsf_messages():
    """Parse the sample JSF file once per test module (it's slow)."""
    if not SAMPLE_JSF.exists():
        pytest.skip(f"Sample JSF file not found at {SAMPLE_JSF}")
    return read_jsf(SAMPLE_JSF)


@pytest.fixture(scope="module")
def real_survey(jsf_messages):
    """Convert the sample JSF messages to a survey once per test module."""
    return convert(jsf_messages, source_files=[SAMPLE_JSF])


# ============================================================
# READER
# ============================================================

@pytest.mark.integration
class TestReadJsf:
    def test_returns_list(self, jsf_messages):
        assert isinstance(jsf_messages, list)

    def test_nonempty(self, jsf_messages):
        assert len(jsf_messages) > 0

    def test_messages_have_header(self, jsf_messages):
        from sidescanpy.io.jsf.messages import JSFMessage
        # All messages should be JSFMessage subclass instances with a header
        for m in jsf_messages[:10]:
            assert isinstance(m, JSFMessage)
            assert hasattr(m, "_header")


# ============================================================
# CONVERTER → SURVEY
# ============================================================

@pytest.mark.integration
class TestConvertToSurvey:
    def test_returns_survey(self, real_survey):
        assert isinstance(real_survey, SideScanSurvey)

    def test_survey_has_at_least_one_frequency(self, real_survey):
        assert real_survey.n_frequencies >= 1

    def test_source_files_preserved(self, real_survey):
        assert real_survey.source_files == [SAMPLE_JSF]

    def test_source_format_is_jsf(self, real_survey):
        assert real_survey.source_format == "JSF"

    def test_survey_name_default_from_filename(self, real_survey):
        # Default survey_name = source_files[0].stem
        assert real_survey.survey_name == SAMPLE_JSF.stem


# ============================================================
# SCHEMA VALIDATION (the key correctness check)
# ============================================================

@pytest.mark.integration
class TestSchemaValidation:
    def test_each_frequency_channel_validates(self, real_survey):
        for fk in real_survey.frequencies:
            real_survey.get_frequency(fk).dataset.sidescan.validate()

    def test_navigation_validates_if_present(self, real_survey):
        if real_survey.navigation is not None:
            real_survey.navigation.navigation_record.validate()

    def test_attitude_validates_if_present(self, real_survey):
        if real_survey.attitude is not None:
            real_survey.attitude.attitude_record.validate()

    def test_dvl_validates_if_present(self, real_survey):
        if real_survey.dvl is not None:
            real_survey.dvl.dvl_record.validate()

    def test_sensors_validate_if_present(self, real_survey):
        for sensor_type in real_survey.sensor_types:
            real_survey.get_sensor(sensor_type).sensor_record.validate()


# ============================================================
# SURVEY-LEVEL PROPERTIES
# ============================================================

@pytest.mark.integration
class TestSurveyProperties:
    def test_time_range_is_populated(self, real_survey):
        tr = real_survey.time_range
        assert tr is not None
        t_min, t_max = tr
        assert t_min <= t_max

    def test_bounding_box_is_populated(self, real_survey):
        bbox = real_survey.bounding_box
        # bbox can be None if no positional fixes exist; warn loudly but
        # don't fail. Real surveys typically have valid GPS.
        if bbox is None:
            pytest.skip("bbox is None — sample file lacks positional data")
        lat_min, lat_max, lon_min, lon_max = bbox
        assert lat_min <= lat_max
        assert lon_min <= lon_max
        # Sanity check: real lat/lon ranges
        assert -90.0 <= lat_min <= 90.0
        assert -90.0 <= lat_max <= 90.0
        assert -180.0 <= lon_min <= 180.0
        assert -180.0 <= lon_max <= 180.0


# ============================================================
# FREQUENCY CHANNEL SHAPES
# ============================================================

@pytest.mark.integration
class TestFrequencyChannelShapes:
    def test_each_channel_has_pings(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.n_pings > 0

    def test_each_channel_has_two_channel_dim(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.n_channels == 2

    def test_each_channel_has_samples(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.n_samples > 0

    def test_channel_id_is_named(self, real_survey):
        for fk in real_survey.frequencies:
            ds = real_survey.get_frequency(fk).dataset
            assert list(ds.channel_id.values) == ["port", "starboard"]


# ============================================================
# DIAGNOSTIC PRINTOUT (helpful when debugging fixture changes)
# ============================================================

@pytest.mark.integration
def test_print_survey_summary(real_survey, capsys):
    """Not a real test — just dumps a summary so test output is informative
    when something fails further up. capsys captures and discards by default,
    but `-s` flag will show it."""
    print()
    print(f"Survey:        {real_survey}")
    print(f"Frequencies:   {real_survey.frequency_hz_list}")
    for fk in real_survey.frequencies:
        ch = real_survey.get_frequency(fk)
        print(f"  {fk}: {ch.n_pings} pings × {ch.n_samples} samples")
    print(f"Navigation:    {real_survey.navigation.sizes['obs'] if real_survey.has_navigation else 'empty'}")
    print(f"Attitude:      {real_survey.attitude.sizes['obs']   if real_survey.has_attitude   else 'empty'}")
    print(f"DVL:           {real_survey.dvl.sizes['obs']        if real_survey.has_dvl        else 'empty'}")
    print(f"Sensors:       {real_survey.sensor_types}")
    print(f"Time range:    {real_survey.time_range}")
    print(f"Bounding box:  {real_survey.bounding_box}")
    # No assert — informational