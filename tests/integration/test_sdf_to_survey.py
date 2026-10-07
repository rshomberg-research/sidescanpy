"""
Integration test for SDF reader + converter on real Klein survey data.

Skipped if the fixture file is not present. To enable:
    Place a Klein SDF file at tests/data/sdf/sample.sdf

The fixture used during development was a mid-latitude (~44°N) coastal sidescan
survey (Klein 4900, dual-frequency 455/900 kHz).

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.sdf import convert, read_sdf
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
)

SAMPLE_SDF = Path(__file__).resolve().parents[1] / "data" / "sdf" / "sample.sdf"


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def real_pages():
    """Read SDF file once per test module."""
    if not SAMPLE_SDF.exists():
        pytest.skip(f"Sample SDF file not found at {SAMPLE_SDF}")
    return read_sdf(SAMPLE_SDF)


@pytest.fixture(scope="module")
def real_survey(real_pages):
    """Convert pages to survey once per test module."""
    return convert(real_pages, source_files=[SAMPLE_SDF])


# ============================================================
# READ + CONVERT
# ============================================================

@pytest.mark.integration
class TestReadAndConvert:
    def test_read_returns_pages(self, real_pages):
        assert len(real_pages) > 0

    def test_pages_are_sonar_versions(self, real_pages):
        from sidescanpy.io.sdf.pages import PAGE_VERSIONS_SONAR
        for page in real_pages:
            assert page.page_version in PAGE_VERSIONS_SONAR

    def test_convert_returns_survey(self, real_survey):
        from sidescanpy.core.survey import SideScanSurvey
        assert isinstance(real_survey, SideScanSurvey)

    def test_survey_has_frequency_channels(self, real_survey):
        assert real_survey.n_frequencies >= 1


# ============================================================
# DUAL FREQUENCY (KLEIN-SPECIFIC)
# ============================================================

@pytest.mark.integration
class TestDualFrequency:
    def test_two_frequencies_present(self, real_survey):
        """Klein 4900 records both low (455 kHz) and high (900 kHz) channels."""
        assert real_survey.n_frequencies == 2

    def test_low_freq_channel_present(self, real_survey):
        assert "hz_455000" in real_survey.frequencies

    def test_high_freq_channel_present(self, real_survey):
        assert "hz_900000" in real_survey.frequencies

    def test_each_channel_has_pings(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.n_pings > 0
            assert ch.n_samples > 0


# ============================================================
# PER-PING NAVIGATION
# ============================================================

@pytest.mark.integration
class TestPerPingNavigation:
    def test_latitude_in_maine_range(self, real_survey):
        """Sample fixture is from ~44°N."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            lats = ch.dataset["latitude"].values
            valid = lats[np.isfinite(lats)]
            if valid.size > 0:
                assert 43.0 <= valid.min() <= 45.0
                assert 43.0 <= valid.max() <= 45.0

    def test_longitude_in_maine_range(self, real_survey):
        """Sample fixture longitude in (-69.5, -68) range."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            lons = ch.dataset["longitude"].values
            valid = lons[np.isfinite(lons)]
            if valid.size > 0:
                assert -70.0 <= valid.min() <= -67.0
                assert -70.0 <= valid.max() <= -67.0

    def test_heading_varies(self, real_survey):
        """Heading should vary across the survey (vehicle is moving)."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            headings = ch.dataset["heading_deg"].values
            assert np.std(headings) > 0  # not all identical

    def test_heading_in_valid_range(self, real_survey):
        """0 ≤ heading < 360."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            headings = ch.dataset["heading_deg"].values
            valid = headings[np.isfinite(headings)]
            assert np.all(valid >= 0)
            assert np.all(valid < 360)


# ============================================================
# KLEIN-SPECIFIC ZERO FIELDS
# ============================================================

@pytest.mark.integration
class TestKleinSpecificFields:
    def test_pitch_roll_zero_or_finite(self, real_survey):
        """Klein 4900 in basic config doesn't report pitch/roll; expect zeros."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            pitches = ch.dataset["pitch_deg"].values
            rolls = ch.dataset["roll_deg"].values
            # Either all zero (sensor not used) or all finite
            assert np.all(np.isfinite(pitches))
            assert np.all(np.isfinite(rolls))

    def test_sound_vel_all_nan(self, real_survey):
        """Klein doesn't provide sound velocity; should be all NaN."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            sv = ch.dataset["sound_vel_ms"].values
            assert np.all(np.isnan(sv))

    def test_heave_all_nan(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            heave = ch.dataset["heave_m"].values
            assert np.all(np.isnan(heave))

    def test_pulse_length_zero(self, real_survey):
        """Klein doesn't expose pulse length; we set 0.0."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.dataset.attrs["pulse_length_us"] == 0.0


# ============================================================
# BACKSCATTER
# ============================================================

@pytest.mark.integration
class TestBackscatter:
    def test_backscatter_positive(self, real_survey):
        """Klein samples are uint32 → float32 positive values."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            bs = ch.dataset["backscatter_intensity"].values
            finite = bs[np.isfinite(bs)]
            if finite.size > 0:
                assert finite.min() >= 0

    def test_backscatter_raw_equals_intensity(self, real_survey):
        """Raw and intensity start identical (pre-radiometric)."""
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            np.testing.assert_array_equal(
                ch.dataset["backscatter_raw"].values,
                ch.dataset["backscatter_intensity"].values,
            )

    def test_backscatter_dtype_float32(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            assert ch.dataset["backscatter_intensity"].dtype == np.float32


# ============================================================
# SCHEMA VALIDATION
# ============================================================

@pytest.mark.integration
class TestSchemaValidation:
    def test_each_channel_validates(self, real_survey):
        for fk in real_survey.frequencies:
            ch = real_survey.get_frequency(fk)
            ch.dataset.sidescan.validate()

    def test_navigation_record_validates(self, real_survey):
        if real_survey.has_navigation:
            real_survey.navigation.navigation_record.validate()

    def test_attitude_record_validates(self, real_survey):
        if real_survey.has_attitude:
            real_survey.attitude.attitude_record.validate()

    def test_dvl_record_validates(self, real_survey):
        if real_survey.dvl is not None:
            real_survey.dvl.dvl_record.validate()


# ============================================================
# END-TO-END PIPELINE
# ============================================================

@pytest.mark.integration
class TestPipeline:
    def test_full_pipeline_runs(self, real_survey):
        """Run the full pipeline and verify it completes."""
        survey = interpolate_aux_onto_pings(real_survey)

        for fk, channel in list(survey.freq_channels.items()):
            ds = channel.dataset
            ds = apply_tvg(ds)
            ds = compute_geometric_ranges(ds)
            ds = compute_georeferencing(ds)
            survey.add_frequency(fk, ds)

        # Verify each channel has the new variables
        new_vars = (
            "slant_range_m", "ground_range_m",
            "across_track_m", "along_track_m",
            "latitude_per_sample", "longitude_per_sample",
        )
        for fk in survey.frequencies:
            ch = survey.get_frequency(fk)
            for var in new_vars:
                assert var in ch.dataset.data_vars

    def test_geometric_with_nan_sound_velocity_falls_back(self, real_survey):
        """Klein has NaN sound_vel_ms; geometric ranges should fall back."""
        survey = interpolate_aux_onto_pings(real_survey)

        ch = list(survey.freq_channels.values())[0]
        ds = compute_geometric_ranges(ch.dataset)

        # slant_range_m should be finite for all samples
        slant = ds["slant_range_m"].values
        assert np.all(np.isfinite(slant))

    def test_georeferencing_produces_valid_coords(self, real_survey):
        """Per-sample lat/lon should be in the fixture's expected range after pipeline."""
        survey = interpolate_aux_onto_pings(real_survey)

        for fk, channel in list(survey.freq_channels.items()):
            ds = channel.dataset
            ds = apply_tvg(ds)
            ds = compute_geometric_ranges(ds)
            ds = compute_georeferencing(ds)
            survey.add_frequency(fk, ds)

        for fk in survey.frequencies:
            ch = survey.get_frequency(fk)
            lat = ch.dataset["latitude_per_sample"].values
            lon = ch.dataset["longitude_per_sample"].values
            valid_lat = lat[np.isfinite(lat)]
            valid_lon = lon[np.isfinite(lon)]
            assert valid_lat.size > 0
            assert 43.0 <= valid_lat.min() <= 45.0
            assert -70.0 <= valid_lon.min() <= -67.0


# ============================================================
# HDF5 ROUND-TRIP
# ============================================================

@pytest.mark.integration
class TestRoundTrip:
    def test_write_and_read_preserves_survey(self, real_survey, tmp_path_factory):
        """SDF → survey → HDF5 → survey round-trip."""
        out_path = tmp_path_factory.mktemp("sdf_roundtrip") / "test.ss.h5"
        write_survey(real_survey, out_path)

        loaded = read_survey(out_path)
        assert loaded.n_frequencies == real_survey.n_frequencies

        for fk in real_survey.frequencies:
            orig = real_survey.get_frequency(fk).dataset
            loaded_ds = loaded.get_frequency(fk).dataset
            assert orig.sizes == loaded_ds.sizes

    def test_processed_survey_round_trip(self, real_survey, tmp_path_factory):
        """Processed survey (with all pipeline outputs) should round-trip."""
        survey = interpolate_aux_onto_pings(real_survey)
        for fk, channel in list(survey.freq_channels.items()):
            ds = channel.dataset
            ds = apply_tvg(ds)
            ds = compute_geometric_ranges(ds)
            ds = compute_georeferencing(ds)
            survey.add_frequency(fk, ds)

        out_path = tmp_path_factory.mktemp("sdf_processed") / "processed.ss.h5"
        write_survey(survey, out_path)

        loaded = read_survey(out_path)
        for fk in survey.frequencies:
            loaded_ds = loaded.get_frequency(fk).dataset
            assert "latitude_per_sample" in loaded_ds.data_vars
            assert "ground_range_m" in loaded_ds.data_vars


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================

@pytest.mark.integration
def test_print_pipeline_summary(real_survey, capsys):
    """Print survey summary when running with -s flag."""
    print()
    print("=" * 60)
    print(f"SDF Survey: {real_survey}")
    print("=" * 60)
    for fk in real_survey.frequencies:
        ch = real_survey.get_frequency(fk)
        ds = ch.dataset
        print(f"\n--- {fk} ---")
        print(f"  Pings × Channels × Samples: {dict(ds.sizes)}")
        print(f"  Sample rate: {ds.attrs.get('sample_rate_hz', 'unknown')} Hz")
        print(f"  Processing level: {ds.attrs.get('processing_level', 'unknown')}")

        bs = ds["backscatter_intensity"].values
        finite = bs[np.isfinite(bs)]
        if finite.size:
            print(f"  Backscatter: min={finite.min():.0f}, max={finite.max():.0f}, "
                  f"mean={finite.mean():.0f}")

        lats = ds["latitude"].values
        valid_lats = lats[np.isfinite(lats)]
        if valid_lats.size:
            print(f"  Latitude range: {valid_lats.min():.4f}° to {valid_lats.max():.4f}°")
        lons = ds["longitude"].values
        valid_lons = lons[np.isfinite(lons)]
        if valid_lons.size:
            print(f"  Longitude range: {valid_lons.min():.4f}° to {valid_lons.max():.4f}°")
