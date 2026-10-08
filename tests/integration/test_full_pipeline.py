"""
End-to-end integration test for the full SideScanPy processing pipeline.

Runs every step against a real JSF fixture file:
    read_jsf → convert → interpolate_aux_onto_pings →
    apply_tvg → compute_geometric_ranges → compute_georeferencing →
    write_survey → read_survey

Verifies that:
  - The pipeline runs to completion without errors
  - Each output dataset still validates against the schema
  - Per-sample variables have plausible distributions and ranges
  - Round-trip through HDF5 preserves the new variables

Marked @pytest.mark.integration; skipped by default. Run with
`pytest --slow`.

2026 rshomberg@uri.edu
"""

import json
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.jsf import convert, read_jsf
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
)

SAMPLE_JSF = Path(__file__).resolve().parents[1] / "data" / "jsf" / "sample.jsf"


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def real_survey():
    """Parse JSF + convert to survey once per test module."""
    if not SAMPLE_JSF.exists():
        pytest.skip(f"Sample JSF file not found at {SAMPLE_JSF}")
    messages = read_jsf(SAMPLE_JSF)
    return convert(messages, source_files=[SAMPLE_JSF])


@pytest.fixture(scope="module")
def processed_survey(real_survey):
    """Run the full processing pipeline once per test module."""
    survey = interpolate_aux_onto_pings(real_survey)
    for fk, channel in list(survey.freq_channels.items()):
        ds = channel.dataset
        ds = apply_tvg(ds)
        ds = compute_geometric_ranges(ds)
        ds = compute_georeferencing(ds)
        survey.add_frequency(fk, ds)
    return survey


@pytest.fixture(scope="module")
def written_path(processed_survey, tmp_path_factory):
    """Write the fully-processed survey to a fresh temp file."""
    p = tmp_path_factory.mktemp("pipeline") / "processed.ss.h5"
    write_survey(processed_survey, p)
    return p


# ============================================================
# PIPELINE COMPLETION
# ============================================================

@pytest.mark.integration
class TestPipelineCompletion:
    def test_processed_survey_has_frequency_channels(self, processed_survey):
        assert processed_survey.n_frequencies >= 1

    def test_each_channel_has_all_processed_variables(self, processed_survey):
        """Every channel should have the full pipeline's outputs."""
        expected_vars = {
            "slant_range_m",
            "ground_range_m",
            "across_track_m",
            "along_track_m",
            "vertical_offset_m",
            "depth_per_sample_m",
            "latitude_per_sample",
            "longitude_per_sample",
        }
        for fk, channel in processed_survey.freq_channels.items():
            present = set(channel.dataset.data_vars)
            missing = expected_vars - present
            assert not missing, f"Channel {fk} missing vars: {missing}"

    def test_processing_history_records_all_steps(self, processed_survey):
        """Each channel should record TVG, geometric, georef steps."""
        for _fk, channel in processed_survey.freq_channels.items():
            history = json.loads(channel.dataset.attrs["processing_history"])
            steps = [entry["step"] for entry in history]
            assert "tvg" in steps
            assert "geometric" in steps
            assert "georef" in steps

    def test_processing_level_derived(self, processed_survey):
        """processing_level is derived from history: tvg+geometric+georef -> georef."""
        for _fk, channel in processed_survey.freq_channels.items():
            assert channel.dataset.sidescan.processing_level == "georef"

    def test_each_channel_validates(self, processed_survey):
        for _fk, channel in processed_survey.freq_channels.items():
            channel.dataset.sidescan.validate()


# ============================================================
# RADIOMETRIC OUTPUT SANITY
# ============================================================

@pytest.mark.integration
class TestRadiometricOutput:
    def test_backscatter_in_db_range(self, processed_survey):
        """backscatter_intensity after TVG should be in plausible dB range."""
        for fk, channel in processed_survey.freq_channels.items():
            data = channel.dataset["backscatter_intensity"].values
            finite = data[np.isfinite(data)]
            # Sidescan dB values typically in -150 to +60 range
            assert finite.min() > -200, f"{fk}: min too low ({finite.min()})"
            assert finite.max() < 100,  f"{fk}: max too high ({finite.max()})"

    def test_tvg_inverts_to_original(self, real_survey, processed_survey):
        """No raw copy is kept: inverting the recorded TVG recovers the input."""
        from sidescanpy.processing import uncorrected_backscatter

        for fk in real_survey.frequencies:
            orig = real_survey.get_frequency(fk).dataset["backscatter_intensity"].values
            back = uncorrected_backscatter(processed_survey.get_frequency(fk).dataset).values
            np.testing.assert_allclose(back, orig, rtol=1e-5, atol=1e-6)


# ============================================================
# GEOMETRIC OUTPUT SANITY
# ============================================================

@pytest.mark.integration
class TestGeometricOutput:
    def test_slant_range_increases_with_sample(self, processed_survey):
        """slant_range_m must be monotonic along sample axis."""
        for fk, channel in processed_survey.freq_channels.items():
            slant = channel.dataset["slant_range_m"].values
            # Check first ping, port channel
            diffs = np.diff(slant[0, 0, :])
            assert np.all(diffs >= 0), f"{fk}: slant range not monotonic"

    def test_ground_range_has_water_column_nans(self, processed_survey):
        """ground_range_m should be NaN for water-column samples."""
        for fk, channel in processed_survey.freq_channels.items():
            ground = channel.dataset["ground_range_m"].values
            assert np.any(np.isnan(ground)), \
                f"{fk}: expected some NaN in ground_range_m (water column)"

    def test_across_track_signed_correctly(self, processed_survey):
        """Port channel = negative across_track, starboard = positive."""
        for fk, channel in processed_survey.freq_channels.items():
            across = channel.dataset["across_track_m"].values
            port = across[:, 0, :]    # channel 0 = port
            stbd = across[:, 1, :]    # channel 1 = starboard

            port_finite = port[np.isfinite(port)]
            stbd_finite = stbd[np.isfinite(stbd)]

            assert np.all(port_finite <= 0), f"{fk}: port across_track has positive values"
            assert np.all(stbd_finite >= 0), f"{fk}: stbd across_track has negative values"

    def test_ground_range_within_max_slant(self, processed_survey):
        """ground_range should never exceed slant_range."""
        for fk, channel in processed_survey.freq_channels.items():
            slant = channel.dataset["slant_range_m"].values
            ground = channel.dataset["ground_range_m"].values
            valid = np.isfinite(ground)
            assert np.all(ground[valid] <= slant[valid] + 1e-3), \
                f"{fk}: ground exceeds slant"


# ============================================================
# GEOREFERENCING OUTPUT SANITY
# ============================================================

@pytest.mark.integration
class TestGeoreferencingOutput:
    def test_lat_per_sample_in_valid_range(self, processed_survey):
        for fk, channel in processed_survey.freq_channels.items():
            lat = channel.dataset["latitude_per_sample"].values
            finite = lat[np.isfinite(lat)]
            assert finite.size > 0, f"{fk}: no finite per-sample latitudes"
            assert np.all(finite >= -90), f"{fk}: lat below -90"
            assert np.all(finite <= 90),  f"{fk}: lat above 90"

    def test_lon_per_sample_in_valid_range(self, processed_survey):
        for _fk, channel in processed_survey.freq_channels.items():
            lon = channel.dataset["longitude_per_sample"].values
            finite = lon[np.isfinite(lon)]
            assert finite.size > 0
            assert np.all(finite >= -180)
            assert np.all(finite <= 180)

    def test_per_sample_positions_cluster_near_track(self, processed_survey):
        """Per-sample positions should be within reasonable distance of
        the per-ping track. Loose check: lat/lon range across all samples
        is similar to lat/lon range across pings (within a few tenths of
        a degree, since swath is sub-km)."""
        for fk, channel in processed_survey.freq_channels.items():
            ds = channel.dataset
            ping_lat = ds["latitude"].values
            ping_lon = ds["longitude"].values

            sample_lat = ds["latitude_per_sample"].values
            sample_lon = ds["longitude_per_sample"].values

            sample_lat_finite = sample_lat[np.isfinite(sample_lat)]
            sample_lon_finite = sample_lon[np.isfinite(sample_lon)]

            # Per-sample bounding box should not be wildly larger than
            # per-ping bounding box plus a swath width's worth of degrees.
            # 0.05° ≈ 5 km is a generous bound on swath spread.
            ping_lat_span = ping_lat.max() - ping_lat.min()
            ping_lon_span = ping_lon.max() - ping_lon.min()
            sample_lat_span = sample_lat_finite.max() - sample_lat_finite.min()
            sample_lon_span = sample_lon_finite.max() - sample_lon_finite.min()

            assert sample_lat_span < ping_lat_span + 0.05, \
                f"{fk}: sample lat span {sample_lat_span} unreasonably large"
            assert sample_lon_span < ping_lon_span + 0.05, \
                f"{fk}: sample lon span {sample_lon_span} unreasonably large"

    def test_starboard_lat_lon_differs_from_port(self, processed_survey):
        """Port and starboard sample positions should differ — they look
        in opposite directions."""
        for fk, channel in processed_survey.freq_channels.items():
            lat = channel.dataset["latitude_per_sample"].values
            # Pick a ping with finite values in both channels at the same sample
            for ping_idx in range(channel.n_pings):
                for sample_idx in range(channel.n_samples):
                    p_lat = lat[ping_idx, 0, sample_idx]
                    s_lat = lat[ping_idx, 1, sample_idx]
                    if np.isfinite(p_lat) and np.isfinite(s_lat):
                        assert p_lat != s_lat, \
                            f"{fk}: port and starboard sample at same point"
                        return
            pytest.fail(f"{fk}: no valid samples to compare")


# ============================================================
# HDF5 ROUND-TRIP WITH ALL NEW VARIABLES
# ============================================================

@pytest.mark.integration
class TestRoundTrip:
    def test_processed_survey_writes_to_disk(self, written_path):
        assert written_path.exists()
        assert written_path.stat().st_size > 0

    def test_processed_survey_reads_back(self, written_path):
        loaded = read_survey(written_path)
        assert loaded is not None
        assert loaded.n_frequencies >= 1

    def test_processed_variables_round_trip(self, processed_survey, written_path):
        """All new per-sample variables must survive HDF5 round-trip."""
        loaded = read_survey(written_path)

        new_vars = (
            "slant_range_m",
            "ground_range_m",
            "across_track_m",
            "along_track_m",
            "vertical_offset_m",
            "depth_per_sample_m",
            "latitude_per_sample",
            "longitude_per_sample",
        )

        for fk in processed_survey.frequencies:
            orig_ds = processed_survey.get_frequency(fk).dataset
            loaded_ds = loaded.get_frequency(fk).dataset
            for var in new_vars:
                assert var in loaded_ds.data_vars, \
                    f"{fk}: {var} missing after round-trip"
                np.testing.assert_array_equal(
                    loaded_ds[var].values,
                    orig_ds[var].values,
                    err_msg=f"{fk}: {var} differs after round-trip",
                )

    def test_processing_history_round_trips(self, processed_survey, written_path):
        loaded = read_survey(written_path)
        for fk in processed_survey.frequencies:
            orig_history = processed_survey.get_frequency(fk).dataset.sidescan.processing_history
            loaded_history = loaded.get_frequency(fk).dataset.sidescan.processing_history
            assert len(loaded_history) == len(orig_history)
            for orig_step, loaded_step in zip(orig_history, loaded_history):
                assert orig_step["step"] == loaded_step["step"]


# ============================================================
# DIAGNOSTIC PRINTOUT (helpful when debugging)
# ============================================================

@pytest.mark.integration
def test_print_pipeline_summary(processed_survey, capsys):
    """Not a real test — prints a summary so test output is informative.
    Run with `pytest -s` to see the output."""
    print()
    print("=" * 60)
    print(f"Processed survey: {processed_survey}")
    print("=" * 60)
    for fk in processed_survey.frequencies:
        ch = processed_survey.get_frequency(fk)
        ds = ch.dataset
        print(f"\n--- {fk} ---")
        print(f"Pings × Channels × Samples: {ds.sizes}")
        print(f"Processing level: {ds.sidescan.processing_level}")

        # Backscatter sanity
        bs = ds["backscatter_intensity"].values
        finite = bs[np.isfinite(bs)]
        if finite.size:
            print(f"Backscatter (dB): "
                  f"min={finite.min():.1f}  max={finite.max():.1f}  "
                  f"mean={finite.mean():.1f}  median={np.median(finite):.1f}")

        # Geometric sanity
        ground = ds["ground_range_m"].values
        ground_finite = ground[np.isfinite(ground)]
        if ground_finite.size:
            print(f"Ground range (m): "
                  f"min={ground_finite.min():.2f}  max={ground_finite.max():.2f}  "
                  f"NaN={np.isnan(ground).sum()}/{ground.size} ({100*np.isnan(ground).sum()/ground.size:.1f}%)")

        # Lat/lon sanity
        lat = ds["latitude_per_sample"].values
        lon = ds["longitude_per_sample"].values
        lat_finite = lat[np.isfinite(lat)]
        lon_finite = lon[np.isfinite(lon)]
        if lat_finite.size:
            print(f"Per-sample lat range: {lat_finite.min():.6f} to {lat_finite.max():.6f}")
            print(f"Per-sample lon range: {lon_finite.min():.6f} to {lon_finite.max():.6f}")
