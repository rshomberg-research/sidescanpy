"""
End-to-end integration test: JSF → SideScanSurvey → HDF5 → SideScanSurvey.

Proves that the full IO stack preserves information across the conversion
+ serialization boundary. If this passes, the module is ready for use on
real survey data.

Marked @pytest.mark.integration; skipped by default. Run with `pytest --slow`.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.jsf import read_jsf
from sidescanpy.io.jsf.converter import convert

SAMPLE_JSF = Path(__file__).resolve().parents[1] / "data" / "jsf" / "sample.jsf"


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def real_survey():
    if not SAMPLE_JSF.exists():
        pytest.skip(f"Sample JSF file not found at {SAMPLE_JSF}")
    messages = read_jsf(SAMPLE_JSF)
    return convert(messages, source_files=[SAMPLE_JSF])


@pytest.fixture
def written_path(real_survey, tmp_path):
    """Write the real survey to a fresh temp file and return the path."""
    p = tmp_path / "survey.ss.h5"
    write_survey(real_survey, p)
    return p


# ============================================================
# WRITE
# ============================================================

@pytest.mark.integration
class TestWriteRealSurvey:
    def test_file_created(self, written_path):
        assert written_path.exists()
        assert written_path.stat().st_size > 0

    def test_file_size_reasonable(self, real_survey, written_path):
        """File size should be roughly the size of the backscatter data
        (compressed). Sanity check that compression is actually happening."""
        backscatter_bytes = sum(
            ch.n_pings * ch.n_channels * ch.n_samples * 4 * 2  # 2 vars, float32
            for ch in real_survey.freq_channels.values()
        )
        # Compressed file should be at most ~half the raw size for typical sonar
        # data. Loose bound — tighten if compression regresses.
        assert written_path.stat().st_size < backscatter_bytes


# ============================================================
# READ + ROUND-TRIP IDENTITY
# ============================================================

@pytest.mark.integration
class TestRoundTripIdentity:
    def test_frequency_datasets_identical(self, real_survey, written_path):
        loaded = read_survey(written_path)
        for fk in real_survey.frequencies:
            a = real_survey.get_frequency(fk).dataset
            b = loaded.get_frequency(fk).dataset
            xr.testing.assert_identical(a, b)

    def test_navigation_identical(self, real_survey, written_path):
        loaded = read_survey(written_path)
        if real_survey.navigation is not None:
            xr.testing.assert_identical(loaded.navigation, real_survey.navigation)

    def test_attitude_identical(self, real_survey, written_path):
        loaded = read_survey(written_path)
        if real_survey.attitude is not None:
            xr.testing.assert_identical(loaded.attitude, real_survey.attitude)

    def test_dvl_identical(self, real_survey, written_path):
        loaded = read_survey(written_path)
        if real_survey.dvl is not None:
            xr.testing.assert_identical(loaded.dvl, real_survey.dvl)

    def test_sensors_identical(self, real_survey, written_path):
        loaded = read_survey(written_path)
        assert sorted(loaded.sensor_types) == sorted(real_survey.sensor_types)
        for sensor_type in real_survey.sensor_types:
            xr.testing.assert_identical(
                loaded.get_sensor(sensor_type),
                real_survey.get_sensor(sensor_type),
            )

    def test_metadata_preserved(self, real_survey, written_path):
        loaded = read_survey(written_path)
        assert loaded.survey_name   == real_survey.survey_name
        assert loaded.source_format == real_survey.source_format
        assert [Path(s) for s in loaded.source_files] == \
               [Path(s) for s in real_survey.source_files]


# ============================================================
# LAZY-LOADING ROUND-TRIP
# ============================================================

@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks")
class TestRoundTripLazy:
    def test_chunks_param_returns_dask_arrays(self, written_path):
        loaded = read_survey(written_path, chunks={"ping": 256})
        for fk in loaded.frequencies:
            backscatter = loaded.get_frequency(fk).dataset["backscatter_intensity"].variable.data
            assert isinstance(backscatter, da.Array), (
                f"Frequency {fk} backscatter should be dask-backed under chunks"
            )

    def test_lazy_data_equals_eager(self, real_survey, written_path):
        eager = read_survey(written_path)
        lazy  = read_survey(written_path, chunks={"ping": 256})

        for fk in real_survey.frequencies:
            np.testing.assert_array_equal(
                lazy.get_frequency(fk).dataset["backscatter_intensity"].values,   # forces compute
                eager.get_frequency(fk).dataset["backscatter_intensity"].values,
            )

    def test_aux_records_loaded_eagerly_under_chunks(self, written_path):
        loaded = read_survey(written_path, chunks={"ping": 256})
        if loaded.has_navigation:
            data = loaded.navigation["latitude"].variable.data
            assert isinstance(data, np.ndarray), (
                "Navigation should not be dask-backed even when chunks= is given"
            )
