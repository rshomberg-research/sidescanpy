"""
Unit tests for sidescanpy.io.hdf5.reader.

Tests focus on reader-side behavior: returns SideScanSurvey, handles missing
files, lazy loading via chunks param.

Round-trip identity is tested in test_round_trip.py.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np
import pytest

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5.reader import read_survey
from sidescanpy.io.hdf5.writer import write_survey


# ============================================================
# RETURN TYPE
# ============================================================

@pytest.mark.unit
class TestReadReturnType:
    def test_returns_sidescan_survey(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert isinstance(loaded, SideScanSurvey)

    def test_loaded_frequency_channels_validate(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        for fk in loaded.frequencies:
            loaded.get_frequency(fk).dataset.sidescan.validate()

    def test_loaded_aux_records_validate(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        if loaded.has_navigation:
            loaded.navigation.navigation_record.validate()
        if loaded.has_attitude:
            loaded.attitude.attitude_record.validate()
        if loaded.has_dvl:
            loaded.dvl.dvl_record.validate()
        for sensor_type in loaded.sensor_types:
            loaded.get_sensor(sensor_type).sensor_record.validate()


# ============================================================
# ERROR HANDLING
# ============================================================

@pytest.mark.unit
class TestReadErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_survey(tmp_path / "does_not_exist.ss.h5")

    def test_accepts_path_object(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded is not None

    def test_accepts_string_path(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(str(p))
        assert loaded is not None


# ============================================================
# LAZY LOADING (chunks parameter)
# ============================================================

@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks")
class TestReadLazyLoading:
    def test_default_load_is_eager(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)

        fk = loaded.frequencies[0]
        backscatter = loaded.get_frequency(fk).dataset["backscatter_raw"].variable.data
        assert isinstance(backscatter, np.ndarray)

    def test_chunks_param_returns_dask(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p, chunks={"ping": 5})

        import dask.array as da
        for fk in loaded.frequencies:
            backscatter = loaded.get_frequency(fk).dataset["backscatter_raw"].variable.data
            assert isinstance(backscatter, da.Array), (
                f"Frequency {fk} backscatter should be dask-backed under chunks"
            )

    def test_lazy_data_equals_eager(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        eager = read_survey(p)
        lazy  = read_survey(p, chunks={"ping": 5})
        for fk in eager.frequencies:
            np.testing.assert_array_equal(
                lazy.get_frequency(fk).dataset["backscatter_raw"].values,
                eager.get_frequency(fk).dataset["backscatter_raw"].values,
            )

    def test_aux_records_loaded_eagerly_even_with_chunks(
        self, sample_full_survey, tmp_path
    ):
        """Aux records are tiny; they should not be dask-wrapped even when
        chunks= is specified for the sonar data."""
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p, chunks={"ping": 5})

        if loaded.has_navigation:
            data = loaded.navigation["latitude"].variable.data
            assert isinstance(data, np.ndarray), (
                "Navigation record should not be dask-backed"
            )


# ============================================================
# READ-ONLY PROPERTIES (no write needed if pre-written file exists)
# ============================================================

@pytest.mark.unit
class TestReadProperties:
    def test_frequencies_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        # Use sorted comparison; HDF5 group ordering isn't guaranteed semantic
        assert sorted(loaded.frequencies) == sorted(sample_full_survey.frequencies)

    def test_sensor_types_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert sorted(loaded.sensor_types) == sorted(sample_full_survey.sensor_types)

    def test_n_frequencies_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded.n_frequencies == sample_full_survey.n_frequencies

    def test_survey_name_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded.survey_name == sample_full_survey.survey_name

    def test_source_format_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        assert loaded.source_format == sample_full_survey.source_format

    def test_source_files_match(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        loaded = read_survey(p)
        # source_files is list[Path]; serialization round-trips as strings
        # converted back to Path. Compare as Path objects.
        assert [Path(x) for x in loaded.source_files] == \
               [Path(x) for x in sample_full_survey.source_files]