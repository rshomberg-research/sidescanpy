"""
Unit tests for sidescanpy.io.hdf5.writer.

Tests focus on writer-side behavior: file creation, overwrite semantics,
on-disk group structure, root attrs, and per-variable encoding (compression).

Round-trip identity is tested separately in test_round_trip.py.

2026 rshomberg@uri.edu
"""

import json

import h5netcdf
import pytest

from sidescanpy.core.schema import (
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_AUX_STREAMS,
    HDF5_GROUP_DVL,
    HDF5_GROUP_FREQUENCIES,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_SENSORS,
)
from sidescanpy.io.hdf5.writer import write_survey

# ============================================================
# FILE CREATION
# ============================================================

@pytest.mark.unit
class TestWriteFileCreation:
    def test_creates_file(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_accepts_path_object(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        assert p.exists()

    def test_accepts_string_path(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, str(p))
        assert p.exists()


# ============================================================
# OVERWRITE SEMANTICS
# ============================================================

@pytest.mark.unit
class TestWriteOverwrite:
    def test_overwrite_false_raises_when_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with pytest.raises(FileExistsError):
            write_survey(sample_full_survey, p)

    def test_overwrite_true_replaces(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        write_survey(sample_full_survey, p, overwrite=True)
        assert p.exists()

    def test_overwrite_default_is_false(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with pytest.raises(FileExistsError):
            write_survey(sample_full_survey, p)


# ============================================================
# GROUP STRUCTURE ON DISK
# ============================================================

@pytest.mark.unit
class TestWriteGroupStructure:
    def test_frequencies_group_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert HDF5_GROUP_FREQUENCIES in f.groups

    def test_aux_streams_group_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert HDF5_GROUP_AUX_STREAMS in f.groups

    def test_each_frequency_has_subgroup(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            freq_groups = list(f[HDF5_GROUP_FREQUENCIES].groups)
        for fk in sample_full_survey.frequencies:
            assert fk in freq_groups

    def test_navigation_subgroup_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert "navigation" in f[HDF5_GROUP_AUX_STREAMS].groups

    def test_attitude_subgroup_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert "attitude" in f[HDF5_GROUP_AUX_STREAMS].groups

    def test_dvl_subgroup_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert "dvl" in f[HDF5_GROUP_AUX_STREAMS].groups

    def test_sensors_subgroup_exists_when_sensors_present(
        self, sample_full_survey, tmp_path
    ):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert "sensors" in f[HDF5_GROUP_AUX_STREAMS].groups

    def test_sensors_subgroup_omitted_when_no_sensors(
        self, sample_multi_frequency_survey, tmp_path
    ):
        """sample_multi_frequency_survey has no sensors. Parent group should
        not be created."""
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_multi_frequency_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            aux_subgroups = list(f[HDF5_GROUP_AUX_STREAMS].groups)
        assert "sensors" not in aux_subgroups

    def test_individual_sensor_subgroup_exists(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            sensor_groups = list(f[HDF5_GROUP_SENSORS].groups)
        for sensor_type in sample_full_survey.sensor_types:
            assert sensor_type in sensor_groups


# ============================================================
# ROOT ATTRS
# ============================================================

@pytest.mark.unit
class TestWriteRootAttrs:
    def test_survey_name_in_root_attrs(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert f.attrs["survey_name"] == sample_full_survey.survey_name

    def test_source_format_in_root_attrs(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            assert f.attrs["source_format"] == sample_full_survey.source_format

    def test_source_files_round_trip_via_json(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "r") as f:
            raw = f.attrs["source_files_json"]
        decoded = json.loads(raw)
        # source_files is list[Path] on the survey; serialized as list of strings
        assert decoded == [str(p) for p in sample_full_survey.source_files]


# ============================================================
# PER-VARIABLE ENCODING (COMPRESSION)
# ============================================================

@pytest.mark.unit
class TestWriteCompression:
    def test_backscatter_is_compressed_per_frequency(
        self, sample_full_survey, tmp_path
    ):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)

        with h5netcdf.File(str(p), "r") as f:
            for fk in sample_full_survey.frequencies:
                grp = f[f"{HDF5_GROUP_FREQUENCIES}/{fk}"]
                for var in ("backscatter_raw", "backscatter_intensity"):
                    assert grp[var].compression == "gzip", (
                        f"{fk}/{var} should be gzip-compressed"
                    )

    def test_per_ping_scalars_not_compressed(self, sample_full_survey, tmp_path):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)

        with h5netcdf.File(str(p), "r") as f:
            fk = sample_full_survey.frequencies[0]
            grp = f[f"{HDF5_GROUP_FREQUENCIES}/{fk}"]
            for var in ("latitude", "longitude", "heading_deg"):
                assert grp[var].compression is None

    def test_aux_records_not_compressed(self, sample_full_survey, tmp_path):
        """Aux records are tiny; compression is overhead."""
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)

        with h5netcdf.File(str(p), "r") as f:
            for var in ("latitude", "longitude"):
                assert f[HDF5_GROUP_NAVIGATION][var].compression is None
            for var in ("roll_deg", "pitch_deg", "heave_m"):
                assert f[HDF5_GROUP_ATTITUDE][var].compression is None
            assert f[HDF5_GROUP_DVL]["altitude_m"].compression is None


# ============================================================
# ENCODING POLICY FOR PER-SAMPLE GEOMETRY
# ============================================================

@pytest.mark.unit
class TestGeometryEncodingPolicy:
    def test_geometry_vars_compressed_chunked_float32(self, sample_survey_channel, tmp_path):
        from sidescanpy.core.survey import SideScanSurvey
        from sidescanpy.processing.geometric import compute_geometric_ranges
        from sidescanpy.processing.georeferencing import compute_georeferencing

        ds = compute_georeferencing(compute_geometric_ranges(sample_survey_channel.dataset))
        # Force a float64 geometry var to prove the downcast.
        ds["across_track_m"] = ds["across_track_m"].astype("float64")
        survey = SideScanSurvey(source_files=[], source_format="test", survey_name="s")
        survey.add_frequency("hz_1", ds)
        p = tmp_path / "s.ss.h5"
        write_survey(survey, p)

        with h5netcdf.File(str(p), "r") as f:
            grp = f[f"{HDF5_GROUP_FREQUENCIES}/hz_1"]
            for var in ("across_track_m", "along_track_m", "ground_range_m",
                        "slant_range_m", "vertical_offset_m", "depth_per_sample_m"):
                v = grp[var]
                assert v.compression == "gzip", var
                assert v.shuffle, var
                assert v.chunks is not None, var
                assert v.dtype == "float32", var
            assert grp["latitude_per_sample"].dtype == "float64"
            assert grp["longitude_per_sample"].dtype == "float64"
            assert grp["latitude_per_sample"].compression == "gzip"
