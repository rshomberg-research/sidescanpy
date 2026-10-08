"""
Unit tests for sidescanpy.io.hdf5.survey_dir (multi-file directory helpers).

2026 rshomberg@uri.edu
"""

import pytest

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import (
    atomic_write_survey,
    available_frequencies,
    check_disk_space,
    concat_survey_frequency,
    load_channel_dataset,
    read_survey,
    read_survey_dir,
    read_survey_pings,
    write_survey,
)
from tests.conftest import make_minimal_sidescan_dataset


def _write(path, n_pings, freq_keys=("hz_900000",)):
    survey = SideScanSurvey(source_files=["a.sdf"], source_format="SDF", survey_name=path.stem)
    for k in freq_keys:
        survey.add_frequency(k, make_minimal_sidescan_dataset(n_pings=n_pings))
    write_survey(survey, path)
    return survey


@pytest.fixture
def survey_dir(tmp_path):
    _write(tmp_path / "a.ss.h5", 5)
    _write(tmp_path / "b.ss.h5", 7, freq_keys=("hz_900000", "hz_600000"))
    return tmp_path


def test_concat_survey_frequency_reports_file_starts(survey_dir):
    files = sorted(survey_dir.glob("*.ss.h5"))
    ds, starts = concat_survey_frequency(files, "hz_900000", None)
    assert ds.sizes["ping"] == 12
    assert starts == {files[0]: 0, files[1]: 5}


def test_concat_survey_frequency_missing_key_is_none(survey_dir):
    ds, starts = concat_survey_frequency(sorted(survey_dir.glob("*.ss.h5")), "hz_1", None)
    assert ds is None
    assert starts == {}


def test_read_survey_dir_skips_files_lacking_the_key(survey_dir):
    assert read_survey_dir(survey_dir, freq_key="hz_600000").sizes["ping"] == 7
    assert read_survey_dir(survey_dir, freq_key="hz_900000", chunks={"ping": 4}).sizes["ping"] == 12


def test_read_survey_dir_errors(tmp_path, survey_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_survey_dir(empty, freq_key="hz_900000")
    with pytest.raises(ValueError):
        read_survey_dir(survey_dir, freq_key="hz_1")


def test_available_frequencies_union_for_dir_and_single_file(survey_dir):
    assert available_frequencies(survey_dir, {}) == {"hz_900000", "hz_600000"}
    assert available_frequencies(survey_dir / "a.ss.h5", {}) == {"hz_900000"}


def test_load_channel_dataset_file_and_dir(survey_dir):
    assert load_channel_dataset(survey_dir, "hz_900000", {}).sizes["ping"] == 12
    assert load_channel_dataset(survey_dir / "a.ss.h5", "hz_900000", {}).sizes["ping"] == 5
    with pytest.raises(ValueError):
        load_channel_dataset(survey_dir / "a.ss.h5", "hz_600000", {})


def test_check_disk_space(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 100)
    assert check_disk_space(p, 2.0)
    assert not check_disk_space(p, 1e18)


def test_atomic_write_survey_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "a.ss.h5"
    _write(path, 5)
    new = _write(tmp_path / "b.ss.h5", 9)
    atomic_write_survey(new, path)
    assert read_survey(path).get_frequency("hz_900000").dataset.sizes["ping"] == 9
    assert not list(tmp_path.glob("*.tmp"))


def test_read_survey_pings_single_file_cuts_every_channel(tmp_path):
    _write(tmp_path / "a.ss.h5", 10, freq_keys=("hz_900000", "hz_600000"))
    survey = read_survey_pings(tmp_path / "a.ss.h5", 2, 6)
    assert survey.survey_name == "a.ss" and survey.source_format == "SDF"
    for k in ("hz_900000", "hz_600000"):
        ds = survey.get_frequency(k).dataset
        assert ds.sizes["ping"] == 4
        assert ds["ping_number"].values.tolist() == [3, 4, 5, 6]
        assert not hasattr(ds["backscatter_intensity"].data, "chunks")


def test_read_survey_pings_directory(survey_dir):
    survey = read_survey_pings(survey_dir, 3, None)
    assert survey.survey_name == survey_dir.name and survey.source_format == "ss_h5"
    assert [p.name for p in map(__import__("pathlib").Path, survey.source_files)] == ["a.ss.h5", "b.ss.h5"]
    assert survey.get_frequency("hz_900000").dataset.sizes["ping"] == 12 - 3
