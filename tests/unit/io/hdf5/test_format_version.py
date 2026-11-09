"""
Tests for the ``.ss.h5`` format version check on read.

2026 rshomberg@uri.edu
"""

import logging

import h5netcdf

from sidescanpy.core.schema import SS_FORMAT_VERSION
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.hdf5.reader import check_format_version


class TestCheckFormatVersion:
    def test_returns_none_when_attribute_missing(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sidescanpy"):
            assert check_format_version({}, "old.ss.h5") is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_no_warning_when_current_version(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            assert check_format_version({"format_version": SS_FORMAT_VERSION}) == SS_FORMAT_VERSION
        assert not caplog.records

    def test_warns_when_newer_version(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            check_format_version({"format_version": "99.0"}, "new.ss.h5")
        assert any("newer" in r.getMessage() for r in caplog.records)

    def test_warns_when_unparseable_version(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            assert check_format_version({"format_version": "draft"}) == "draft"
        assert any("unrecognised" in r.getMessage() for r in caplog.records)


class TestReadSurveyChecksVersion:
    def test_read_warns_when_file_newer(self, sample_full_survey, tmp_path, caplog):
        p = tmp_path / "survey.ss.h5"
        write_survey(sample_full_survey, p)
        with h5netcdf.File(str(p), "a") as f:
            f.attrs["format_version"] = "99.0"
        with caplog.at_level(logging.WARNING, logger="sidescanpy"):
            read_survey(p)
        assert any("newer" in r.getMessage() for r in caplog.records)
