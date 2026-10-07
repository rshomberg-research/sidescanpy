"""
Unit tests for sidescanpy.io.dispatch — format detection and
read+convert dispatch.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import pytest

from sidescanpy.io.dispatch import detect_format, read_and_convert

# ============================================================
# detect_format
# ============================================================

class TestDetectFormat:
    def test_single_jsf_file(self):
        assert detect_format([Path("survey.jsf")]) == "jsf"

    def test_single_sdf_file(self):
        assert detect_format([Path("survey.sdf")]) == "sdf"

    def test_multiple_jsf_files(self):
        paths = [Path("a.jsf"), Path("b.jsf"), Path("c.jsf")]
        assert detect_format(paths) == "jsf"

    def test_case_insensitive_extension(self):
        assert detect_format([Path("SURVEY.JSF")]) == "jsf"

    def test_mixed_extensions_raises(self):
        paths = [Path("a.jsf"), Path("b.sdf")]
        with pytest.raises(ValueError, match="mixed extensions"):
            detect_format(paths)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            detect_format([Path("survey.xtf")])


# ============================================================
# read_and_convert
# ============================================================

class TestReadAndConvert:
    def test_jsf_dispatch_reads_and_converts_all_files(self, monkeypatch):
        read_calls = []
        convert_calls = []

        def fake_read_jsf(p):
            read_calls.append(p)
            return [f"msg-from-{p.name}"]

        def fake_jsf_convert(messages, *, source_files, survey_name):
            convert_calls.append((messages, source_files, survey_name))
            return "SURVEY"

        monkeypatch.setattr("sidescanpy.io.dispatch.read_jsf", fake_read_jsf)
        monkeypatch.setattr("sidescanpy.io.dispatch.jsf_convert", fake_jsf_convert)

        paths = [Path("a.jsf"), Path("b.jsf")]
        result = read_and_convert(paths, "jsf")

        assert result == "SURVEY"
        assert read_calls == paths
        messages, source_files, survey_name = convert_calls[0]
        assert messages == ["msg-from-a.jsf", "msg-from-b.jsf"]
        assert source_files == paths
        assert survey_name == "a"  # defaults to first file's stem

    def test_sdf_dispatch_reads_and_converts_all_files(self, monkeypatch):
        def fake_read_sdf(p):
            return [f"page-from-{p.name}"]

        def fake_sdf_convert(pages, *, source_files, survey_name):
            return {"pages": pages, "source_files": source_files, "survey_name": survey_name}

        monkeypatch.setattr("sidescanpy.io.dispatch.read_sdf", fake_read_sdf)
        monkeypatch.setattr("sidescanpy.io.dispatch.sdf_convert", fake_sdf_convert)

        paths = [Path("line1.sdf")]
        result = read_and_convert(paths, "sdf", survey_name="custom_name")

        assert result["pages"] == ["page-from-line1.sdf"]
        assert result["survey_name"] == "custom_name"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            read_and_convert([Path("a.jsf")], "xtf")
