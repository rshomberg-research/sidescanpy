"""
Unit tests for sidescanpy.io.dispatch — format detection and
read+convert dispatch.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import pytest

from sidescanpy.io.dispatch import (
    CATALOG_JSF_DISCOVER,
    CATALOG_SDF_KLEIN,
    detect_catalog_format,
    detect_format,
    find_sonar_files,
    load_survey,
    read_and_convert,
    read_target_catalog,
)

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

    def test_xtf_any_case(self):
        assert detect_format([Path("survey.xtf")]) == "xtf"
        assert detect_format([Path("A.XTF"), Path("b.Xtf")]) == "xtf"

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match=r"Unsupported format.*\.xtf"):
            detect_format([Path("survey.s7k")])


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

    def test_xtf_dispatch_reads_and_converts_all_files(self, monkeypatch):
        def fake_read_xtf(p):
            return f"pings-from-{p.name}"

        def fake_xtf_convert(pings, *, source_files, survey_name):
            return {"pings": pings, "source_files": source_files, "survey_name": survey_name}

        monkeypatch.setattr("sidescanpy.io.dispatch.read_xtf", fake_read_xtf)
        monkeypatch.setattr("sidescanpy.io.dispatch.xtf_convert", fake_xtf_convert)

        paths = [Path("L1.XTF"), Path("L2.XTF")]
        result = read_and_convert(paths, "xtf")

        assert result["pings"] == ["pings-from-L1.XTF", "pings-from-L2.XTF"]
        assert result["source_files"] == paths
        assert result["survey_name"] == "L1"

    def test_xtf_end_to_end_on_synthetic_file(self, tmp_path):
        from tests.unit.io.xtf.conftest import (
            dual_frequency_infos,
            file_header,
            ramp,
            sonar_packet,
            write_xtf,
        )

        pk = [sonar_packet([(c, ramp(32, "<u2", i + c)) for c in range(4)], ping_number=i)
              for i in range(3)]
        path = write_xtf(tmp_path / "LINE.XTF", file_header(dual_frequency_infos()), pk)
        survey = read_and_convert([path], detect_format([path]))
        assert survey.source_format == "xtf"
        assert survey.frequencies == ["hz_100000", "hz_400000"]
        assert load_survey(path).frequencies == survey.frequencies

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            read_and_convert([Path("a.jsf")], "s7k")


class TestFindSonarFiles:
    def test_auto_finds_all_formats_sorted(self, tmp_path):
        for name in ("b.sdf", "a.jsf", "c.txt", "d.XTF", "e.xtf"):
            (tmp_path / name).write_bytes(b"")
        assert [p.name for p in find_sonar_files(tmp_path)] == [
            "a.jsf", "b.sdf", "d.XTF", "e.xtf",
        ]

    def test_xtf_any_case(self, tmp_path):
        for name in ("a.XTF", "b.xtf", "c.Xtf", "d.jsf"):
            (tmp_path / name).write_bytes(b"")
        (tmp_path / "dir.xtf").mkdir()
        assert [p.name for p in find_sonar_files(tmp_path, "xtf")] == ["a.XTF", "b.xtf", "c.Xtf"]

    def test_single_format(self, tmp_path):
        for name in ("b.sdf", "a.jsf"):
            (tmp_path / name).write_bytes(b"")
        assert [p.name for p in find_sonar_files(tmp_path, "sdf")] == ["b.sdf"]

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="input_format"):
            find_sonar_files(tmp_path, "s7k")


class TestLoadSurvey:
    def test_reads_ss_h5(self, tmp_path, sample_multi_frequency_survey):
        from sidescanpy.io.hdf5 import write_survey

        write_survey(sample_multi_frequency_survey, tmp_path / "a.ss.h5")
        survey = load_survey(tmp_path / "a.ss.h5")
        assert set(survey.frequencies) == set(sample_multi_frequency_survey.frequencies)

    def test_raw_file_is_converted(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        import sidescanpy.io.dispatch as dispatch

        calls = []
        monkeypatch.setattr(dispatch, "read_and_convert", lambda paths, fmt: calls.append((paths, fmt)) or sample_multi_frequency_survey)
        assert load_survey(tmp_path / "x.sdf") is sample_multi_frequency_survey
        assert calls == [([tmp_path / "x.sdf"], "sdf")]


class TestTargetCatalog:
    def test_klein(self, tmp_path):
        (tmp_path / "Target_1536923484.tgt").write_bytes(b"")
        assert detect_catalog_format(tmp_path) == CATALOG_SDF_KLEIN

    def test_jsf(self, tmp_path):
        (tmp_path / "Target-0001.target").write_bytes(b"")
        assert detect_catalog_format(tmp_path) == CATALOG_JSF_DISCOVER

    def test_ambiguous_raises(self, tmp_path):
        (tmp_path / "Target_1536923484.tgt").write_bytes(b"")
        (tmp_path / "Target-0001.target").write_bytes(b"")
        with pytest.raises(ValueError, match="ambiguous"):
            detect_catalog_format(tmp_path)

    def test_neither_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No target catalog files"):
            detect_catalog_format(tmp_path)

    def test_reads_jsf_catalog(self, tmp_path):
        from tests.unit.io.jsf.test_targets import _make_target_file

        _make_target_file(tmp_path / "Target-0001.target")
        fmt, targets = read_target_catalog(tmp_path)
        assert fmt == CATALOG_JSF_DISCOVER
        assert [t.target_id for t in targets] == ["0001"] or len(targets) == 1

    def test_reads_klein_catalog(self, tmp_path):
        from tests.unit.io.sdf.test_targets import _make_window, _write_target

        _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=10, range_m=12.0, window=_make_window())
        fmt, targets = read_target_catalog(tmp_path)
        assert fmt == CATALOG_SDF_KLEIN
        assert len(targets) == 1
        assert targets[0].latitude is None
