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
    read_and_convert_segments,
    read_target_catalog,
    segment_stem,
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

        def fake_xtf_convert(pings, *, source_files, survey_name, nav_crs=None):
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

    @pytest.mark.parametrize("as_list", [True, False])
    def test_str_paths_accepted(self, tmp_path, as_list):
        from tests.unit.io.xtf.conftest import (
            dual_frequency_infos,
            file_header,
            ramp,
            sonar_packet,
            write_xtf,
        )

        pk = [sonar_packet([(c, ramp(32, "<u2", i + c)) for c in range(4)], ping_number=i)
              for i in range(3)]
        path = str(write_xtf(tmp_path / "LINE.XTF", file_header(dual_frequency_infos()), pk))
        arg = [path] if as_list else path
        survey = read_and_convert(arg, detect_format(arg))
        assert survey.survey_name == "LINE"
        assert survey.source_files == [Path(path)]
        assert survey.frequencies == ["hz_100000", "hz_400000"]

    def test_str_paths_for_jsf_and_sdf(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("sidescanpy.io.dispatch.read_jsf", lambda p: seen.setdefault("jsf", p) and [])
        monkeypatch.setattr("sidescanpy.io.dispatch.read_sdf", lambda p: seen.setdefault("sdf", p) and [])
        monkeypatch.setattr("sidescanpy.io.dispatch.jsf_convert", lambda m, *, source_files, survey_name: survey_name)
        monkeypatch.setattr("sidescanpy.io.dispatch.sdf_convert", lambda m, *, source_files, survey_name: survey_name)
        assert read_and_convert(["dir/a.jsf"], "jsf") == "a"
        assert read_and_convert(["dir/b.sdf"], "sdf") == "b"
        assert seen == {"jsf": Path("dir/a.jsf"), "sdf": Path("dir/b.sdf")}

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            read_and_convert([Path("a.jsf")], "s7k")


class TestReadAndConvertSegments:
    @pytest.mark.parametrize("name, fmt", [("a.jsf", "jsf")])
    def test_jsf_unchanged_single_survey(self, monkeypatch, name, fmt):
        calls = []
        monkeypatch.setattr(
            "sidescanpy.io.dispatch.read_and_convert",
            lambda paths, f, survey_name=None: calls.append((paths, f, survey_name)) or "SURVEY",
        )
        path = Path(name)
        assert read_and_convert_segments(path) == [(path.stem, "SURVEY")]
        assert calls == [([path], fmt, path.stem)]

    def test_jsf_survey_name_override(self, monkeypatch):
        monkeypatch.setattr(
            "sidescanpy.io.dispatch.read_and_convert", lambda paths, f, survey_name=None: survey_name
        )
        assert read_and_convert_segments(Path("a.jsf"), survey_name="x") == [("x", "x")]

    def _xtf(self, tmp_path, ranges):
        from tests.unit.io.xtf.conftest import (
            dual_frequency_infos,
            file_header,
            ramp,
            sonar_packet,
            write_xtf,
        )

        pk = [
            sonar_packet([(c, ramp(32, "<u2", i + c)) for c in range(4)], ping_number=i,
                         chan_fields={"SlantRange": r, "TimeDuration": r / 750.0})
            for i, r in enumerate(ranges)
        ]
        return write_xtf(tmp_path / "LINE.XTF", file_header(dual_frequency_infos()), pk)

    def test_xtf_single_rate_named_after_stem(self, tmp_path):
        path = self._xtf(tmp_path, [100.0] * 5)
        ((stem, survey),) = read_and_convert_segments(path)
        assert stem == "LINE"
        assert survey.survey_name == "LINE"
        assert survey.source_files == [path]
        assert survey.get_frequency("hz_100000").dataset.attrs["xtf_segment_count"] == 1

    def test_xtf_range_change_split_into_segments(self, tmp_path):
        path = self._xtf(tmp_path, [100.0] * 6 + [150.0] * 2 + [200.0] * 5)
        out = read_and_convert_segments(path, min_pings=4)
        assert [stem for stem, _ in out] == ["LINE_seg01", "LINE_seg02"]
        assert [s.survey_name for _, s in out] == ["LINE_seg01", "LINE_seg02"]
        ds = [s.get_frequency("hz_100000").dataset for _, s in out]
        assert [d.sizes["ping"] for d in ds] == [6, 5]
        assert ds[0].attrs["sample_rate_hz"] == pytest.approx(32 * 750.0 / 100.0)
        assert ds[1].attrs["sample_rate_hz"] == pytest.approx(32 * 750.0 / 200.0)
        assert ds[1].attrs["xtf_pings_dropped_short_runs"] == 2

    def test_xtf_all_runs_short_raises(self, tmp_path):
        path = self._xtf(tmp_path, [100.0] * 3 + [200.0] * 3)
        with pytest.raises(ValueError, match="at least 50 pings"):
            read_and_convert_segments(path)

    def test_segment_stem(self):
        assert segment_stem("a", 1, 1) == "a"
        assert segment_stem("a", 2, 3) == "a_seg02"
        assert segment_stem("a", 7, 120) == "a_seg007"


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
