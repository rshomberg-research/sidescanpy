"""
Unit tests for sidescan-convert — survey (.jsf/.sdf/.xtf -> .ss.h5) and
targets (catalog -> targets.json) subcommands, replacing the old
per-project 01_convert.py / 11_read_targets.py scripts.

CLI orchestration is tested here with the underlying format readers
monkeypatched out (already covered by their own module's tests in
tests/unit/io/) so these tests stay fast and focused on dispatch,
flag-wiring, and file I/O.

2026 rshomberg@uri.edu
"""

import argparse

import pytest

from sidescanpy.cli import convert
from sidescanpy.core.target import Target
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.targets.io import read_targets_csv, read_targets_json


def _target(target_id, lat=None, lon=None, **extra_attrs):
    return Target(
        target_id=target_id, source_format="x", catalog_source="x",
        latitude=lat, longitude=lon, position_confidence="unknown",
        extra_attributes=extra_attrs,
    )


class TestRunSurvey:
    def _args(self, tmp_path, **overrides):
        defaults = dict(
            input_dir=str(tmp_path / "raw"), output_dir=str(tmp_path / "ss_h5"),
            input_format="auto", force=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_converts_new_file(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "20241017181822.jsf").write_bytes(b"")
        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda path, survey_name=None, min_pings=50: [(path.stem, sample_multi_frequency_survey)])

        assert convert.run_survey(self._args(tmp_path)) == 0

        out = tmp_path / "ss_h5" / "20241017181822.ss.h5"
        assert out.exists()
        result = read_survey(out)
        assert set(result.freq_channels) == {"hz_1495000", "hz_547500"}

    def test_skips_existing_output_without_force(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.jsf").write_bytes(b"")
        out_dir = tmp_path / "ss_h5"
        out_dir.mkdir()
        (out_dir / "a.ss.h5").write_bytes(b"sentinel")

        called = []
        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda p, **k: called.append(1) or [(p.stem, sample_multi_frequency_survey)])

        assert convert.run_survey(self._args(tmp_path)) == 0
        assert called == []
        assert (out_dir / "a.ss.h5").read_bytes() == b"sentinel"

    def test_force_reprocesses(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.jsf").write_bytes(b"")
        out_dir = tmp_path / "ss_h5"
        out_dir.mkdir()
        (out_dir / "a.ss.h5").write_bytes(b"sentinel")

        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda p, **k: [(p.stem, sample_multi_frequency_survey)])

        assert convert.run_survey(self._args(tmp_path, force=True)) == 0
        assert (out_dir / "a.ss.h5").read_bytes() != b"sentinel"

    def test_no_files_found_returns_1(self, tmp_path):
        (tmp_path / "raw").mkdir()
        assert convert.run_survey(self._args(tmp_path)) == 1

    def test_missing_input_dir_returns_1(self, tmp_path):
        assert convert.run_survey(self._args(tmp_path)) == 1


class TestCalibrateOffsetRequiresUtcOffset:
    def test_missing_local_utc_offset_is_an_argparse_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["sidescan-convert", "targets", str(tmp_path), "-o", str(tmp_path / "o"),
             "--ss-h5-dir", str(tmp_path), "--calibrate-offset"],
        )
        with pytest.raises(SystemExit) as exc:
            convert.main()
        assert exc.value.code == 2
        assert "--local-utc-offset-hours" in capsys.readouterr().err


class TestRunTargets:
    def _args(self, tmp_path, **overrides):
        defaults = dict(
            source=tmp_path / "cat", output=tmp_path / "out",
            ss_h5_dir=None, max_time_delta_s=300.0, time_offset_s=None,
            calibrate_offset=False, local_utc_offset_hours=4.0, plot=True,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_jsf_catalog(self, tmp_path, monkeypatch):
        cat = tmp_path / "cat"
        cat.mkdir()
        (cat / "Target-0001.target").write_bytes(b"")
        targets = [_target("0001", 41.5, -71.4), _target("0002")]
        monkeypatch.setattr(convert, "read_target_catalog", lambda source: ("jsf_discover", targets))

        assert convert.run_targets(self._args(tmp_path)) == 0

        written = read_targets_json(tmp_path / "out" / "targets.json")
        assert len(written) == 2
        assert (tmp_path / "out" / "position_scatter.png").exists()

        written_csv = read_targets_csv(tmp_path / "out" / "targets.csv")
        assert len(written_csv) == 2

    def test_klein_catalog_no_ss_h5_dir_no_position(self, tmp_path, monkeypatch):
        cat = tmp_path / "cat"
        cat.mkdir()
        (cat / "Target_1536923484.tgt").write_bytes(b"")
        targets = [_target("1536923484")]
        monkeypatch.setattr(convert, "read_target_catalog", lambda source: ("sdf_klein", targets))

        assert convert.run_targets(self._args(tmp_path)) == 0

        written = read_targets_json(tmp_path / "out" / "targets.json")
        assert len(written) == 1
        assert not (tmp_path / "out" / "position_scatter.png").exists()

    def test_klein_catalog_with_calibrate_offset(self, tmp_path, monkeypatch):
        cat = tmp_path / "cat"
        cat.mkdir()
        (cat / "Target_1536923484.tgt").write_bytes(b"")
        ss_h5_dir = tmp_path / "ss_h5"
        ss_h5_dir.mkdir()
        raw_targets = [_target("1536923484")]
        resolved_targets = [_target("1536923484", 41.5, -71.4)]

        calls = {}
        monkeypatch.setattr(convert, "read_target_catalog", lambda source: ("sdf_klein", raw_targets))
        monkeypatch.setattr(convert, "calibrate_clock_offset", lambda d, local_utc_offset_hours: calls.setdefault("offset", 500.0) or 500.0)

        def _fake_resolve(targets, d, offset_s, max_time_delta_s):
            calls["offset_s_used"] = offset_s
            return resolved_targets

        monkeypatch.setattr(convert, "resolve_targets_offset_corrected", _fake_resolve)

        args = self._args(tmp_path, ss_h5_dir=ss_h5_dir, calibrate_offset=True)
        assert convert.run_targets(args) == 0
        assert calls["offset_s_used"] == 500.0

        written = read_targets_json(tmp_path / "out" / "targets.json")
        assert written[0].latitude == pytest.approx(41.5)
        assert (tmp_path / "out" / "position_scatter.png").exists()

    def test_klein_catalog_with_ss_h5_dir_no_offset_uses_nearest_ping(self, tmp_path, monkeypatch):
        cat = tmp_path / "cat"
        cat.mkdir()
        (cat / "Target_1536923484.tgt").write_bytes(b"")
        ss_h5_dir = tmp_path / "ss_h5"
        ss_h5_dir.mkdir()
        raw_targets = [_target("1536923484")]
        resolved_targets = [_target("1536923484", 41.5, -71.4)]

        monkeypatch.setattr(convert, "read_target_catalog", lambda source: ("sdf_klein", raw_targets))
        monkeypatch.setattr(convert, "resolve_targets_nearest_ping", lambda targets, d, max_time_delta_s: resolved_targets)

        args = self._args(tmp_path, ss_h5_dir=ss_h5_dir)
        assert convert.run_targets(args) == 0
        written = read_targets_json(tmp_path / "out" / "targets.json")
        assert written[0].latitude == pytest.approx(41.5)

    def test_no_catalog_files_returns_1(self, tmp_path):
        (tmp_path / "cat").mkdir()
        assert convert.run_targets(self._args(tmp_path)) == 1

    def test_scatter_only_with_plot(self, tmp_path, monkeypatch):
        (tmp_path / "cat").mkdir()
        targets = [_target("0001", 41.5, -71.4)]
        monkeypatch.setattr(convert, "read_target_catalog", lambda source: ("jsf_discover", targets))
        assert convert.run_targets(self._args(tmp_path, plot=False)) == 0
        assert (tmp_path / "out" / "targets.json").exists()
        assert not (tmp_path / "out" / "position_scatter.png").exists()


class TestSurveyXtf:
    def test_converts_synthetic_xtf_end_to_end(self, tmp_path):
        from tests.unit.io.xtf.conftest import (
            dual_frequency_infos,
            file_header,
            ramp,
            sonar_packet,
            write_xtf,
        )

        raw = tmp_path / "raw"
        raw.mkdir()
        pk = [sonar_packet([(c, ramp(256, "<u2", i + c)) for c in range(4)], ping_number=i)
              for i in range(20)]
        write_xtf(raw / "LINE1.XTF", file_header(dual_frequency_infos()), pk)
        args = argparse.Namespace(
            input_dir=str(raw), output_dir=str(tmp_path / "ss_h5"),
            input_format="xtf", force=False, crop_farrange=False,
        )
        assert convert.run_survey(args) == 0
        result = read_survey(tmp_path / "ss_h5" / "LINE1.ss.h5")
        assert result.source_format == "xtf"
        assert set(result.freq_channels) == {"hz_100000", "hz_400000"}


    def test_range_change_writes_one_output_per_segment(self, tmp_path):
        from tests.unit.io.xtf.conftest import (
            dual_frequency_infos,
            file_header,
            ramp,
            sonar_packet,
            write_xtf,
        )

        raw = tmp_path / "raw"
        raw.mkdir()
        pk = [
            sonar_packet([(c, ramp(256, "<u2", i + c)) for c in range(4)], ping_number=i,
                         chan_fields={"SlantRange": r, "TimeDuration": r / 750.0})
            for i, r in enumerate([100.0] * 12 + [150.0] * 2 + [200.0] * 10)
        ]
        write_xtf(raw / "LINE1.XTF", file_header(dual_frequency_infos()), pk)
        out = tmp_path / "ss_h5"
        args = ["survey", "--input-dir", str(raw), "--output-dir", str(out),
                "--no-crop-farrange", "--min-segment-pings", "5"]
        assert convert.main(args) == 0
        assert sorted(p.name for p in out.iterdir()) == ["LINE1_seg01.ss.h5", "LINE1_seg02.ss.h5"]
        seg2 = read_survey(out / "LINE1_seg02.ss.h5")
        assert seg2.survey_name == "LINE1_seg02"
        ds = seg2.get_frequency("hz_100000").dataset
        assert ds.sizes["ping"] == 10
        assert ds.attrs["sample_rate_hz"] == pytest.approx(256 * 750.0 / 200.0)

        # Existing segment outputs are skipped one by one; the missing one is rewritten.
        (out / "LINE1_seg01.ss.h5").unlink()
        mtime = (out / "LINE1_seg02.ss.h5").stat().st_mtime_ns
        assert convert.main(args) == 0
        assert (out / "LINE1_seg01.ss.h5").exists()
        assert (out / "LINE1_seg02.ss.h5").stat().st_mtime_ns == mtime


class TestMainDispatch:
    def test_survey_subcommand_via_argv(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.jsf").write_bytes(b"")
        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda p, **k: [(p.stem, sample_multi_frequency_survey)])
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-convert", "survey",
                "--input-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "ss_h5"),
            ],
        )
        assert convert.main() == 0
        assert (tmp_path / "ss_h5" / "a.ss.h5").exists()

    def test_verbose_accepted_after_subcommand(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        """Real bug found running scripts against real data: -v is only
        ever useful after the subcommand in practice (argparse requires
        top-level flags before the subcommand name), but the top-level
        parser and each subparser both defining -v with the same dest
        risks the subparser's default silently overwriting a real -v
        set before the subcommand -- see the SUPPRESS comment at the
        definition site. This covers the actually-common case (-v
        after the subcommand); test_verbose_before_subcommand_not_clobbered
        below covers the overwrite risk directly."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.jsf").write_bytes(b"")
        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda p, **k: [(p.stem, sample_multi_frequency_survey)])
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-convert", "survey",
                "--input-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "ss_h5"),
                "-v",
            ],
        )
        assert convert.main() == 0
        assert (tmp_path / "ss_h5" / "a.ss.h5").exists()

    def test_verbose_before_subcommand_not_clobbered(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        """-v BEFORE the subcommand must survive the subparser's own
        (unset) default -- this is the exact regression the SUPPRESS
        default guards against."""
        captured = {}
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.jsf").write_bytes(b"")
        monkeypatch.setattr(convert, "detect_format", lambda paths: "jsf")
        monkeypatch.setattr(convert, "read_and_convert_segments", lambda p, **k: [(p.stem, sample_multi_frequency_survey)])

        def _fake_run_survey(args):
            captured["verbose"] = args.verbose
            return 0

        monkeypatch.setattr(convert, "run_survey", _fake_run_survey)
        monkeypatch.setattr(
            "sys.argv",
            [
                "sidescan-convert", "-v", "survey",
                "--input-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "ss_h5"),
            ],
        )
        assert convert.main() == 0
        assert captured["verbose"] is True
