"""
CLI wiring: every subcommand's ``--help``, flag -> function-argument
wiring for the flags added with the thin-CLI split, and exit codes.
Library behavior is tested with the library.

2026 rshomberg@uri.edu
"""

import pytest

from sidescanpy.cli import analyze, convert, correct, export

_SUBCOMMANDS = {
    convert: ["survey", "targets"],
    export: ["waterfall", "grid", "track", "chip"],
    analyze: [
        "investigate", "tvg-fit", "segment", "nadir", "beam-geometry", "usable-range", "units",
        "nav-attitude", "nav-summary", "recommend-config",
    ],
}


def _all_help_argvs():
    for module, subs in _SUBCOMMANDS.items():
        yield module, ["--help"]
        for sub in subs:
            yield module, [sub, "--help"]
    yield correct, ["--help"]


@pytest.mark.parametrize("module, argv", list(_all_help_argvs()), ids=lambda v: " ".join(v) if isinstance(v, list) else v.__name__)
def test_help_exits_zero(module, argv, capsys):
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    if argv != ["--help"] or module is correct:
        assert "--config" in out and "--log-file" in out


def test_subcommand_names_match_parsers():
    for module, subs in _SUBCOMMANDS.items():
        _parser, subparsers = module.build_parser()
        assert set(subparsers.choices) == set(subs)


def test_missing_subcommand_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        analyze.main([])
    assert exc.value.code == 2


class TestConvertWiring:
    def _parse(self, argv):
        parser, _ = convert.build_parser()
        return parser.parse_args(argv)

    def test_crop_farrange_default_on_and_negatable(self):
        base = ["survey", "--input-dir", "i", "--output-dir", "o"]
        assert self._parse(base).crop_farrange is True
        assert self._parse([*base, "--no-crop-farrange"]).crop_farrange is False

    def test_no_crop_skips_crop(self, tmp_path, monkeypatch, sample_multi_frequency_survey):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.sdf").write_bytes(b"")
        monkeypatch.setattr(convert, "read_and_convert", lambda *a, **k: sample_multi_frequency_survey)
        monkeypatch.setattr(convert, "crop_far_range_noise", lambda s: pytest.fail("crop must not run"))
        rc = convert.main(["survey", "--input-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "out"),
                           "--no-crop-farrange"])
        assert rc == 0
        assert (tmp_path / "out" / "a.ss.h5").exists()

    def test_plot_defaults_off(self):
        assert self._parse(["targets", "--source", "s", "--output", "o"]).plot is False


class TestExportWiring:
    def test_grid_geotiff_flag(self):
        parser, _ = export.build_parser()
        base = ["grid", "--ss-h5-dir", "d", "--output-dir", "o"]
        assert parser.parse_args(base).geotiff is False
        assert parser.parse_args([*base, "--geotiff"]).geotiff is True


class TestAnalyzeWiring:
    def _dispatch(self, monkeypatch, argv, name):
        seen = []
        _parser, subparsers = analyze.build_parser()
        for sub in subparsers.choices.values():
            if sub.get_default("func").__name__ == name:
                sub.set_defaults(func=lambda args: seen.append(args) or 0)
        monkeypatch.setattr(analyze, "build_parser", lambda: (_parser, subparsers))
        assert analyze.main(argv) == 0
        return seen[0]

    def test_nav_summary_alias_runs_nav_attitude(self, monkeypatch):
        args = self._dispatch(monkeypatch, ["nav-summary", "--input", "x.ss.h5", "--output-dir", "o"], "run_nav_attitude")
        assert args.subcommand == "nav-summary" and args.heading_var == "heading_deg"

    @pytest.mark.parametrize("sub, extra", [
        ("units", ["--frequency", "1"]), ("nadir", ["--frequency", "1"]), ("segment", []),
        ("tvg-fit", None), ("investigate", []),
    ])
    def test_plots_default_on_and_negatable(self, sub, extra):
        parser, _ = analyze.build_parser()
        if extra is None:
            base = [sub, "--ss-h5", "x", "--frequency", "1", "--output-dir", "o"]
        else:
            base = [sub, "--input", "x", "--output-dir", "o", *extra]
        assert parser.parse_args(base).plots is True
        assert parser.parse_args([*base, "--no-plots"]).plots is False

    def test_no_plots_writes_metrics_only(self, tmp_path, ss_h5_for_units):
        out = tmp_path / "out"
        rc = analyze.main(["units", "--input", str(ss_h5_for_units), "--frequency", "1495000",
                           "--output-dir", str(out), "--no-plots"])
        assert rc == 0
        assert [p.name for p in out.iterdir()] == ["metrics.json"]


@pytest.fixture
def ss_h5_for_units(tmp_path, sample_multi_frequency_survey):
    from sidescanpy.io.hdf5 import write_survey

    path = tmp_path / "a.ss.h5"
    write_survey(sample_multi_frequency_survey, path)
    return path
