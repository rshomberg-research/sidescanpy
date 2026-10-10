"""
Unit tests for sidescanpy.cli._config — shared --config YAML/JSON
defaults support for the sidescan-* CLIs.

2026 rshomberg@uri.edu
"""

import json

import pytest

from sidescanpy.cli._common import add_common_args, setup_logging_from_args
from sidescanpy.cli._config import CLIParser, load_config, parse_args


def _single(tmp_path=None):
    p = CLIParser(prog="tool")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--channel", default="both")
    add_common_args(p, subcommand=False)
    return p


def _multi():
    top = CLIParser(prog="tool")
    sub = top.add_subparsers(dest="subcommand", required=True)
    a = sub.add_parser("cmd_a", aliases=["cmd-a-alias"])
    a.add_argument("--gamma", type=float, default=1.0)
    group = a.add_mutually_exclusive_group()
    group.add_argument("--grouped", type=int, default=0)
    b = sub.add_parser("cmd_b")
    b.add_argument("--other", type=int, default=0)
    for p in (a, b):
        add_common_args(p)
    return top, sub.choices


def _write(tmp_path, data, name="cfg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data) if name.endswith(".json") else data)
    return str(path)


class TestCLIParser:
    def test_records_option_dests_including_groups(self):
        top, subs = _multi()
        assert {"gamma", "grouped", "config", "strict_config", "verbose"} <= subs["cmd_a"].option_dests
        assert "other" in subs["cmd_b"].option_dests
        assert subs["cmd-a-alias"] is subs["cmd_a"]


class TestParseArgsSingle:
    def test_no_config_is_a_plain_parse(self):
        args = parse_args(_single(), ["--gamma", "2"])
        assert args.gamma == 2.0 and args.config is None

    def test_seeds_recognized_keys(self, tmp_path):
        args = parse_args(_single(), ["--config", _write(tmp_path, {"gamma": 2.5, "channel": "port"})])
        assert args.gamma == 2.5
        assert args.channel == "port"

    def test_explicit_flag_overrides_config(self, tmp_path):
        args = parse_args(_single(), ["--config", _write(tmp_path, {"gamma": 2.5}), "--gamma", "3.5"])
        assert args.gamma == 3.5

    def test_unknown_key_warns_and_is_ignored(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="sidescanpy.cli._config"):
            args = parse_args(_single(), ["--config", _write(tmp_path, {"gamma": 2.5, "not_a_flag": 1})])
            setup_logging_from_args(args)
        assert args.gamma == 2.5
        assert not hasattr(args, "not_a_flag")
        assert "not_a_flag" in caplog.text and "'tool'" in caplog.text

    def test_strict_config_errors(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            parse_args(_single(), ["--config", _write(tmp_path, {"typo": 1}), "--strict-config"])
        assert "typo" in capsys.readouterr().err

    def test_yaml_config(self, tmp_path):
        args = parse_args(_single(), ["--config", _write(tmp_path, "gamma: 4.0\n", "cfg.yaml")])
        assert args.gamma == 4.0

    def test_config_key_cannot_set_config_itself(self, tmp_path, caplog):
        path = _write(tmp_path, {"config": "elsewhere.yaml"})
        with caplog.at_level("WARNING", logger="sidescanpy.cli._config"):
            args = parse_args(_single(), ["--config", path])
            setup_logging_from_args(args)
        assert str(args.config) == path
        assert "config" in caplog.text


class TestParseArgsSubcommands:
    def test_seeds_chosen_subcommand(self, tmp_path):
        top, subs = _multi()
        args = parse_args(top, ["cmd_a", "--config", _write(tmp_path, {"gamma": 2.0, "grouped": 3})], subcommands=subs)
        assert args.gamma == 2.0 and args.grouped == 3

    def test_alias_name_resolves(self, tmp_path):
        top, subs = _multi()
        args = parse_args(top, ["cmd-a-alias", "--config", _write(tmp_path, {"gamma": 2.0})], subcommands=subs)
        assert args.gamma == 2.0

    def test_key_valid_for_other_subcommand_is_not_flagged(self, tmp_path, caplog):
        top, subs = _multi()
        with caplog.at_level("WARNING", logger="sidescanpy.cli._config"):
            args = parse_args(top, ["cmd_a", "--config", _write(tmp_path, {"gamma": 2.0, "other": 3})], subcommands=subs)
            setup_logging_from_args(args)
        assert "other" not in caplog.text
        assert not hasattr(args, "other")

    def test_unknown_key_names_subcommand(self, tmp_path, caplog):
        top, subs = _multi()
        with caplog.at_level("WARNING", logger="sidescanpy.cli._config"):
            args = parse_args(top, ["cmd_a", "--config", _write(tmp_path, {"typo_key": 1})], subcommands=subs)
            setup_logging_from_args(args)
        assert "typo_key" in caplog.text and "cmd_a" in caplog.text

    def test_strict_config_errors_naming_key(self, tmp_path, capsys):
        top, subs = _multi()
        with pytest.raises(SystemExit):
            parse_args(top, ["cmd_a", "--strict-config", "--config", _write(tmp_path, {"typo_key": 1})], subcommands=subs)
        assert "typo_key" in capsys.readouterr().err


class TestLoadConfig:
    def test_loads_json_by_extension(self, tmp_path):
        config_path = tmp_path / "survey.json"
        config_path.write_text(json.dumps({"gamma": 2.5, "channel": "port"}))
        assert load_config(config_path) == {"gamma": 2.5, "channel": "port"}

    def test_loads_yaml_by_extension(self, tmp_path):
        config_path = tmp_path / "survey.yaml"
        config_path.write_text("gamma: 2.5\nchannel: port\n")
        assert load_config(config_path) == {"gamma": 2.5, "channel": "port"}

    def test_loads_yml_extension(self, tmp_path):
        config_path = tmp_path / "survey.yml"
        config_path.write_text("gamma: 2.5\n")
        assert load_config(config_path) == {"gamma": 2.5}

    def test_empty_yaml_is_empty_config(self, tmp_path):
        config_path = tmp_path / "survey.yaml"
        config_path.write_text("")
        assert load_config(config_path) == {}

    def test_unsuffixed_falls_back_to_json(self, tmp_path):
        config_path = tmp_path / "survey_config"
        config_path.write_text(json.dumps({"gamma": 2.5}))
        assert load_config(config_path) == {"gamma": 2.5}

