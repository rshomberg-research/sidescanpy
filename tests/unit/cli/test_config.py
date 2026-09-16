"""
Unit tests for sidescanpy.cli._config — shared --config JSON-defaults
support for the sidescan-* CLIs.

2026 rshomberg@uri.edu
"""

import argparse
import json

from sidescanpy.cli._config import apply_config_defaults, find_config_path


class TestFindConfigPath:
    def test_space_separated(self):
        assert find_config_path(["chip", "--config", "x.json", "--gamma", "2.0"]) == __import__("pathlib").Path("x.json")

    def test_equals_form(self):
        from pathlib import Path
        assert find_config_path(["chip", "--config=x.json"]) == Path("x.json")

    def test_absent_returns_none(self):
        assert find_config_path(["chip", "--gamma", "2.0"]) is None

    def test_trailing_flag_with_no_value_returns_none(self):
        assert find_config_path(["chip", "--config"]) is None


class TestApplyConfigDefaults:
    def _parser(self):
        p = argparse.ArgumentParser()
        p.add_argument("--gamma", type=float, default=1.0)
        p.add_argument("--channel", default="both")
        return p

    def test_seeds_recognized_keys(self):
        p = self._parser()
        apply_config_defaults(p, {"gamma": 2.5, "channel": "port"})
        args = p.parse_args([])
        assert args.gamma == 2.5
        assert args.channel == "port"

    def test_explicit_flag_overrides_config(self):
        p = self._parser()
        apply_config_defaults(p, {"gamma": 2.5})
        args = p.parse_args(["--gamma", "3.5"])
        assert args.gamma == 3.5

    def test_unrecognized_keys_ignored_not_error(self):
        p = self._parser()
        apply_config_defaults(p, {"gamma": 2.5, "not_a_real_flag": 123})
        args = p.parse_args([])
        assert args.gamma == 2.5
        assert not hasattr(args, "not_a_real_flag")

    def test_end_to_end_config_file(self, tmp_path):
        config_path = tmp_path / "survey.json"
        config_path.write_text(json.dumps({"gamma": 2.5}))

        p = self._parser()
        config = json.loads(config_path.read_text())
        apply_config_defaults(p, config)
        args = p.parse_args([])
        assert args.gamma == 2.5
