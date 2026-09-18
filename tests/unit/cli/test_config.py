"""
Unit tests for sidescanpy.cli._config — shared --config YAML/JSON
defaults support for the sidescan-* CLIs.

2026 rshomberg@uri.edu
"""

import argparse
import json

from sidescanpy.cli._config import apply_config_defaults, find_config_path, load_config


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
        config = load_config(config_path)
        apply_config_defaults(p, config)
        args = p.parse_args([])
        assert args.gamma == 2.5


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

    def test_unsuffixed_falls_back_to_json(self, tmp_path):
        """Matches every config file already in use before YAML support
        existed -- an extensionless/oddly-suffixed file keeps parsing as
        JSON exactly as it did before this function existed."""
        config_path = tmp_path / "survey_config"
        config_path.write_text(json.dumps({"gamma": 2.5}))
        assert load_config(config_path) == {"gamma": 2.5}

    def test_end_to_end_yaml_seeds_argparse_defaults(self, tmp_path):
        config_path = tmp_path / "survey.yaml"
        config_path.write_text("spreading_loss_db: 20.0\n")
        p = argparse.ArgumentParser()
        p.add_argument("--spreading-loss-db", dest="spreading_loss_db", type=float, default=0.0)

        apply_config_defaults(p, load_config(config_path))
        args = p.parse_args([])
        assert args.spreading_loss_db == 20.0
