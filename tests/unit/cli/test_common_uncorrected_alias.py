"""Deprecated spellings of the uncorrected-signal choice map to 'uncorrected'."""

import argparse
import logging

from sidescanpy.cli._common import uncorrected_alias


def _parser(*old):
    p = argparse.ArgumentParser()
    p.add_argument("--value-var", type=uncorrected_alias(*old), default="uncorrected")
    return p


def test_old_spelling_maps_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="sidescanpy.cli._common"):
        args = _parser("raw", "backscatter_raw").parse_args(["--value-var", "raw"])
    assert args.value_var == "uncorrected"
    assert "'raw' is deprecated; use 'uncorrected'" in caplog.text


def test_new_spelling_and_other_values_untouched(caplog):
    with caplog.at_level(logging.WARNING, logger="sidescanpy.cli._common"):
        assert _parser("raw").parse_args(["--value-var", "uncorrected"]).value_var == "uncorrected"
        assert _parser("raw").parse_args(["--value-var", "db"]).value_var == "db"
    assert not caplog.records
