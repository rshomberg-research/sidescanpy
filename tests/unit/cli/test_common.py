"""
Unit tests for sidescanpy.cli._common.resolve_freq_keys.

2026 rshomberg@uri.edu
"""

from sidescanpy.cli._common import resolve_freq_keys


class TestResolveFreqKeys:
    def test_all_returns_every_key_sorted(self):
        assert resolve_freq_keys({"hz_547500", "hz_1495000"}, "all") == ["hz_1495000", "hz_547500"]

    def test_bare_number_resolves_to_hz_prefixed_key(self):
        assert resolve_freq_keys(["hz_1495000", "hz_547500"], "1495000") == ["hz_1495000"]

    def test_falls_back_to_literal_freq_key_for_non_hz_naming(self):
        """Not every consumer uses JSF/SDF's "hz_<freq>" convention (e.g.
        vendor-format's "sensor1_data1") -- a request that doesn't match any
        "hz_"-prefixed key should resolve to itself, not silently produce
        a freq_key that doesn't exist anywhere."""
        assert resolve_freq_keys(["sensor1_data1", "sensor1_data2"], "sensor1_data1") == ["sensor1_data1"]

    def test_unresolvable_request_still_returns_something_for_caller_to_report(self):
        """Neither "hz_<value>" nor the literal value exists -- returns
        the literal value anyway, matching prior behavior of callers that
        detect a missing frequency themselves (e.g. "no file has this
        frequency, skipping") rather than raising here."""
        assert resolve_freq_keys(["hz_1495000"], "999999") == ["999999"]
