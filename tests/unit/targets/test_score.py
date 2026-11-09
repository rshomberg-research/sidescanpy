"""
Tests for sidescanpy.targets.score: candidates matched to a catalog by
geodesic distance, with precision and recall.

2026 rshomberg@uri.edu
"""

import math

import pytest

from sidescanpy.core.target import Target
from sidescanpy.targets.score import match_targets

LAT0, LON0 = 41.5, -71.4
M_PER_DEG_LAT = 111_045.0  # near 41.5 N


def _t(tid, north_m=0.0, *, score=None, positioned=True):
    lat = LAT0 + north_m / M_PER_DEG_LAT if positioned else None
    lon = LON0 if positioned else None
    extra = {} if score is None else {"score": score}
    return Target(tid, "test", "", latitude=lat, longitude=lon, extra_attributes=extra)


class TestMatchTargets:
    def test_all_hit_when_candidates_on_catalog(self):
        cat = [_t("a", 0), _t("b", 100)]
        res = match_targets([_t("x", 1), _t("y", 99)], cat, max_distance_m=5)
        assert sorted((c, k) for c, k, _ in res.matches) == [(0, 0), (1, 1)]
        assert res.precision == 1.0 and res.recall == 1.0
        assert res.matches[0][2] == pytest.approx(1.0, abs=0.05)

    def test_false_alarm_and_miss_when_out_of_range(self):
        res = match_targets([_t("x", 0), _t("y", 50)], [_t("a", 0), _t("b", 200)], max_distance_m=10)
        assert [(c, k) for c, k, _ in res.matches] == [(0, 0)]
        assert res.false_alarms == [1] and res.misses == [1]
        assert res.precision == 0.5 and res.recall == 0.5

    def test_each_catalog_target_used_once_closest_first(self):
        # Two candidates near one catalog target: the closer one wins.
        res = match_targets([_t("far", 4), _t("near", 1)], [_t("a", 0)], max_distance_m=10)
        assert [(c, k) for c, k, _ in res.matches] == [(1, 0)]
        assert res.false_alarms == [0]

    def test_min_score_leaves_out_low_candidates(self):
        res = match_targets([_t("x", 0, score=0.9), _t("y", 50, score=0.1)], [_t("a", 0)],
                            max_distance_m=10, min_score=0.5)
        assert res.skipped_candidates == [1]
        assert res.false_alarms == [] and res.precision == 1.0

    def test_unpositioned_entries_skipped(self):
        res = match_targets([_t("x", positioned=False)], [_t("a"), _t("b", positioned=False)],
                            max_distance_m=10)
        assert res.skipped_candidates == [0] and res.skipped_catalog == [1]
        assert res.misses == [0]
        assert math.isnan(res.precision) and res.recall == 0.0

    def test_summary_mentions_counts(self):
        res = match_targets([_t("x", 0)], [_t("a", 0)], max_distance_m=5)
        assert res.summary().startswith("1 hit(s), 0 false alarm(s), 0 miss(es) within 5 m")
