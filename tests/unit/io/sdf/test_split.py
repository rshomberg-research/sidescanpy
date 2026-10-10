"""
Unit tests for sidescanpy.io.sdf.split: cutting one file's pages at
range-setting (sample-rate) changes, on synthetic dual-frequency pages.

2026 rshomberg@uri.edu
"""

import pytest

from sidescanpy.io.sdf.split import sample_rate_runs, split_by_sample_rate
from tests.unit.io.sdf.conftest import make_sdf_sonar_page


def _pages(settings):
    """One 455 kHz and one 900 kHz page per ping; ``settings`` is [(range_m, n_hf, n_lf), ...]."""
    pages = []
    for i, (rng, n_hf, n_lf) in enumerate(settings):
        t = 1746136205.0 + i * 0.1
        pages.append(make_sdf_sonar_page(page_version=3502, center_freq_hz=900_000, ping_number=i,
                                         num_samples=n_hf, range_m=rng, timestamp=t))
        pages.append(make_sdf_sonar_page(page_version=3501, center_freq_hz=455_000, ping_number=i,
                                         num_samples=n_lf, range_m=rng, timestamp=t))
    return pages


class TestSampleRateRuns:
    def test_constant_setting_is_one_run(self):
        pages = _pages([(60, 100, 100)] * 5)
        assert sample_rate_runs(pages) == [(0, 10)]

    def test_range_change_with_rate_change_cuts(self):
        # 60 m / 100 samples -> 50 m / 166 samples (the high frequency doubles its rate)
        pages = _pages([(60, 100, 100)] * 4 + [(50, 166, 83)] * 3)
        assert sample_rate_runs(pages) == [(0, 8), (8, 14)]

    def test_range_change_at_constant_rate_does_not_cut(self):
        # Samples scale with range: same metres per sample, no cut needed.
        pages = _pages([(60, 120, 120)] * 3 + [(50, 100, 100)] * 3)
        assert sample_rate_runs(pages) == [(0, 12)]

    def test_empty(self):
        assert sample_rate_runs([]) == []


class TestSplitBySampleRate:
    def test_short_middle_run_dropped(self):
        pages = _pages([(60, 100, 100)] * 6 + [(50, 166, 83)] * 2 + [(40, 200, 100)] * 5)
        runs = split_by_sample_rate(pages, min_pings=4)
        assert [len(r) for r in runs] == [12, 10]
        assert {p.range_m for p in runs[1]} == {40}

    def test_single_run_kept_whatever_its_length(self):
        pages = _pages([(60, 100, 100)] * 2)
        assert [len(r) for r in split_by_sample_rate(pages, min_pings=50)] == [4]

    def test_all_runs_short_returns_empty(self, caplog):
        pages = _pages([(60, 100, 100)] * 2 + [(50, 166, 83)] * 2)
        assert split_by_sample_rate(pages, min_pings=10) == []
        assert "shorter than" in caplog.text


class TestDispatchSdfSegments:
    def test_sdf_range_change_split_into_segments(self, monkeypatch, tmp_path):
        from sidescanpy.io.dispatch import read_and_convert_segments

        pages = _pages([(60, 100, 100)] * 6 + [(40, 133, 67)] * 6)
        monkeypatch.setattr("sidescanpy.io.dispatch.read_sdf", lambda p: pages)
        path = tmp_path / "sonar_data180914080900.sdf"
        out = read_and_convert_segments(path, min_pings=4)
        assert [stem for stem, _ in out] == ["sonar_data180914080900_seg01", "sonar_data180914080900_seg02"]
        hf = [s.get_frequency("hz_900000").dataset for _, s in out]
        assert [d.sizes["ping"] for d in hf] == [6, 6]
        assert hf[0].attrs["sample_rate_hz"] == pytest.approx(100 * 1500 / (2 * 60))
        assert hf[1].attrs["sample_rate_hz"] == pytest.approx(133 * 1500 / (2 * 40))

    def test_sdf_constant_setting_single_survey_named_after_stem(self, monkeypatch, tmp_path):
        from sidescanpy.io.dispatch import read_and_convert_segments

        pages = _pages([(60, 100, 100)] * 3)
        monkeypatch.setattr("sidescanpy.io.dispatch.read_sdf", lambda p: pages)
        ((stem, survey),) = read_and_convert_segments(tmp_path / "x.sdf")
        assert stem == "x"
        assert survey.survey_name == "x"
