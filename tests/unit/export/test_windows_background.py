"""
Unit tests for sidescanpy.export.windows.background.

The fixture survey carries a huge marker value in the backscatter at every
cataloged target's real position, so "the window contains a target" is
checked independently of the exclusion logic under test: a rendered window
that covered a target would show the marker.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pytest

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.core.target import Target
from sidescanpy.export.windows import (
    BackgroundCandidate,
    WindowSpec,
    nan_fraction,
    sample_background_candidates,
    sample_background_windows,
)
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing import correct_survey
from sidescanpy.targets.correlate import build_position_index, place_catalog
from tests.conftest import make_minimal_sidescan_dataset

N_PINGS, N_SAMPLES = 600, 100
MARKER = 1e6
LAT = np.linspace(40.0, 40.0005, N_PINGS)
LON = np.linspace(-70.0, -70.0005, N_PINGS)
# (ping, channel, sample, confidence) -- the "unknown" one must still be avoided.
TARGETS = [(150, 1, 60, "verified"), (450, 0, 40, "verified"), (560, 1, 30, "unknown")]


@pytest.fixture(scope="module")
def marker_survey(tmp_path_factory):
    d = tmp_path_factory.mktemp("bg") / "ss_h5"
    d.mkdir()
    rng = np.random.default_rng(0)
    ds = make_minimal_sidescan_dataset(n_pings=N_PINGS, n_samples=N_SAMPLES, freq_hz_start=1_495_000)
    ds.attrs["sample_rate_hz"] = 40_000.0  # 1.875 cm/sample: a ~1.9 m swath per side
    ds["backscatter_intensity"].values[:] = rng.uniform(5.0, 10.0, ds["backscatter_intensity"].shape)
    for p, ch, s, _conf in TARGETS:
        ds["backscatter_intensity"].values[p - 1:p + 2, ch, s - 2:s + 3] = MARKER
    ds["altitude_m"] = ("ping", np.full(N_PINGS, 0.1, dtype=np.float32))
    ds["latitude"] = ("ping", LAT)
    ds["longitude"] = ("ping", LON)
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
    survey.add_frequency("hz_1495000", ds)
    write_survey(correct_survey(survey), d / "a.ss.h5")
    catalog = [
        Target(
            target_id=str(i), source_format="x", catalog_source="x", latitude=float(LAT[p]), longitude=float(LON[p]),
            channel=ch, sample=s, position_confidence=conf,
        )
        for i, (p, ch, s, conf) in enumerate(TARGETS)
    ]
    return build_position_index(d), catalog


def _shows_marker(image):
    finite = image[np.isfinite(image)]
    return finite.size > 0 and finite.max() > MARKER / 100


CASES = [
    ("radiometric", "chip", "ground", 16),
    ("radiometric", "single-swath", "ground", 32),
    ("radiometric", "dual-swath", "ground", 32),
    ("geometric", "chip", "ground", 16),
    ("geometric", "single-swath", "ground", 32),
    ("geometric", "dual-swath", "ground", 32),
    ("geometric", "chip", "slant", 16),
    ("geometric", "single-swath", "slant", 32),
    ("geometric", "dual-swath", "slant", 32),
    ("georef", "chip", "ground", 16),
    ("georef", "single-swath", "ground", 32),
    ("georef", "dual-swath", "ground", 32),
]


class TestSampleBackgroundWindows:
    @pytest.mark.parametrize("level, style, across_axis, size_px", CASES)
    def test_no_window_contains_a_target(self, marker_survey, level, style, across_axis, size_px):
        index, catalog = marker_survey
        spec = WindowSpec(style=style, level=level, size_px=size_px, jitter_px=2, across_axis=across_axis)
        windows = sample_background_windows(index, catalog, spec, 3, np.random.default_rng(5), max_nan_frac=1.0)
        assert len({w.index for w in windows}) == 3
        for w in windows:
            assert w.window.targets == []
            assert not _shows_marker(w.window.image), (w.candidate, w.footprint)
            assert w.footprint.frame == ("pixel" if level == "radiometric" else level)
            assert w.style == style

    @pytest.mark.parametrize("style", ["chip", "single-swath", "dual-swath"])
    @pytest.mark.parametrize("level", ["radiometric", "geometric", "georef"])
    def test_a_window_on_a_target_is_rejected(self, marker_survey, caplog, level, style):
        """Non-vacuous check: centered on a target, the window is rejected by
        the frame check (not only the geographic guard)."""
        index, catalog = marker_survey
        (path,) = index
        spec = WindowSpec(style=style, level=level, size_px=16)
        caplog.set_level(logging.DEBUG, logger="sidescanpy.export.windows.background")
        on_target = [BackgroundCandidate(path, "hz_1495000", 150, 1, 60)]
        windows = sample_background_windows(index, catalog, spec, 1, np.random.default_rng(0), candidates=on_target, max_nan_frac=1.0)
        assert windows == []
        assert "'target': 1" in caplog.text

    def test_supplied_candidates_are_screened_in_order(self, marker_survey):
        index, catalog = marker_survey
        (path,) = index
        spec = WindowSpec(style="chip", level="geometric", size_px=16)
        cands = [BackgroundCandidate(path, "hz_1495000", 450, 0, 40), BackgroundCandidate(path, "hz_1495000", 300, 1, 50)]
        windows = sample_background_windows(index, catalog, spec, 2, np.random.default_rng(0), candidates=cands, max_nan_frac=1.0)
        assert [w.candidate for w in windows] == cands[1:]

    def test_geographic_guard_covers_targets_without_a_frame_position(self, marker_survey, caplog):
        """A center-only pick (no sample) can't be placed in a window frame;
        its recorded lat/lon still excludes nearby windows."""
        index, catalog = marker_survey
        (path,) = index
        unplaceable = Target(
            target_id="x", source_format="x", catalog_source="x", latitude=float(LAT[300]), longitude=float(LON[300]),
        )
        spec = WindowSpec(style="chip", level="geometric", size_px=16)
        cand = [BackgroundCandidate(path, "hz_1495000", 300, 1, 50)]
        caplog.set_level(logging.DEBUG, logger="sidescanpy.export.windows.background")
        assert sample_background_windows(index, catalog, spec, 1, np.random.default_rng(0), candidates=cand, max_nan_frac=1.0)
        assert sample_background_windows(
            index, [*catalog, unplaceable], spec, 1, np.random.default_rng(0), candidates=cand, max_nan_frac=1.0,
        ) == []
        assert "'geographic_guard': 1" in caplog.text

    def test_low_confidence_targets_are_avoided(self, marker_survey):
        index, catalog = marker_survey
        spec = WindowSpec(style="dual-swath", level="geometric", size_px=32)
        windows = sample_background_windows(
            index, catalog, spec, 1, np.random.default_rng(0), ping_ranges={"a.ss.h5": [(555, 566)]},
            max_nan_frac=1.0, max_attempts=5,
        )
        assert windows == []

    def test_margin_rejects_nearby_windows(self, marker_survey):
        index, catalog = marker_survey
        spec = WindowSpec(style="chip", level="geometric", size_px=16)
        ranges = {"a.ss.h5": [(250, 350)]}  # 100+ pings (>11 m) from every target
        near = sample_background_windows(index, catalog, spec, 2, np.random.default_rng(1), ping_ranges=ranges, max_nan_frac=1.0)
        assert len({w.index for w in near}) == 2
        far = sample_background_windows(
            index, catalog, spec, 2, np.random.default_rng(1), ping_ranges=ranges, max_nan_frac=1.0, margin_m=50.0,
            max_attempts=10,
        )
        assert far == []

    def test_nan_fraction_rejection(self, marker_survey):
        """Binned resampling leaves the masked nadir gap NaN in a dual-swath window."""
        index, catalog = marker_survey
        spec = WindowSpec(style="dual-swath", level="geometric", size_px=32, resample_method="binned")
        ranges = {"a.ss.h5": [(280, 320)]}
        loose = sample_background_windows(index, catalog, spec, 2, np.random.default_rng(2), ping_ranges=ranges, max_nan_frac=1.0)
        assert loose and all(w.nan_fraction > 0 for w in loose)
        strict = sample_background_windows(
            index, catalog, spec, 2, np.random.default_rng(2), ping_ranges=ranges, max_nan_frac=0.0, max_attempts=10,
        )
        assert strict == []
        for w in loose:
            assert w.nan_fraction == pytest.approx(nan_fraction(w.window.image))

    def test_draws_share_a_center_and_are_numbered(self, marker_survey):
        index, catalog = marker_survey
        spec = WindowSpec(style="chip", level="geometric", size_px=16, jitter_px=3)
        windows = sample_background_windows(
            index, catalog, spec, 2, np.random.default_rng(3), ping_ranges={"a.ss.h5": [(250, 350)]},
            max_nan_frac=1.0, n_draws=3,
        )
        by_index = {}
        for w in windows:
            by_index.setdefault(w.index, []).append(w)
        assert sorted(by_index) == [0, 1]
        for draws in by_index.values():
            assert [w.draw for w in draws] == list(range(len(draws)))
            assert len({w.candidate for w in draws}) == 1

    def test_seeded_runs_are_reproducible(self, marker_survey):
        index, catalog = marker_survey
        spec = WindowSpec(style="chip", level="radiometric", size_px=16, jitter_px=2)
        a = sample_background_windows(index, catalog, spec, 2, np.random.default_rng(9), max_nan_frac=1.0)
        b = sample_background_windows(index, catalog, spec, 2, np.random.default_rng(9), max_nan_frac=1.0)
        assert [w.candidate for w in a] == [w.candidate for w in b]
        for wa, wb in zip(a, b, strict=True):
            np.testing.assert_array_equal(wa.window.image, wb.window.image)

    def test_zero_requested_returns_nothing(self, marker_survey):
        index, catalog = marker_survey
        assert sample_background_windows(index, catalog, WindowSpec("chip", "geometric", 16), 0, np.random.default_rng(0)) == []


class TestCandidates:
    def _index(self):
        return {
            "/x/a.ss.h5": {"hz_1": {"n_pings": 100, "n_samples": 10}, "hz_2": {"n_pings": 100, "n_samples": 10}},
            "/x/b.ss.h5": {"hz_1": {"n_pings": 50, "n_samples": 10}},
        }

    def test_uniform_by_ping_over_files_and_frequencies(self):
        cands = sample_background_candidates(self._index(), 5000, np.random.default_rng(0))
        assert all(isinstance(c, BackgroundCandidate) for c in cands)
        share_b = np.mean([c.path.name == "b.ss.h5" for c in cands])
        assert share_b == pytest.approx(50 / 250, abs=0.03)
        assert {c.channel for c in cands} == {0, 1}
        assert all(0 <= c.sample < 10 for c in cands)

    def test_freq_keys_and_ping_ranges_restrict(self):
        cands = sample_background_candidates(
            self._index(), 500, np.random.default_rng(0), freq_keys=["hz_1"], ping_ranges={"a.ss.h5": [(10, 20), (90, 200)]},
        )
        assert {c.freq_key for c in cands} == {"hz_1"}
        assert {c.path.name for c in cands} == {"a.ss.h5"}
        assert all(10 <= c.ping < 20 or 90 <= c.ping < 100 for c in cands)
        assert any(c.ping >= 90 for c in cands) and any(c.ping < 20 for c in cands)

    def test_nothing_allowed_returns_empty(self):
        assert sample_background_candidates(self._index(), 3, np.random.default_rng(0), ping_ranges={}) == []


def test_place_catalog_projects_every_frequency(marker_survey):
    index, catalog = marker_survey
    placed = place_catalog(catalog, index, 25.0)
    ((path, freq_key),) = placed.keys()
    assert freq_key == "hz_1495000"
    assert {(t.target_id, ping) for items in placed[(path, freq_key)].values() for t, ping in items} == {
        ("0", 150), ("1", 450), ("2", 560),
    }
