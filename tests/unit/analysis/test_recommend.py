"""
Unit tests for sidescanpy.analysis.recommend.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.analysis.investigation import StageResult
from sidescanpy.analysis.recommend import measure_ping_geometry, render_recommended_config
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.conftest import make_minimal_sidescan_dataset


def _uniform_track_dataset(n_pings=100):
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=50)
    ds["latitude"] = ("ping", np.linspace(41.0000, 41.0100, n_pings))
    ds["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
    return ds


class TestMeasurePingGeometry:
    def test_measures_spacing_on_uniform_track(self):
        ds = _uniform_track_dataset()
        result = measure_ping_geometry(ds)
        assert result["ping_spacing_m_median"] is not None
        assert result["ping_spacing_m_median"] > 0

    def test_too_few_pings_returns_none(self):
        ds = _uniform_track_dataset(n_pings=1)
        result = measure_ping_geometry(ds)
        assert result["ping_spacing_m_median"] is None

    def test_no_across_track_m_returns_none_swath(self):
        ds = _uniform_track_dataset()
        assert "across_track_m" not in ds
        result = measure_ping_geometry(ds)
        assert result["swath_width_m"] is None

    def test_swath_width_measured_when_across_track_present(self):
        ds = _uniform_track_dataset()
        n_pings, n_channels, n_samples = ds.sizes["ping"], ds.sizes["channel"], ds.sizes["sample"]
        across = np.linspace(-50.0, 50.0, n_samples)
        ds["across_track_m"] = (
            ("ping", "channel", "sample"),
            np.broadcast_to(across, (n_pings, n_channels, n_samples)).copy(),
        )
        result = measure_ping_geometry(ds)
        assert result["swath_width_m"] == pytest.approx(100.0, rel=0.05)

    def test_sample_interval_measured_from_slant_range(self):
        # sound_vel_ms=1500.0, sample_rate_hz=400_000 (tests/conftest.py's
        # own fixture defaults) -> 1500 / (2 * 400_000) = 0.001875 m.
        ds = compute_geometric_ranges(_uniform_track_dataset())
        result = measure_ping_geometry(ds)
        assert result["sample_interval_m"] == pytest.approx(0.001875)

    def test_no_slant_range_m_returns_none_sample_interval(self):
        ds = _uniform_track_dataset()
        assert "slant_range_m" not in ds
        result = measure_ping_geometry(ds)
        assert result["sample_interval_m"] is None

    def test_outlier_jump_does_not_dominate_median(self):
        ds = _uniform_track_dataset(n_pings=100)
        lat = ds["latitude"].values.copy()
        lat[50:] += 1.0  # one huge jump, like detect_gaps' own injected-gap test
        ds["latitude"] = ("ping", lat)
        result = measure_ping_geometry(ds)
        # The median should still reflect the ~99 normal-spacing steps,
        # not be dragged toward the one outlier jump.
        assert result["ping_spacing_m_median"] < 100.0


def _stage(stage, metrics=None, classification=None, notes=None):
    return StageResult(stage=stage, metrics=metrics or {}, classification=classification, notes=notes or [])


class TestRenderRecommendedConfig:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            source_name="a.ss.h5", freq_key="hz_1495000", ping_range=(0, 5000), value_var="backscatter_raw",
            stage1=_stage("stage1_characterize", {"dtype": "float32"}),
            stage2=_stage("stage2_nadir", {"altitude_sample_median": 12.3, "first_bottom_return_sample_median": 145.2}, classification="gap"),
            stage2_5=_stage("stage2_5_beam_geometry", {"near_angle_deg_median": 25.0, "far_angle_deg": 60.0, "beam_width_deg": 35.0}, classification="determined"),
            stage3=_stage("stage3_usable_range", {"usable_range_m": 46.2}),
            stage4=_stage("stage4_decay", {"best_candidate": "absorption_only", "best_rmse_db": 1.23}, classification="absorption_only"),
            stage5=_stage("stage5_units", {}, classification="raw_counts"),
            geometry={"ping_spacing_m_median": 0.18, "swath_width_m": 132.4},
        )
        kwargs.update(overrides)
        return kwargs

    def test_absorption_only_recommends_no_spreading(self):
        text = render_recommended_config(**self._base_kwargs())
        assert "no_spreading: true" in text
        assert "no_absorption: true" not in text

    def test_spreading_only_recommends_no_absorption(self):
        stage4 = _stage("stage4_decay", {"best_candidate": "spreading_only", "best_rmse_db": 0.5}, classification="spreading_only")
        text = render_recommended_config(**self._base_kwargs(stage4=stage4))
        assert "no_absorption: true" in text
        assert "no_spreading: true" not in text

    def test_full_recommends_both_disabled(self):
        stage4 = _stage("stage4_decay", {"best_candidate": "flat", "best_rmse_db": 0.1}, classification="full")
        text = render_recommended_config(**self._base_kwargs(stage4=stage4))
        assert "no_spreading: true" in text
        assert "no_absorption: true" in text

    def test_none_recommends_no_overrides(self):
        stage4 = _stage("stage4_decay", {"best_candidate": "full_physics", "best_rmse_db": 2.0}, classification="none")
        text = render_recommended_config(**self._base_kwargs(stage4=stage4))
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text

    def test_ambiguous_recommends_nothing_and_says_so(self):
        stage4 = _stage("stage4_decay", {"best_candidate": "spreading_only", "best_rmse_db": 4.0}, classification="ambiguous")
        text = render_recommended_config(**self._base_kwargs(stage4=stage4))
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text
        assert "not confident enough" in text

    def test_stage4_none_notes_missing_usable_range(self):
        text = render_recommended_config(**self._base_kwargs(stage4=None))
        assert "could not be determined" in text
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text

    def test_resolution_m_emitted_from_geometry(self):
        text = render_recommended_config(**self._base_kwargs())
        assert "resolution_m: 0.1800" in text

    def test_resolution_m_uses_coarser_of_spacing_and_sample_interval(self):
        # Typical real case: ping spacing (along-track) coarser than the
        # range-axis sample interval -- resolution_m should still be the
        # ping spacing, not the finer sample interval.
        geometry = {"ping_spacing_m_median": 0.18, "swath_width_m": 132.4, "sample_interval_m": 0.0019}
        text = render_recommended_config(**self._base_kwargs(geometry=geometry))
        assert "resolution_m: 0.1800" in text
        assert "sample interval" in text

    def test_resolution_m_uses_sample_interval_when_it_is_coarser(self):
        # Edge case: sample interval coarser than ping spacing -- max()
        # must pick it, not always default to ping spacing.
        geometry = {"ping_spacing_m_median": 0.02, "swath_width_m": 132.4, "sample_interval_m": 0.05}
        text = render_recommended_config(**self._base_kwargs(geometry=geometry))
        assert "resolution_m: 0.0500" in text

    def test_missing_geometry_notes_manual_value_needed(self):
        text = render_recommended_config(**self._base_kwargs(geometry={"ping_spacing_m_median": None, "swath_width_m": None}))
        assert "resolution_m:" not in text
        assert "manually chosen value" in text

    def test_stage2_5_none_omits_beam_geometry_block(self):
        text = render_recommended_config(**self._base_kwargs(stage2_5=None))
        assert "beam_geometry" not in text

    def test_every_emitted_key_is_yaml_parseable(self):
        import yaml

        text = render_recommended_config(**self._base_kwargs())
        parsed = yaml.safe_load(text)
        assert parsed["no_spreading"] is True
        assert parsed["resolution_m"] == pytest.approx(0.18)
