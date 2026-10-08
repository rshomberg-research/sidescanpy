"""
Unit tests for sidescanpy.cli._recommend.
"""

import pytest

from sidescanpy.analysis.beam import BeamGeometryResult
from sidescanpy.analysis.characterize import UnitsResult, ValueSummary
from sidescanpy.analysis.geometry import PingGeometry
from sidescanpy.analysis.nadir import NadirResult
from sidescanpy.analysis.tvg import RangeDecayResult
from sidescanpy.analysis.usable_range import UsableRangeResult
from sidescanpy.cli._recommend import render_recommended_config


def _decay(best_candidate, best_rmse_db, tvg_status):
    return RangeDecayResult(best_candidate=best_candidate, best_rmse_db=best_rmse_db, tvg_status=tvg_status)


class TestRenderRecommendedConfig:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            source_name="a.ss.h5", freq_key="hz_1495000", ping_range=(0, 5000), value_var="uncorrected",
            values=ValueSummary(dtype="float32"),
            nadir=NadirResult(nadir_type="gap", altitude_sample_median=12.3, first_bottom_return_sample_median=145.2),
            beam=BeamGeometryResult(confidence="determined", near_angle_deg_median=25.0, far_angle_deg=60.0, beam_width_deg=35.0),
            usable_range=UsableRangeResult(usable_range_m=46.2),
            decay=_decay("absorption_only", 1.23, "absorption_only"),
            units=UnitsResult(units="raw_counts"),
            geometry=PingGeometry(ping_spacing_m_median=0.18, swath_width_m=132.4),
        )
        kwargs.update(overrides)
        return kwargs

    def test_absorption_only_recommends_no_spreading(self):
        text = render_recommended_config(**self._base_kwargs())
        assert "no_spreading: true" in text
        assert "no_absorption: true" not in text

    def test_spreading_only_recommends_no_absorption(self):
        decay = _decay("spreading_only", 0.5, "spreading_only")
        text = render_recommended_config(**self._base_kwargs(decay=decay))
        assert "no_absorption: true" in text
        assert "no_spreading: true" not in text

    def test_full_recommends_both_disabled(self):
        decay = _decay("flat", 0.1, "full")
        text = render_recommended_config(**self._base_kwargs(decay=decay))
        assert "no_spreading: true" in text
        assert "no_absorption: true" in text

    def test_none_recommends_no_overrides(self):
        decay = _decay("full_physics", 2.0, "none")
        text = render_recommended_config(**self._base_kwargs(decay=decay))
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text

    def test_ambiguous_recommends_nothing_and_says_so(self):
        decay = _decay("spreading_only", 4.0, "ambiguous")
        text = render_recommended_config(**self._base_kwargs(decay=decay))
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text
        assert "not confident enough" in text

    def test_no_decay_notes_missing_usable_range(self):
        text = render_recommended_config(**self._base_kwargs(decay=None))
        assert "could not be determined" in text
        assert "no_spreading: true" not in text
        assert "no_absorption: true" not in text

    def test_resolution_m_emitted_from_geometry(self):
        # No sample interval in the base geometry: half the ping spacing.
        text = render_recommended_config(**self._base_kwargs())
        assert "resolution_m: 0.0900" in text

    def test_resolution_m_is_half_the_finer_of_spacing_and_sample_interval(self):
        # Oversample to half the finest native axis and interpolate.
        # Typical case: sample interval much finer than ping spacing.
        geometry = PingGeometry(ping_spacing_m_median=0.30, swath_width_m=132.4, sample_interval_m=0.02)
        text = render_recommended_config(**self._base_kwargs(geometry=geometry))
        assert "resolution_m: 0.0100" in text
        assert "max(ping spacing, sample interval) = 0.3000" in text  # coarse alternative still shown

    def test_resolution_m_uses_ping_spacing_when_it_is_finer(self):
        geometry = PingGeometry(ping_spacing_m_median=0.02, swath_width_m=132.4, sample_interval_m=0.05)
        text = render_recommended_config(**self._base_kwargs(geometry=geometry))
        assert "resolution_m: 0.0100" in text

    def test_missing_geometry_notes_manual_value_needed(self):
        text = render_recommended_config(**self._base_kwargs(geometry=PingGeometry()))
        assert "resolution_m:" not in text
        assert "manually chosen value" in text

    def test_no_beam_omits_beam_geometry_block(self):
        text = render_recommended_config(**self._base_kwargs(beam=None))
        assert "beam_geometry" not in text

    def test_every_emitted_key_is_yaml_parseable(self):
        import yaml

        text = render_recommended_config(**self._base_kwargs())
        parsed = yaml.safe_load(text)
        assert parsed["no_spreading"] is True
        assert parsed["resolution_m"] == pytest.approx(0.09)
