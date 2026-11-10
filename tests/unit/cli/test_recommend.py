"""
Unit tests for sidescanpy.cli._recommend.
"""

import pytest

from sidescanpy.analysis.beam import BeamGeometryResult
from sidescanpy.analysis.characterize import UnitsResult, ValueSummary
from sidescanpy.analysis.geometry import PingGeometry
from sidescanpy.analysis.nadir import NadirResult
from sidescanpy.analysis.tvg_fit import TVGCheckResult, TVGFitResult, TVGModelFit, tvg_model_name
from sidescanpy.analysis.usable_range import UsableRangeResult
from sidescanpy.cli._recommend import render_recommended_config


def _fit(n, alpha, rms=0.5, **kw):
    name = tvg_model_name(n, alpha > 0)
    models = {
        name: TVGModelFit(name=name, spreading_loss_db=n, absorption_db_per_m=alpha, offset_db=50.0, rms_db=rms),
        "40 log R": TVGModelFit(name="40 log R", spreading_loss_db=40.0, absorption_db_per_m=0.0, offset_db=60.0,
                                rms_db=rms + 0.4),
    }
    defaults = dict(models=models, best=name, runner_up="40 log R", window_m=(15.5, 60.5), window_source="knee",
                    bin_width_m=1.0, free_n=n + 1.0, free_n_no_absorption=n + 10.0, contrast_db=20.0)
    defaults.update(kw)
    return TVGFitResult(**defaults)


class TestRenderRecommendedConfig:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            source_name="a.ss.h5", freq_key="hz_1495000", ping_range=(0, 5000), value_var="uncorrected",
            values=ValueSummary(dtype="float32"),
            nadir=NadirResult(nadir_type="gap", altitude_sample_median=12.3, first_bottom_return_sample_median=145.2),
            beam=BeamGeometryResult(confidence="determined", near_angle_deg_median=25.0, far_angle_deg=60.0, beam_width_deg=35.0),
            usable_range=UsableRangeResult(usable_range_m=46.2),
            tvg_fit=_fit(30.0, 0.14),
            tvg_check=TVGCheckResult(spread_db=0.52, trend_db_per_10m=-0.1),
            units=UnitsResult(units="raw_counts"),
            geometry=PingGeometry(ping_spacing_m_median=0.18, swath_width_m=132.4),
        )
        kwargs.update(overrides)
        return kwargs

    def test_best_model_terms_written_directly(self):
        import yaml

        parsed = yaml.safe_load(render_recommended_config(**self._base_kwargs()))
        assert parsed["spreading_loss_db"] == 30
        assert parsed["absorption_db_per_m"] == pytest.approx(0.14)
        assert "no_spreading" not in parsed and "no_absorption" not in parsed

    def test_evidence_in_comments(self):
        text = render_recommended_config(**self._base_kwargs())
        assert "best model 30 log R + abs, 0.50 dB RMS over 15.5-60.5 m" in text
        assert "Runner-up 40 log R (0.90 dB)" in text
        assert "0.52 dB spread" in text
        assert "removed: none" in text

    @pytest.mark.parametrize(("n", "alpha", "status"), [(0.0, 0.0, "full"), (20.0, 0.0, "absorption_only"),
                                                          (0.0, 0.14, "spreading_only")])
    def test_status_derived_from_model(self, n, alpha, status):
        import yaml

        text = render_recommended_config(**self._base_kwargs(tvg_fit=_fit(n, alpha)))
        parsed = yaml.safe_load(text)
        assert parsed["spreading_loss_db"] == n
        assert parsed["absorption_db_per_m"] == pytest.approx(alpha)
        assert f"removed: {status}" in text

    def test_fallback_and_low_contrast_flagged(self):
        fit = _fit(0.0, 0.0, window_source="end_of_record", low_contrast=True, contrast_db=2.0)
        text = render_recommended_config(**self._base_kwargs(tvg_fit=fit))
        assert "No noise knee" in text
        assert "Low contrast (2.0 dB" in text

    def test_no_fit_writes_no_tvg_keys(self):
        text = render_recommended_config(**self._base_kwargs(tvg_fit=TVGFitResult(notes=["empty profile"]),
                                                             tvg_check=None))
        assert "no model could be fitted" in text
        assert "spreading_loss_db:" not in text
        assert "absorption_db_per_m:" not in text

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
        assert parsed["spreading_loss_db"] == 30
        assert parsed["resolution_m"] == pytest.approx(0.09)
