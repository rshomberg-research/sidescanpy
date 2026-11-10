"""
Tests for sidescanpy.viz.tvg.plot_tvg_fit.
"""

import matplotlib.pyplot as plt
import pytest

from sidescanpy.analysis import TVGFitResult, check_tvg, fit_tvg_models
from sidescanpy.processing import compute_geometric_ranges
from sidescanpy.viz.tvg import plot_tvg_fit
from tests.unit.synthetic import make_range_decay_dataset


@pytest.fixture(scope="module")
def fit_and_check():
    ds = compute_geometric_ranges(make_range_decay_dataset(n_pings=60, absorption_db_per_m=0.05))
    fit = fit_tvg_models(ds)
    return fit, check_tvg(ds, fit.tvg_params, fit.window_m)


class TestPlotTVGFit:
    def test_fit_and_check_new_figure(self, fit_and_check):
        fit, check = fit_and_check
        fig, (ax, ax_check) = plot_tvg_fit(fit, check)
        assert ax_check is not None
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "RMS in window"
        labels = [t.get_text() for t in legend.get_texts()]
        assert any(label.startswith(fit.best) and "(best)" in label for label in labels)
        assert ax_check.get_ylim() == (-5.0, 5.0)
        assert any("spread" in t.get_text() for t in ax_check.texts)
        plt.close(fig)

    def test_fit_only(self, fit_and_check):
        fit, _ = fit_and_check
        fig, (ax, ax_check) = plot_tvg_fit(fit)
        assert ax_check is None
        assert ax.get_xlabel() == "slant range (m)"
        plt.close(fig)

    def test_draws_into_given_axes(self, fit_and_check):
        fit, check = fit_and_check
        fig, (a, b) = plt.subplots(2, 1)
        out_fig, (ax, ax_check) = plot_tvg_fit(fit, check, ax=a, ax_check=b)
        assert out_fig is fig and ax is a and ax_check is b
        plt.close(fig)

    def test_check_without_ax_check_raises(self, fit_and_check):
        fit, check = fit_and_check
        fig, a = plt.subplots()
        with pytest.raises(ValueError, match="ax_check"):
            plot_tvg_fit(fit, check, ax=a)
        plt.close(fig)

    def test_empty_result(self):
        fig, (ax, _) = plot_tvg_fit(TVGFitResult())
        assert any("no level" in t.get_text() for t in ax.texts)
        plt.close(fig)

    def test_fallback_flagged_on_plot(self):
        ds = compute_geometric_ranges(make_range_decay_dataset(n_pings=60, spreading_loss_db=0.0))
        fit = fit_tvg_models(ds)
        fig, (ax, _) = plot_tvg_fit(fit)
        assert any("no noise knee" in t.get_text() for t in ax.texts)
        plt.close(fig)
