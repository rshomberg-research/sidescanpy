"""
The range-decay classifier, its result classes and plots warn
DeprecationWarning (removal planned in 0.12).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sidescanpy.analysis import (
    TVGCandidateFit,
    TVGCandidateFitResult,
    fit_range_decay,
    fit_tvg_curve,
)
from sidescanpy.processing import backscatter_vs_range, compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams
from sidescanpy.viz import tvg as viz_tvg
from tests.unit.synthetic import make_range_decay_dataset


@pytest.fixture(scope="module")
def ds():
    return compute_geometric_ranges(make_range_decay_dataset(n_pings=30))


class TestDeprecations:
    def test_fit_range_decay_warns_once(self, ds):
        with pytest.warns(DeprecationWarning, match="fit_range_decay is deprecated") as record:
            result = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=80.0)
        assert result.fit is not None
        assert len([w for w in record if issubclass(w.category, DeprecationWarning)]) == 1
        assert "0.12" in str(record[0].message)

    def test_fit_tvg_curve_warns(self, ds):
        db = 20 * np.log10(np.clip(ds["backscatter_intensity"], 1e-6, None))
        binned = backscatter_vs_range(ds.assign(_db=db), "_db", "slant_range_m", bin_width_m=2.0)
        with pytest.warns(DeprecationWarning, match="fit_tvg_curve"):
            fit_tvg_curve(binned, absorption_db_per_m=0.05)

    def test_candidate_classes_warn(self):
        with pytest.warns(DeprecationWarning, match="TVGCandidateFit"):
            cand = TVGCandidateFit(name="flat", params=TVGParams(), scale=0.0, rmse_db=1.0, fixed_scale_rmse_db=1.0,
                                   r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=10)
        with pytest.warns(DeprecationWarning, match="TVGCandidateFitResult"):
            TVGCandidateFitResult(candidates=[cand], best=cand, observed_range_m=np.zeros(1),
                                  observed_loss_db=np.zeros(1))

    @pytest.mark.parametrize("name", ["plot_range_decay_fit", "plot_range_decay_residuals"])
    def test_decay_plots_warn(self, ds, name):
        with pytest.warns(DeprecationWarning):
            result = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=80.0)
        with pytest.warns(DeprecationWarning, match=name) as record:
            fig, _ = getattr(viz_tvg, name)(result)
        assert len([w for w in record if issubclass(w.category, DeprecationWarning)]) == 1
        plt.close(fig)

    def test_candidate_plot_and_limits_warn(self, ds):
        with pytest.warns(DeprecationWarning):
            result = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=80.0)
        fig, ax = plt.subplots()
        with pytest.warns(DeprecationWarning, match="plot_candidate_fit"):
            viz_tvg.plot_candidate_fit(result.fit.best, result.fit.observed_range_m, result.fit.observed_loss_db, ax=ax)
        with pytest.warns(DeprecationWarning, match="set_fit_window_limits"):
            viz_tvg.set_fit_window_limits(ax, result)
        plt.close(fig)
