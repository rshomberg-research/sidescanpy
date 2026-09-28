"""
Unit tests for sidescanpy.viz.nav_attitude.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sidescanpy.analysis.nav_attitude import nav_attitude_summary
from sidescanpy.viz.nav_attitude import plot_nav_attitude


def _make_ds(*, n_pings: int = 10, **overrides) -> xr.Dataset:
    data_vars = {
        "heading_deg": (["ping"], np.linspace(10.0, 100.0, n_pings, dtype=np.float64)),
        "pitch_deg": (["ping"], np.zeros(n_pings)),
        "roll_deg": (["ping"], np.zeros(n_pings)),
        "altitude_m": (["ping"], np.full(n_pings, 5.0)),
        "depth_m": (["ping"], np.full(n_pings, 20.0)),
        "heave_m": (["ping"], np.zeros(n_pings)),
        "latitude": (["ping"], np.linspace(40.0, 40.01, n_pings)),
        "longitude": (["ping"], np.linspace(-70.0, -69.99, n_pings)),
    }
    data_vars.update(overrides)
    return xr.Dataset(data_vars)


class TestPlotNavAttitude:
    def test_returns_four_axes(self):
        ds = _make_ds()
        fig, axes = plot_nav_attitude(ds)
        assert len(axes) == 4
        plt.close(fig)

    def test_ping_offset_shifts_x_axis(self):
        ds = _make_ds(n_pings=5)
        fig, (ax_heading, _, _, _) = plot_nav_attitude(ds, ping_offset=1000)
        # the heading trace is plotted last, after the convention axhlines
        xdata = ax_heading.lines[-1].get_xdata()
        assert xdata[0] == 1000
        assert xdata[-1] == 1004
        plt.close(fig)

    def test_out_of_range_heading_gets_marked(self):
        ds = _make_ds(n_pings=3, heading_deg=(["ping"], np.array([-400.0, -300.0, -250.0])))
        result = nav_attitude_summary(ds)
        fig, (ax_heading, _, _, _) = plot_nav_attitude(ds, result=result)
        # a scatter collection was added for the out-of-range points
        assert len(ax_heading.collections) >= 1
        plt.close(fig)

    def test_no_out_of_range_marker_when_heading_in_range(self):
        ds = _make_ds(n_pings=5, heading_deg=(["ping"], np.array([0.0, 90.0, 180.0, 270.0, 359.9])))
        result = nav_attitude_summary(ds)
        fig, (ax_heading, _, _, _) = plot_nav_attitude(ds, result=result)
        assert len(ax_heading.collections) == 0
        plt.close(fig)

    def test_missing_position_vars_does_not_raise(self):
        ds = _make_ds().drop_vars(["latitude", "longitude"])
        fig, (_, _, _, ax_track) = plot_nav_attitude(ds)
        assert len(ax_track.lines) == 0
        plt.close(fig)

    def test_missing_attitude_vars_does_not_raise(self):
        ds = _make_ds().drop_vars(["pitch_deg", "roll_deg"])
        fig, axes = plot_nav_attitude(ds)
        assert len(axes) == 4
        plt.close(fig)
