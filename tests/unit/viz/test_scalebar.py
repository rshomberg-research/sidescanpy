"""
Unit tests for sidescanpy.viz.scalebar.

2026 rshomberg@uri.edu
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from sidescanpy.viz.scalebar import _format_scale_label, _nice_scale_length, add_scale_bar


class TestNiceScaleLength:
    @pytest.mark.parametrize("target,expected", [
        (37.0, 50.0),
        (12.0, 10.0),
        (0.6, 0.5),
        (340.0, 200.0),
        (3400.0, 2000.0),
        (999.0, 1000.0),
    ])
    def test_rounds_to_1_2_5_sequence(self, target, expected):
        assert _nice_scale_length(target) == pytest.approx(expected)

    def test_non_positive_returns_one(self):
        assert _nice_scale_length(0.0) == 1.0
        assert _nice_scale_length(-5.0) == 1.0

    def test_nan_returns_one(self):
        assert _nice_scale_length(float("nan")) == 1.0


class TestFormatScaleLabel:
    def test_meters_below_1000(self):
        assert _format_scale_label(50.0) == "50 m"

    def test_switches_to_km_at_1000(self):
        assert _format_scale_label(1000.0) == "1 km"

    def test_fractional_km(self):
        assert _format_scale_label(1500.0) == "1.5 km"


class TestAddScaleBar:
    def _ax(self, xlim=(0.0, 1000.0), ylim=(0.0, 500.0)):
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return fig, ax

    def test_adds_a_line_and_text(self):
        fig, ax = self._ax()
        n_lines_before = len(ax.lines)
        n_texts_before = len(ax.texts)
        add_scale_bar(ax)
        # main bar + 2 end ticks = 3 new lines
        assert len(ax.lines) == n_lines_before + 3
        assert len(ax.texts) == n_texts_before + 1
        plt.close(fig)

    def test_meters_per_data_unit_scales_the_label(self):
        """1000 data-units wide axes at meters_per_data_unit=1 -> a
        ~200m bar (20% of 1000); at meters_per_data_unit=1000 (e.g. km
        units) -> a ~200,000m (200km) bar for the same view."""
        fig, ax = self._ax()
        add_scale_bar(ax, meters_per_data_unit=1.0)
        label_1 = ax.texts[-1].get_text()
        plt.close(fig)

        fig, ax = self._ax()
        add_scale_bar(ax, meters_per_data_unit=1000.0)
        label_1000 = ax.texts[-1].get_text()
        plt.close(fig)

        assert label_1 == "200 m"
        assert label_1000 == "200 km"

    def test_invalid_location_raises(self):
        fig, ax = self._ax()
        with pytest.raises(ValueError, match="Unknown location"):
            add_scale_bar(ax, location="middle")
        plt.close(fig)

    def test_degenerate_meters_per_data_unit_is_a_noop(self):
        """matplotlib itself auto-widens a truly zero-width set_xlim
        call, so the way to reach add_scale_bar's own degenerate-span
        guard through the public API is a degenerate conversion factor,
        not degenerate limits."""
        fig, ax = self._ax()
        n_lines_before = len(ax.lines)
        add_scale_bar(ax, meters_per_data_unit=0.0)
        assert len(ax.lines) == n_lines_before
        plt.close(fig)

    @pytest.mark.parametrize("location", ["lower right", "lower left", "upper right", "upper left"])
    def test_every_location_accepted(self, location):
        fig, ax = self._ax()
        add_scale_bar(ax, location=location)
        assert len(ax.texts) == 1
        plt.close(fig)
