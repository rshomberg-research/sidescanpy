"""
Unit tests for sidescanpy.viz.waterfall — focused on the ping_offset
parameter added to support rendering an already-extracted segment of a
larger (e.g. multi-file concatenated) survey with correct global
y-axis labels.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sidescanpy.viz.waterfall import _compute_y_full, plot_waterfall


class TestComputeYFullPingOffset:
    def test_default_offset_is_local_arange(self, sample_dataset):
        y = _compute_y_full(sample_dataset, "ping")
        np.testing.assert_array_equal(y, np.arange(sample_dataset.sizes["ping"]))

    def test_nonzero_offset_shifts_ping_values(self, sample_dataset):
        n = sample_dataset.sizes["ping"]
        y = _compute_y_full(sample_dataset, "ping", ping_offset=500)
        np.testing.assert_array_equal(y, np.arange(500, 500 + n))

    def test_offset_ignored_for_time_axis(self, sample_dataset):
        y_default = _compute_y_full(sample_dataset, "time")
        y_offset = _compute_y_full(sample_dataset, "time", ping_offset=500)
        np.testing.assert_array_equal(y_default, y_offset)

    def test_offset_ignored_for_along_track_axis(self, sample_dataset):
        y_default = _compute_y_full(sample_dataset, "along_track")
        y_offset = _compute_y_full(sample_dataset, "along_track", ping_offset=500)
        np.testing.assert_array_equal(y_default, y_offset)


class TestPlotWaterfallPingOffset:
    def test_default_title_shows_local_ping_range(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, y_axis="ping", ax=ax)
            n = sample_dataset.sizes["ping"]
            assert f"pings 0–{n}" in ax.get_title()
        finally:
            plt.close(fig)

    def test_ping_offset_shows_global_ping_range_in_title(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            n = sample_dataset.sizes["ping"]
            plot_waterfall(sample_dataset, y_axis="ping", ping_offset=1000, ax=ax)
            assert f"pings 1000–{1000 + n}" in ax.get_title()
        finally:
            plt.close(fig)

    def test_ping_offset_does_not_error_for_other_y_axes(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, y_axis="along_track", ping_offset=1000, ax=ax)
        finally:
            plt.close(fig)
