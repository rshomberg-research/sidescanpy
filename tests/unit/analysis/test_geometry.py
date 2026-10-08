"""
Unit tests for sidescanpy.analysis.geometry.
"""

import numpy as np
import pytest

from sidescanpy.analysis.geometry import measure_ping_geometry
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
        assert result.ping_spacing_m_median is not None
        assert result.ping_spacing_m_median > 0

    def test_too_few_pings_returns_none(self):
        ds = _uniform_track_dataset(n_pings=1)
        result = measure_ping_geometry(ds)
        assert result.ping_spacing_m_median is None

    def test_no_across_track_m_returns_none_swath(self):
        ds = _uniform_track_dataset()
        assert "across_track_m" not in ds
        result = measure_ping_geometry(ds)
        assert result.swath_width_m is None

    def test_swath_width_measured_when_across_track_present(self):
        ds = _uniform_track_dataset()
        n_pings, n_channels, n_samples = ds.sizes["ping"], ds.sizes["channel"], ds.sizes["sample"]
        across = np.linspace(-50.0, 50.0, n_samples)
        ds["across_track_m"] = (
            ("ping", "channel", "sample"),
            np.broadcast_to(across, (n_pings, n_channels, n_samples)).copy(),
        )
        result = measure_ping_geometry(ds)
        assert result.swath_width_m == pytest.approx(100.0, rel=0.05)

    def test_sample_interval_measured_from_slant_range(self):
        # sound_vel_ms=1500.0, sample_rate_hz=400_000 (tests/conftest.py's
        # own fixture defaults) -> 1500 / (2 * 400_000) = 0.001875 m.
        ds = compute_geometric_ranges(_uniform_track_dataset())
        result = measure_ping_geometry(ds)
        assert result.sample_interval_m == pytest.approx(0.001875)

    def test_no_slant_range_m_returns_none_sample_interval(self):
        ds = _uniform_track_dataset()
        assert "slant_range_m" not in ds
        result = measure_ping_geometry(ds)
        assert result.sample_interval_m is None

    def test_outlier_jump_does_not_dominate_median(self):
        ds = _uniform_track_dataset(n_pings=100)
        lat = ds["latitude"].values.copy()
        lat[50:] += 1.0  # one huge jump, like detect_gaps' own injected-gap test
        ds["latitude"] = ("ping", lat)
        result = measure_ping_geometry(ds)
        # The median should still reflect the ~99 normal-spacing steps,
        # not be dragged toward the one outlier jump.
        assert result.ping_spacing_m_median < 100.0

    def test_to_dict_is_json_serializable(self):
        import json

        result = measure_ping_geometry(_uniform_track_dataset())
        assert set(json.loads(json.dumps(result.to_dict()))) == {
            "ping_spacing_m_median", "swath_width_m", "sample_interval_m",
        }
