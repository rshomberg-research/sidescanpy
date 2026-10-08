"""grids.recommend_resolution: half the finer of ping spacing and sample interval."""

import pytest

from sidescanpy.analysis.geometry import PingGeometry, measure_ping_geometry
from sidescanpy.grids import recommend_resolution


@pytest.mark.parametrize("spacing, interval, expected", [
    (0.4, 0.1, 0.05),   # sample interval finer
    (0.06, 0.1, 0.03),  # ping spacing finer
    (0.2, None, 0.1),   # no slant range -> half the ping spacing
    (None, 0.1, None),  # no ping spacing -> no recommendation
    (None, None, None),
])
def test_from_geometry(spacing, interval, expected):
    g = PingGeometry(ping_spacing_m_median=spacing, sample_interval_m=interval)
    assert recommend_resolution(g) == expected


def test_from_dataset_matches_measured_geometry():
    from sidescanpy.processing import correct
    from tests.conftest import make_minimal_sidescan_dataset

    ds = make_minimal_sidescan_dataset(n_pings=30, n_samples=50)
    ds = correct(ds, ["geometric", "georef"])
    g = measure_ping_geometry(ds)
    assert g.ping_spacing_m_median is not None and g.sample_interval_m is not None
    assert recommend_resolution(ds) == min(g.ping_spacing_m_median, g.sample_interval_m) / 2.0
