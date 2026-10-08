"""
Shared fixtures for grids unit tests.

Provides 3D (ping, channel, sample) channel datasets matching the
production schema (see core/schema.py REQUIRED_DIMS), with port (channel=0)
and starboard (channel=1) samples at genuinely distinct spatial locations.
"""

import numpy as np
import pytest
import xarray as xr


def make_channel_3d(
    num_pings: int = 500,
    num_samples: int = 4792,
    freq_hz: int = 455_000,
    processing_level: str = "raw",
    seed: int = 42,
) -> xr.Dataset:
    """
    Build a channel dataset with (ping, channel, sample) backscatter and
    per-sample coordinates, mirroring what compute_georeferencing() produces.

    Port (channel=0) samples are offset west of the track, starboard
    (channel=1) samples are offset east, with offset magnitude increasing
    with sample index — mimicking a real sonar fan spreading away from nadir.
    """
    south, west, north, east = 39.0, -71.0, 40.0, -70.0

    pings = np.arange(num_pings)
    channels = np.arange(2)
    samples = np.arange(num_samples)

    lat_track = np.linspace(south, north, num_pings)
    lon_track = np.linspace(west, east, num_pings)

    # Across-track spread in degrees longitude, growing with sample index
    # (far-range samples are farther from nadir than near-range samples).
    across_track_deg = np.linspace(0.0, 0.02, num_samples)

    lat3d = np.tile(lat_track[:, np.newaxis, np.newaxis], (1, 2, num_samples))
    lon3d = np.empty((num_pings, 2, num_samples))
    lon3d[:, 0, :] = lon_track[:, np.newaxis] - across_track_deg[np.newaxis, :]
    lon3d[:, 1, :] = lon_track[:, np.newaxis] + across_track_deg[np.newaxis, :]

    rng = np.random.default_rng(seed)
    backscatter = rng.random((num_pings, 2, num_samples))
    # Bright feature band, offset per side so port/starboard aren't identical.
    band = slice(num_samples // 2, num_samples // 2 + 20)
    backscatter[:, :, band] *= 2.0

    return xr.Dataset(
        {
            "backscatter_intensity": (["ping", "channel", "sample"], backscatter),
            "latitude_per_sample": (["ping", "channel", "sample"], lat3d),
            "longitude_per_sample": (["ping", "channel", "sample"], lon3d),
            "ping": (["ping"], pings),
            "channel": (["channel"], channels),
            "sample": (["sample"], samples),
        },
        attrs={
            "freq_hz_start": freq_hz,
            "processing_level": processing_level,
        },
    )


@pytest.fixture
def channel_3d_real_proportions() -> xr.Dataset:
    """Full-size 3D channel matching real survey proportions: (500, 2, 4792)."""
    return make_channel_3d(num_pings=500, num_samples=4792)


@pytest.fixture
def channel_3d() -> xr.Dataset:
    """Smaller 3D channel for fast per-test resampling."""
    return make_channel_3d(num_pings=50, num_samples=128)
