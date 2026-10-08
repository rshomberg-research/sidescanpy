"""
Synthetic sidescan datasets shared by the analysis and processing unit tests.
"""

import numpy as np
import pandas as pd
import xarray as xr


def make_raw_dataset(
    *,
    n_pings: int = 20,
    n_samples: int = 200,
    sample_rate_hz: float = 1000.0,
    backscatter: np.ndarray | None = None,
    freq_hz: float = 200_000,
) -> xr.Dataset:
    """Minimal valid raw sidescan dataset (uncorrected, two channels)."""
    if backscatter is None:
        backscatter = np.ones((n_pings, 2, n_samples), dtype=np.float32)
    else:
        n_pings, _, n_samples = backscatter.shape

    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], backscatter.astype(np.float32)),
        "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
        "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
        "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
        "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
    }
    attrs = {
        "freq_hz_start": freq_hz,
        "freq_hz_end": freq_hz,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }
    return xr.Dataset(data_vars, coords, attrs)
