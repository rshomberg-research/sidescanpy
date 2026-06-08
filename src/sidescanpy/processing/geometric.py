"""
Geometric range computation for SideScan sonar data.

Computes per-sample slant range and ground range as data variables
attached to the dataset. Does NOT resample backscatter — original sample
positions and values are preserved. Downstream tools (xarray.interp,
rioxarray, etc.) handle resampling onto regular grids on demand.

Slant range is computed from sample index, sound velocity, and sample
rate:

    R_slant[ping, channel, sample] = sample_idx * sound_vel[ping] / (2 * sample_rate)

Ground range is computed from slant range and per-ping altitude using
the Pythagorean relation for a flat seafloor:

    R_ground = sqrt(R_slant^2 - altitude^2)   when R_slant > altitude
             = NaN                              when R_slant <= altitude
                                               (water column / nadir gap)

Both port and starboard channels share the same per-ping altitude (single
transducer assembly). Sound velocity is per-ping; per-sample variation
through the water column is not modeled.

The function does not modify processing_level. processing_history is
appended with a record of the computation.

2026 rshomberg@uri.edu
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from sidescanpy import __version__

logger = logging.getLogger(__name__)


_DEFAULT_SOUND_VEL_MS = 1500.0


def compute_geometric_ranges(ds: xr.Dataset) -> xr.Dataset:
    """
    Attach slant_range_m and ground_range_m data variables to the dataset.

    Both variables have shape ``(ping, channel, sample)`` and are computed
    from per-ping altitude and sound velocity. Backscatter is preserved
    unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Validated sidescan dataset. Must satisfy the schema (see
        ``sidescanpy.core.schema``). Per-ping altitude and sound velocity
        come from ``altitude_m`` and ``sound_vel_ms``; ``sample_rate_hz``
        is read from dataset attrs.

    Returns
    -------
    xr.Dataset
        New dataset (input is not modified) with two added data variables:

        - ``slant_range_m`` : per-sample slant range in meters
        - ``ground_range_m`` : per-sample ground range in meters
          (NaN inside the nadir gap)

        ``processing_level`` is unchanged. ``processing_history`` is
        appended with a ``compute_geometric_ranges`` entry.

    Notes
    -----
    NaN handling:

    - If a ping has NaN ``altitude_m``, all ground_range_m values for that
      ping are NaN.
    - If a ping has NaN ``sound_vel_ms``, the function falls back to
      1500 m/s for slant range computation. A warning is logged.
    - Samples where slant range <= altitude are NaN in ground_range_m
      (water column samples; sound hasn't reached the bottom yet).
    """
    ds.sidescan.validate()

    n_pings = ds.sizes["ping"]
    n_channels = ds.sizes["channel"]
    n_samples = ds.sizes["sample"]
    sample_rate = float(ds.attrs["sample_rate_hz"])

    # Resolve per-ping sound velocity, falling back to default for NaN values
    sound_vel = ds["sound_vel_ms"].values.astype(np.float64)
    nan_sv = np.isnan(sound_vel)
    if nan_sv.any():
        n_nan = int(nan_sv.sum())
        logger.warning(
            f"{n_nan} of {n_pings} pings have NaN sound_vel_ms; "
            f"falling back to {_DEFAULT_SOUND_VEL_MS} m/s for those pings"
        )
        sound_vel = np.where(nan_sv, _DEFAULT_SOUND_VEL_MS, sound_vel)

    altitude = ds["altitude_m"].values.astype(np.float64)

    # ----- slant range -----
    # slant[ping, channel, sample] = sample_idx * sound_vel[ping] / (2 * sample_rate)
    sample_idx = np.arange(n_samples, dtype=np.float64)              # (sample,)
    sample_interval_per_ping = sound_vel / (2.0 * sample_rate)        # (ping,)

    # Broadcast: (ping, 1, sample) * (sample,) → (ping, 1, sample)
    slant_per_ping = sample_interval_per_ping[:, None] * sample_idx[None, :]   # (ping, sample)
    # Tile across channels: (ping, channel, sample)
    slant_range = np.broadcast_to(
        slant_per_ping[:, None, :],
        (n_pings, n_channels, n_samples),
    ).astype(np.float32)

    # ----- ground range -----
    # ground = sqrt(slant^2 - altitude^2) where slant > altitude, else NaN
    # Broadcast altitude (ping,) → (ping, 1, 1)
    altitude_b = altitude[:, None, None]                             # (ping, 1, 1)

    with np.errstate(invalid="ignore"):
        diff = slant_range.astype(np.float64) ** 2 - altitude_b ** 2
        ground_range = np.where(
            (diff > 0) & ~np.isnan(altitude_b),
            np.sqrt(np.maximum(diff, 0.0)),
            np.nan,
        ).astype(np.float32)

    # ----- assemble output -----
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)

    out = out.assign(
        slant_range_m=(("ping", "channel", "sample"), slant_range),
        ground_range_m=(("ping", "channel", "sample"), ground_range),
    )

    out["slant_range_m"].attrs = {
        "units": "m",
        "long_name": "slant range (transducer to sample)",
        "description": "Distance from transducer along the slant path.",
    }
    out["ground_range_m"].attrs = {
        "units": "m",
        "long_name": "ground range (across the seafloor)",
        "description": (
            "Horizontal distance across the seafloor from nadir to the "
            "sample position, assuming a flat bottom at altitude_m below "
            "the towfish. NaN inside the nadir gap (slant <= altitude)."
        ),
    }

    # ----- update processing history -----
    history_str = out.attrs.get("processing_history", "[]")
    try:
        history = json.loads(history_str)
        if not isinstance(history, list):
            history = []
    except (TypeError, ValueError):
        history = []

    history.append(
        {
            "step": "compute_geometric_ranges",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {
                "altitude_source": "altitude_m",
                "sound_velocity_source": "sound_vel_ms",
                "default_sound_vel_ms": _DEFAULT_SOUND_VEL_MS,
                "n_pings_with_nan_sv": int(nan_sv.sum()),
            },
            "function": "sidescanpy.processing.geometric.compute_geometric_ranges",
            "version": __version__,
        }
    )
    out.attrs["processing_history"] = json.dumps(history)

    logger.info(
        "compute_geometric_ranges: added slant_range_m and ground_range_m "
        "for %d pings × %d channels × %d samples",
        n_pings, n_channels, n_samples,
    )

    return out