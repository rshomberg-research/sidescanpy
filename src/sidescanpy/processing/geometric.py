"""
Geometric range computation for SideScan sonar data.

Computes per-sample geometric variables as data variables attached to
the dataset. Does NOT resample backscatter — original sample positions
and values are preserved. Downstream tools (xarray.interp, rioxarray,
etc.) handle resampling onto regular grids on demand.

Outputs (all shape (ping, channel, sample))::

    slant_range_m       attitude-invariant time-of-flight range
    ground_range_m      sqrt(slant^2 - altitude^2), unsigned: horizontal
                        distance from the nadir point (flat bottom; the
                        same with or without pitch)
    across_track_m      signed Y component in the platform frame
                        (negative = port, positive = starboard)
    along_track_m       signed X component in the platform frame, the
                        pitch-induced offset (positive = forward of nadir)
    vertical_offset_m   Z component from platform = altitude_m for flat
                        bottom (broadcast). Future: extends to bathymetry.
    depth_per_sample_m  absolute water depth at sample location
                        = depth_m + altitude_m

Sign conventions:
    - across_track_m: port channel negative, starboard positive
    - along_track_m: positive forward of nadir (nose up puts samples forward)

Pitch model (flat bottom, fan beam perpendicular to the body x-axis):
    With the vehicle pitched by theta (positive nose up) at vertical
    altitude h, the fan plane tilts forward by theta. A sample at slant
    range R hits the seafloor at

        along_track_m  = h * tan(theta)                (every sample of the ping)
        across_track_m = ±sqrt(R^2 - (h / cos(theta))^2)

    and is on the seafloor only when R > h / cos(theta); nearer samples
    are water column and NaN in across_track_m and along_track_m. The
    horizontal distance from nadir, sqrt(along^2 + across^2), equals
    sqrt(R^2 - h^2) = ground_range_m. Pitch therefore shifts the whole
    ping forward or aft by a constant and shortens the across-track
    distance slightly; it does not rotate the swath.

    NaN pitch_deg is treated as 0 (no pitch correction) with a warning,
    as NaN sound velocity falls back to 1500 m/s.

Roll's effect on altitude is intentionally not modeled here. Future
work: when DVL altitude is available it's pre-corrected; when only
fathometer_reading_m is available, attitude correction is applied to
derive altitude_m.

processing_history is appended with a ``"geometric"`` record.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_GEOMETRIC, append_history
from sidescanpy.processing._per_frequency import per_frequency
from sidescanpy.processing.sample_axis import (
    DEFAULT_SOUND_VEL_MS,
    sample_index,
    sample_interval_m,
)

logger = logging.getLogger(__name__)


_DEFAULT_SOUND_VEL_MS = DEFAULT_SOUND_VEL_MS

#: Recorded in the ``geometric`` history record. Records without this key
#: (sidescanpy 0.7.6 and earlier) were made by a model that rotated the
#: ground vector by pitch (along = ground·sin θ), wrong for pitched pings.
PITCH_MODEL = "fan_beam_flat_bottom"


@per_frequency
def compute_geometric_ranges(
    ds: xr.Dataset,
    *,
    expose_unmasked_z: bool = False,
) -> xr.Dataset:
    """
    Attach per-sample geometric variables to the dataset.

    Parameters
    ----------
    ds : xr.Dataset or SideScanSurvey
        Validated sidescan dataset.
        A :class:`~sidescanpy.core.survey.SideScanSurvey` is corrected
        frequency by frequency with the same arguments; a new survey is
        returned and the input is not modified.
    expose_unmasked_z : bool, default False
        If True, vertical_offset_m and depth_per_sample_m are written as
        per-ping values broadcast to (ping, channel, sample) without
        masking the water column. The Z-axis variables become well-defined
        per-ping constants for every sample. The default (False) masks
        them with the same NaN mask as ground_range_m for consistency.

    Returns
    -------
    xr.Dataset or SideScanSurvey
        Same type as ``ds``: a new dataset (input not modified) with six added per-sample data
        variables and a ``"geometric"`` processing_history record.

    Notes
    -----
    NaN handling:
      - NaN altitude_m for a ping → all per-sample geometry NaN for that
        ping (slant_range_m is unaffected; it's altitude-independent).
      - NaN sound_vel_ms for a ping → falls back to 1500 m/s with warning.
      - NaN pitch_deg for a ping → taken as 0 with warning.
      - Water-column samples (slant <= altitude) → NaN in ground_range_m
        and (by default) vertical_offset_m and depth_per_sample_m.
        across_track_m and along_track_m use the pitched-fan limit
        slant <= altitude / cos(pitch), so on pitched pings a few samples
        just past the altitude have a ground range but no across/along
        position.
    """
    ds.sidescan.validate()

    n_pings = ds.sizes["ping"]
    n_channels = ds.sizes["channel"]
    n_samples = ds.sizes["sample"]

    # ----- per-ping inputs -----
    sv = ds["sound_vel_ms"].values.astype(np.float64)
    n_invalid_sv = int((np.isnan(sv) | (sv <= 0)).sum())
    sample_interval_per_ping = sample_interval_m(ds)

    altitude = ds["altitude_m"].values.astype(np.float64)
    n_no_alt = int(np.sum(~np.isfinite(altitude)))
    n_zero_alt = int(np.sum(altitude <= 0))
    if n_no_alt or n_zero_alt:
        logger.warning(
            "%d of %d pings have NaN altitude_m (their ground range is NaN) and %d have "
            "altitude_m <= 0 (ground range = slant range); the 'bottom_track' step can fill "
            "them from the first bottom return",
            n_no_alt, altitude.size, n_zero_alt,
        )
    pitch_deg = ds["pitch_deg"].values.astype(np.float64)
    n_nan_pitch = int(np.sum(~np.isfinite(pitch_deg)))
    if n_nan_pitch:
        logger.warning(
            "%d of %d pings have NaN pitch_deg; their pitch is taken as 0 "
            "(no along-track offset)",
            n_nan_pitch, pitch_deg.size,
        )
        pitch_deg = np.where(np.isfinite(pitch_deg), pitch_deg, 0.0)
    depth = ds["depth_m"].values.astype(np.float64)
    pitch_rad = np.deg2rad(pitch_deg)

    # ----- slant range (attitude-invariant) -----
    sample_idx = sample_index(ds)
    slant_per_ping = sample_interval_per_ping[:, None] * sample_idx[None, :]
    slant_range = np.broadcast_to(
        slant_per_ping[:, None, :],
        (n_pings, n_channels, n_samples),
    ).astype(np.float32)

    # ----- ground range (flat bottom, no pitch correction) -----
    altitude_b = altitude[:, None, None]
    with np.errstate(invalid="ignore"):
        diff = slant_range.astype(np.float64) ** 2 - altitude_b ** 2
        ground_range = np.where(
            (diff > 0) & ~np.isnan(altitude_b),
            np.sqrt(np.maximum(diff, 0.0)),
            np.nan,
        ).astype(np.float32)

    # NaN mask used for masking the dependent variables
    valid_mask = ~np.isnan(ground_range)

    # ----- across-track and along-track (pitched fan beam, flat bottom) -----
    # Body-down in the world frame is (sin θ, 0, cos θ) for nose-up θ, so
    # the fan plane meets the seafloor at along = h·tan θ, and the slant
    # distance to that line is h / cos θ.
    cos_pitch = np.cos(pitch_rad)[:, None, None]
    tan_pitch = np.tan(pitch_rad)[:, None, None]

    # Channel sign: port = -1, starboard = +1.  Channel index 0 = port.
    channel_sign = np.array([-1.0, 1.0], dtype=np.float64)[None, :, None]

    with np.errstate(invalid="ignore"):
        fan_diff = (
            slant_range.astype(np.float64) ** 2 - (altitude_b / cos_pitch) ** 2
        )
        seafloor = (fan_diff > 0) & ~np.isnan(altitude_b)
        across_track = np.where(
            seafloor,
            channel_sign * np.sqrt(np.maximum(fan_diff, 0.0)),
            np.nan,
        ).astype(np.float32)
        along_track = np.where(
            seafloor, altitude_b * tan_pitch, np.nan,
        ).astype(np.float32)

    # ----- vertical offset and depth per sample -----
    # For flat bottom: vertical_offset_m == altitude_m, broadcast.
    # For depth_per_sample_m: == depth_m + altitude_m, broadcast.
    vertical_per_ping = altitude.astype(np.float32)
    depth_per_ping = (depth + altitude).astype(np.float32)

    vertical_offset = np.broadcast_to(
        vertical_per_ping[:, None, None],
        (n_pings, n_channels, n_samples),
    ).astype(np.float32).copy()  # copy to allow masking

    depth_per_sample = np.broadcast_to(
        depth_per_ping[:, None, None],
        (n_pings, n_channels, n_samples),
    ).astype(np.float32).copy()

    if not expose_unmasked_z:
        vertical_offset[~valid_mask] = np.nan
        depth_per_sample[~valid_mask] = np.nan

    # ----- assemble output -----
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)

    out = out.assign(
        slant_range_m=(("ping", "channel", "sample"), slant_range),
        ground_range_m=(("ping", "channel", "sample"), ground_range),
        across_track_m=(("ping", "channel", "sample"), across_track),
        along_track_m=(("ping", "channel", "sample"), along_track.copy()),
        vertical_offset_m=(("ping", "channel", "sample"), vertical_offset),
        depth_per_sample_m=(("ping", "channel", "sample"), depth_per_sample),
    )

    # Variable-level attrs
    out["slant_range_m"].attrs = {
        "units": "m",
        "long_name": "slant range (transducer to sample)",
        "description": "Distance from transducer along the slant path. "
                       "Attitude-invariant.",
    }
    out["ground_range_m"].attrs = {
        "units": "m",
        "long_name": "ground range (across the seafloor)",
        "description": "Horizontal distance from the nadir point "
                       "assuming a flat bottom, sqrt(slant^2 - altitude^2). "
                       "Unsigned; the same with or without pitch.",
    }
    out["across_track_m"].attrs = {
        "units": "m",
        "long_name": "across-track distance from platform, signed",
        "description": "Y component in platform frame. Negative = port, "
                       "positive = starboard. Pitched fan beam: "
                       "sqrt(slant^2 - (altitude / cos(pitch))^2).",
    }
    out["along_track_m"].attrs = {
        "units": "m",
        "long_name": "along-track offset from nadir, signed",
        "description": "X component in platform frame. Positive = forward "
                       "of nadir. Pitched fan beam: altitude * tan(pitch), "
                       "the same for every seafloor sample of a ping.",
    }
    out["vertical_offset_m"].attrs = {
        "units": "m",
        "long_name": "vertical distance from platform to sample",
        "description": "Z component in platform frame. For flat bottom, "
                       "equals altitude_m broadcast across samples.",
    }
    out["depth_per_sample_m"].attrs = {
        "units": "m",
        "long_name": "absolute water depth at sample",
        "description": "depth_m + altitude_m broadcast across samples. "
                       "Absolute depth in the water column.",
    }

    # ----- update processing history -----
    out = append_history(
        out,
        STEP_GEOMETRIC,
        {
            "expose_unmasked_z": expose_unmasked_z,
            "default_sound_vel_ms": _DEFAULT_SOUND_VEL_MS,
            "pitch_model": PITCH_MODEL,
            "n_nan_pitch_as_zero": n_nan_pitch,
        },
        function="sidescanpy.processing.geometric.compute_geometric_ranges",
        version=__version__,
    )

    logger.debug(
        "compute_geometric_ranges: added 6 per-sample variables for %d "
        "pings × %d channels × %d samples (%d pings with invalid sound velocity)",
        n_pings, n_channels, n_samples, n_invalid_sv,
    )

    return out
