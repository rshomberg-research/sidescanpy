"""
Beam geometry from measured boundaries.

:func:`estimate_beam_geometry` triangulates the sidescan beam's near and
far edges from data already measured elsewhere:

- near edge, per (ping, channel): ``arccos(altitude / slant_range)`` at
  the first bottom return (:func:`~sidescanpy.analysis.nadir.analyze_nadir`);
- far edge, survey-wide: ``arctan(usable_range / altitude)`` with the
  usable range from :func:`~sidescanpy.analysis.usable_range.estimate_usable_range`.

A wrong ``altitude_m`` shifts both edges together, so this can't by
itself separate a tilted beam from bad altitude.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin

logger = logging.getLogger(__name__)

_SIDES = ((0, "port"), (1, "starboard"))


@dataclass
class BeamGeometryResult(ResultMixin):
    """Near/far beam-edge angles (degrees from vertical).

    ``confidence`` reflects the cross-ping near-edge measurement:
    ``"determined"``, ``"ambiguous"`` (too much ping-to-ping spread),
    ``"undetermined"`` (too few valid pings), or ``None`` when the
    reference ping's altitude is unusable. ``near_ray`` / ``far_ray``
    hold the single ``ping_idx`` geometry per side (``"port"``,
    ``"starboard"``) for plotting. Far-edge values are ``None`` without
    a usable range.
    """

    confidence: str | None = None
    ping_idx: int | None = None
    altitude_m: float | None = None
    near_ray: dict[str, dict] = field(default_factory=dict)
    far_ray: dict[str, dict] = field(default_factory=dict)
    near_angle_deg_median: float | None = None
    near_angle_deg_std: float | None = None
    near_angle_valid_fraction: float | None = None
    far_angle_deg: float | None = None
    beam_center_angle_deg: float | None = None
    beam_width_deg: float | None = None
    usable_range_m: float | None = None
    notes: list[str] = field(default_factory=list)


def estimate_beam_geometry(
    ds: xr.Dataset,
    first_bottom_return_sample: np.ndarray | None,
    usable_range_m: float | None,
    range_var: str = "across_track_m",
    slant_range_var: str = "slant_range_m",
    *,
    ping_idx: int | None = None,
    near_angle_max_std_deg: float = 15.0,
    min_valid_fraction: float = 0.2,
) -> BeamGeometryResult:
    """Estimate the beam's near-edge, far-edge, centre angle and width.

    Run nadir-noise blanking first on hardware that doesn't blank it
    itself, or the near edge measures the noise burst rather than the
    beam (reported as ``"undetermined"``).

    Parameters
    ----------
    ds : xr.Dataset
        Needs ``range_var``, ``slant_range_var`` and ``altitude_m``.
    first_bottom_return_sample : np.ndarray or None
        ``(ping, channel)`` first-bottom-return sample indices
        (``NadirResult.first_bottom_return_sample``).
    usable_range_m : float or None
        Survey-aggregate usable range for the far edge.
    range_var : str, default "across_track_m"
        Signed cross-track distance (port negative), for the diagram.
    slant_range_var : str, default "slant_range_m"
        Range used for the near-angle back-calculation.
    ping_idx : int, optional
        Ping for the ``near_ray`` / ``far_ray`` diagram geometry
        (default: middle ping). Doesn't affect the cross-ping statistics.
    near_angle_max_std_deg : float, default 15.0
        Cross-ping near-angle std above this → ``"ambiguous"``.
    min_valid_fraction : float, default 0.2
        Minimum fraction of (ping, channel) pairs with a valid near angle
        (finite altitude, ``slant_range >= altitude``, detected boundary);
        below it → ``"undetermined"``.
    """
    for required in (range_var, slant_range_var, "altitude_m"):
        if required not in ds:
            raise ValueError(f"estimate_beam_geometry: {required!r} not found in dataset")

    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2

    altitude = float(ds["altitude_m"].isel(ping=ping_idx).values)
    notes: list[str] = []
    if not np.isfinite(altitude) or altitude <= 0:
        notes.append(f"altitude_m is non-finite/non-positive ({altitude}) for ping {ping_idx} — geometry undefined.")
        return BeamGeometryResult(ping_idx=ping_idx, altitude_m=altitude, notes=notes)

    first_bottom = first_bottom_return_sample
    near_ray: dict[str, dict] = {}
    for ch, label in _SIDES:
        entry = {
            "observed_sample": None, "observed_slant_range_m": None,
            "observed_across_track_m": None, "angle_from_vertical_deg": None,
            "angle_from_horizontal_deg": None,
        }
        b_val = first_bottom[ping_idx, ch] if first_bottom is not None and ping_idx < first_bottom.shape[0] else np.nan
        if np.isfinite(b_val):
            b = int(round(b_val))
            n_samples = ds.sizes["sample"]
            if b < n_samples:
                slant = float(ds[slant_range_var].isel(ping=ping_idx, channel=ch, sample=b).values)
                across = float(ds[range_var].isel(ping=ping_idx, channel=ch, sample=b).values)
                entry["observed_sample"] = b
                entry["observed_slant_range_m"] = slant if np.isfinite(slant) else None
                entry["observed_across_track_m"] = across if np.isfinite(across) else None
                if np.isfinite(slant) and slant >= altitude:
                    cos_angle = np.clip(altitude / slant, -1.0, 1.0)
                    angle_from_vertical = float(np.degrees(np.arccos(cos_angle)))
                    entry["angle_from_vertical_deg"] = angle_from_vertical
                    entry["angle_from_horizontal_deg"] = 90.0 - angle_from_vertical
                    if angle_from_vertical > 88.0:
                        notes.append(
                            f"{label}: near-edge angle ({angle_from_vertical:.1f} deg from vertical) "
                            f"is within 2 deg of horizontal — physically implausible; altitude_m or the "
                            f"detected first bottom return is more likely wrong for this ping."
                        )
                elif np.isfinite(slant):
                    notes.append(
                        f"{label}: observed slant range ({slant:.2f}m) is less than altitude_m "
                        f"({altitude:.2f}m) at the detected first bottom return — geometrically "
                        f"impossible for any beam angle; altitude_m or the detection is unreliable "
                        f"for this ping/channel."
                    )
        near_ray[label] = entry

    far_ray: dict[str, dict] = {}
    for _ch, label in _SIDES:
        entry = {
            "predicted_across_track_m": None, "predicted_slant_range_m": None,
            "angle_from_vertical_deg": None, "angle_from_horizontal_deg": None,
        }
        if usable_range_m is not None and usable_range_m > 0:
            # One survey-wide magnitude; apply across_track_m's sign
            # convention (port negative) so both sides get a far ray.
            sign = -1.0 if label == "port" else 1.0
            angle_from_vertical = float(np.degrees(np.arctan2(usable_range_m, altitude)))
            entry["predicted_across_track_m"] = sign * float(usable_range_m)
            entry["predicted_slant_range_m"] = float(np.hypot(usable_range_m, altitude))
            entry["angle_from_vertical_deg"] = angle_from_vertical
            entry["angle_from_horizontal_deg"] = 90.0 - angle_from_vertical
        far_ray[label] = entry
    if usable_range_m is None:
        notes.append(
            "usable_range_m is None (SNR never reached the far-range threshold) -- the far "
            "edge, beam width, and beam center angle can't be computed."
        )

    near_angle_deg_median = None
    near_angle_deg_std = None
    near_angle_valid_fraction = 0.0
    confidence = "undetermined"

    if first_bottom is not None:
        n_p, n_c = first_bottom.shape
        n_samples = ds.sizes["sample"]
        altitude_all = ds["altitude_m"].values.astype(np.float64)  # (ping,)
        altitude_valid = np.isfinite(altitude_all) & (altitude_all > 0)

        boundary_valid = np.isfinite(first_bottom) & altitude_valid[:, None]
        b_idx = np.where(boundary_valid, np.round(first_bottom), 0).astype(int)
        b_idx = np.clip(b_idx, 0, n_samples - 1)

        slant_arr = ds[slant_range_var].values  # (ping, channel, sample)
        p_grid, c_grid = np.meshgrid(np.arange(n_p), np.arange(n_c), indexing="ij")
        slant_at_boundary = slant_arr[p_grid, c_grid, b_idx]  # (ping, channel)

        altitude_bc = np.broadcast_to(altitude_all[:, None], (n_p, n_c))
        geom_valid = boundary_valid & np.isfinite(slant_at_boundary) & (slant_at_boundary >= altitude_bc)

        cos_angle = np.full((n_p, n_c), np.nan)
        np.divide(altitude_bc, slant_at_boundary, out=cos_angle, where=geom_valid)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_from_vertical_all = np.degrees(np.arccos(cos_angle))
        angle_from_vertical_all = np.where(geom_valid, angle_from_vertical_all, np.nan)

        finite_angles = angle_from_vertical_all[np.isfinite(angle_from_vertical_all)]
        near_angle_valid_fraction = float(finite_angles.size / angle_from_vertical_all.size)
        if finite_angles.size > 0:
            near_angle_deg_median = float(np.median(finite_angles))
            near_angle_deg_std = float(np.std(finite_angles))

        if near_angle_deg_median is None or near_angle_valid_fraction < min_valid_fraction:
            confidence = "undetermined"
            notes.append(
                f"Only {near_angle_valid_fraction:.0%} of (ping, channel) pairs yielded a valid "
                f"near-edge angle -- too few for a confident measurement (need >= "
                f"{min_valid_fraction:.0%}). On hardware that doesn't blank its own nadir noise, "
                f"check that blank_nadir_noise ran first."
            )
        elif near_angle_deg_std is not None and near_angle_deg_std <= near_angle_max_std_deg:
            confidence = "determined"
            notes.append(
                f"Cross-ping near-edge angle: median {near_angle_deg_median:.1f} deg from vertical "
                f"(std {near_angle_deg_std:.1f} deg, {near_angle_valid_fraction:.0%} of "
                f"pings/channels valid) -- consistent measurement."
            )
        else:
            confidence = "ambiguous"
            notes.append(
                f"Cross-ping near-edge angle is inconsistent (median {near_angle_deg_median:.1f} "
                f"deg, std {near_angle_deg_std:.1f} deg across {near_angle_valid_fraction:.0%} "
                f"valid pings/channels) -- not a confident single measurement."
            )

    far_angle_deg = None
    beam_center_angle_deg = None
    beam_width_deg = None
    if usable_range_m is not None and usable_range_m > 0:
        altitude_finite = ds["altitude_m"].values[np.isfinite(ds["altitude_m"].values)]
        altitude_median = float(np.median(altitude_finite)) if altitude_finite.size else altitude
        far_angle_deg = float(np.degrees(np.arctan2(usable_range_m, altitude_median)))
        if near_angle_deg_median is not None:
            beam_center_angle_deg = (near_angle_deg_median + far_angle_deg) / 2.0
            beam_width_deg = far_angle_deg - near_angle_deg_median

    return BeamGeometryResult(
        confidence=confidence,
        ping_idx=ping_idx,
        altitude_m=altitude,
        near_ray=near_ray,
        far_ray=far_ray,
        near_angle_deg_median=near_angle_deg_median,
        near_angle_deg_std=near_angle_deg_std,
        near_angle_valid_fraction=near_angle_valid_fraction,
        far_angle_deg=far_angle_deg,
        beam_center_angle_deg=beam_center_angle_deg,
        beam_width_deg=beam_width_deg,
        usable_range_m=usable_range_m,
        notes=notes,
    )
