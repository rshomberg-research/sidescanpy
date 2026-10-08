"""
Navigation and attitude data-quality summary.

:func:`summarize_navigation` reports per-field statistics for the
per-ping aux fields (heading, pitch, roll, altitude, depth, heave) and
checks which heading convention the data follows. The library's own
convention is compass bearing, clockwise from north, in [0, 360).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin

logger = logging.getLogger(__name__)

NAV_ATTITUDE_VARS: tuple[str, ...] = (
    "heading_deg", "pitch_deg", "roll_deg", "altitude_m", "depth_m", "heave_m",
)


@dataclass
class FieldStats(ResultMixin):
    """NaN count and finite min/max/mean of one per-ping field."""

    n_nan: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


@dataclass
class NavigationSummary(ResultMixin):
    """Per-field statistics plus the heading-convention check.

    ``heading_convention`` is ``"compass_bearing"`` (all finite values in
    [0, 360)), ``"signed"`` (all in [-180, 180)), ``"ambiguous"``
    (neither holds for every value), or ``None`` (no heading, or no
    finite heading values). ``field_stats`` is keyed by variable name and
    covers the ``NAV_ATTITUDE_VARS`` present.
    """

    heading_convention: str | None = None
    field_stats: dict[str, FieldStats] = field(default_factory=dict)
    heading_frac_compass_range: float | None = None
    heading_frac_signed_range: float | None = None
    heading_frac_out_of_range: float | None = None
    notes: list[str] = field(default_factory=list)


def summarize_navigation(ds: xr.Dataset, *, heading_var: str = "heading_deg") -> NavigationSummary:
    """Per-field statistics for every ``NAV_ATTITUDE_VARS`` entry in ``ds``,
    plus the fraction of ``heading_var`` values in each heading convention.

    A value outside both conventions is not necessarily an error (a
    trace through a turn can briefly leave one), so the plotted trace is
    the place to judge it.
    """
    result = NavigationSummary()
    notes = result.notes

    for var in NAV_ATTITUDE_VARS:
        if var not in ds:
            continue
        values = np.asarray(ds[var].values, dtype=np.float64)
        n_nan = int(np.isnan(values).sum())
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            result.field_stats[var] = FieldStats(n_nan=n_nan)
            notes.append(f"{var}: no finite values.")
            continue
        result.field_stats[var] = FieldStats(
            n_nan=n_nan,
            minimum=float(np.min(finite)),
            maximum=float(np.max(finite)),
            mean=float(np.mean(finite)),
        )
        if n_nan:
            notes.append(f"{var}: {n_nan} NaN value(s) ({n_nan / values.size:.1%}).")

    if heading_var in ds:
        heading = np.asarray(ds[heading_var].values, dtype=np.float64)
        finite = heading[np.isfinite(heading)]
        if finite.size:
            in_compass = (finite >= 0.0) & (finite < 360.0)
            in_signed = (finite >= -180.0) & (finite < 180.0)
            frac_compass = float(np.mean(in_compass))
            frac_signed = float(np.mean(in_signed))
            frac_out = float(np.mean(~(in_compass | in_signed)))
            result.heading_frac_compass_range = frac_compass
            result.heading_frac_signed_range = frac_signed
            result.heading_frac_out_of_range = frac_out

            if frac_compass == 1.0:
                result.heading_convention = "compass_bearing"
            elif frac_signed == 1.0:
                result.heading_convention = "signed"
            else:
                result.heading_convention = "ambiguous"
                notes.append(
                    f"{heading_var}: not entirely within either the [0,360) compass-bearing or "
                    f"[-180,180) signed convention -- {frac_out:.1%} of values fall outside both; "
                    "check the plotted trace for a real sensor/conversion problem vs. an ordinary "
                    "survey-line turn before trusting this range's heading."
                )

    return result
