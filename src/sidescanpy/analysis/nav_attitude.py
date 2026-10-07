"""
Navigation/attitude diagnostic -- independent of the 5-stage TVG/
geometry investigation pipeline (`analysis/investigation.py`). This is a
data-quality check on the aux-stream fields themselves (heading/pitch/
roll/altitude/depth/heave/position), not on backscatter.

Motivated by a heading/yaw value that falls outside both the [0,360)
compass-bearing and [-180,180) signed conventions somewhere upstream of
this library (this library's own sign convention -- see CLAUDE.md's
"Sign Conventions" -- is compass bearing, CW from N): a human needs to
look at the actual heading/attitude/position trace for the affected
ping range to tell a real sensor/conversion problem apart from an
ordinary survey-line turn (the field briefly leaves either bucket during
a real maneuver, which isn't itself an error).

2026 rshomberg@uri.edu
"""

import numpy as np
import xarray as xr

# TODO: impplement logging

# FIXME: trying to get rid of the Stages.
from sidescanpy.analysis.investigation import StageResult

NAV_ATTITUDE_VARS: tuple[str, ...] = (
    "heading_deg", "pitch_deg", "roll_deg", "altitude_m", "depth_m", "heave_m",
)


def nav_attitude_summary(ds: xr.Dataset, *, heading_var: str = "heading_deg") -> StageResult:
    """Per-field min/max/mean/NaN-count for every `NAV_ATTITUDE_VARS`
    variable present in `ds`, plus a heading-convention check mirroring
    the [0,360)-vs-[-180,180) bucket test an upstream converter's own
    plausibility checks might run on a raw yaw/heading field -- run here
    against `heading_var` (this library's own, already-converted
    `heading_deg` by default) so a caller can tell whether an
    upstream-reported out-of-range finding survived conversion into this
    library's data, or was already resolved before it got here.

    `classification` is one of:
    - ``"compass_bearing"`` -- every finite value in [0, 360)
    - ``"signed"`` -- every finite value in [-180, 180)
    - ``"ambiguous"`` -- neither convention holds for every value (a
      mix, or genuinely out-of-range values); see
      `heading_frac_out_of_range` for how much of the range that
      applies to.
    - ``None`` -- `heading_var` isn't present in `ds`, or has no finite
      values.
    """
    metrics: dict = {}
    notes: list[str] = []

    for var in NAV_ATTITUDE_VARS:
        if var not in ds:
            continue
        values = np.asarray(ds[var].values, dtype=np.float64)
        n_nan = int(np.isnan(values).sum())
        metrics[f"{var}_n_nan"] = n_nan
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            metrics[f"{var}_min"] = None
            metrics[f"{var}_max"] = None
            metrics[f"{var}_mean"] = None
            notes.append(f"{var}: no finite values.")
            continue
        metrics[f"{var}_min"] = float(np.min(finite))
        metrics[f"{var}_max"] = float(np.max(finite))
        metrics[f"{var}_mean"] = float(np.mean(finite))
        if n_nan:
            notes.append(f"{var}: {n_nan} NaN value(s) ({n_nan / values.size:.1%}).")

    classification = None
    if heading_var in ds:
        heading = np.asarray(ds[heading_var].values, dtype=np.float64)
        finite = heading[np.isfinite(heading)]
        if finite.size:
            in_compass = (finite >= 0.0) & (finite < 360.0)
            in_signed = (finite >= -180.0) & (finite < 180.0)
            frac_compass = float(np.mean(in_compass))
            frac_signed = float(np.mean(in_signed))
            frac_out = float(np.mean(~(in_compass | in_signed)))
            metrics["heading_frac_compass_range"] = frac_compass
            metrics["heading_frac_signed_range"] = frac_signed
            metrics["heading_frac_out_of_range"] = frac_out

            if frac_compass == 1.0:
                classification = "compass_bearing"
            elif frac_signed == 1.0:
                classification = "signed"
            else:
                classification = "ambiguous"
                notes.append(
                    f"{heading_var}: not entirely within either the [0,360) compass-bearing or "
                    f"[-180,180) signed convention -- {frac_out:.1%} of values fall outside both; "
                    "check the plotted trace for a real sensor/conversion problem vs. an ordinary "
                    "survey-line turn before trusting this range's heading."
                )

    return StageResult(stage="nav_attitude", metrics=metrics, classification=classification, notes=notes)
