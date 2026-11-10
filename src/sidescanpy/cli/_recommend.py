"""
Recommended, hand-editable YAML config for ``sidescan-analyze recommend-config``.

:func:`run_analyses` runs the analyses the recommendation draws on (a
small CLI workflow over :mod:`sidescanpy.analysis`);
:func:`render_recommended_config` turns their results into YAML. Every emitted key has the
evidence for it in a comment directly above; a finding with no
downstream config key is reported as a comment only. Hand-formatted
(not ``yaml.dump``) so comments can sit next to keys. Keys match
``sidescan-correct`` / ``sidescan-export`` flag dests, so the output
works with ``--config`` as-is.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from sidescanpy.analysis.beam import BeamGeometryResult, estimate_beam_geometry
from sidescanpy.analysis.characterize import (
    UnitsResult,
    ValueSummary,
    characterize_values,
    infer_units,
)
from sidescanpy.analysis.geometry import PingGeometry, measure_ping_geometry
from sidescanpy.analysis.nadir import NadirResult, analyze_nadir
from sidescanpy.analysis.tvg_fit import TVGCheckResult, TVGFitResult, check_tvg, fit_tvg_models
from sidescanpy.analysis.usable_range import UsableRangeResult, estimate_usable_range
from sidescanpy.grids import recommend_resolution

logger = logging.getLogger(__name__)


def run_analyses(
    ds: xr.Dataset, *, value_var: str, nadir_n_samples: int = 50, bin_width_m: float = 1.0, snr_threshold: float = 3.0,
) -> dict:
    """Every result :func:`render_recommended_config` takes, keyed by its
    argument name, from one channel with geometric ranges.

    The TVG fit is :func:`~sidescanpy.analysis.fit_tvg_models` on slant
    range (``bin_width_m``), with :func:`~sidescanpy.analysis.check_tvg`
    of its best model (``tvg_check``, ``None`` without a fit); ``beam`` is
    ``None`` when it cannot be computed.
    """
    values = characterize_values(ds, value_var=value_var)
    nadir = analyze_nadir(ds, value_var=value_var, nadir_n_samples=nadir_n_samples)
    usable_range = estimate_usable_range(
        ds, value_var=value_var, range_var="slant_range_m", bin_width_m=bin_width_m, snr_threshold=snr_threshold,
    )
    try:
        beam = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable_range.usable_range_m)
    except (ValueError, KeyError) as exc:
        logger.warning(f"beam-geometry could not be computed ({exc}) -- omitting from the recommendation")
        beam = None
    tvg_fit = fit_tvg_models(ds, value_var=value_var, range_var="slant_range_m", bin_width_m=bin_width_m)
    tvg_check = None
    if tvg_fit.best is not None:
        tvg_check = check_tvg(ds, tvg_fit.tvg_params, tvg_fit.window_m, value_var=value_var, bin_width_m=bin_width_m)
    return dict(
        values=values, units=infer_units(values, tvg_fit), nadir=nadir, beam=beam, usable_range=usable_range,
        tvg_fit=tvg_fit, tvg_check=tvg_check, geometry=measure_ping_geometry(ds),
    )


def _fmt(value, spec: str = ".2f") -> str:
    if isinstance(value, (int, float)) and value is not None and np.isfinite(value):
        return format(value, spec)
    return str(value)


def _tvg_lines(fit: TVGFitResult | None, check: TVGCheckResult | None) -> list[str]:
    """The TVG block: the best model's terms as keys, the evidence as comments."""
    if fit is None or fit.best_model is None:
        notes = "; ".join(fit.notes) if fit is not None else "no fit run"
        return [
            "# TVG: no model could be fitted to this section -- set spreading_loss_db /",
            "#   absorption_db_per_m by hand, or rerun `sidescan-analyze tvg-fit --window-m MIN MAX`.",
            f"#   ({notes})",
        ]
    best = fit.best_model
    window = f"{fit.window_m[0]:.1f}-{fit.window_m[1]:.1f} m"
    lines = [
        f"# TVG: best model {best.name}, {best.rms_db:.2f} dB RMS over {window} "
        f"(median level, {fit.bin_width_m:g} m bins).",
    ]
    if fit.runner_up_model is not None:
        lines.append(f"#   Runner-up {fit.runner_up} ({fit.runner_up_model.rms_db:.2f} dB); free n "
                     f"{_fmt(fit.free_n, '.1f')} with absorption, {_fmt(fit.free_n_no_absorption, '.1f')} without.")
    if check is not None and check.spread_db is not None:
        lines.append(f"#   Applied, it leaves {check.spread_db:.2f} dB spread and {check.trend_db_per_10m:+.2f} dB "
                     "per 10 m across the window.")
    lines.append(f"#   The recorder appears to have removed: {fit.tvg_status} "
                 "(none / spreading_only / absorption_only / full).")
    if fit.window_source == "end_of_record":
        lines.append("#   No noise knee: the level does not fall to a floor, so the fit runs to the end")
        lines.append("#   of the record. Typical of data the recorder has already levelled.")
    if fit.low_contrast:
        lines.append(f"#   Low contrast ({fit.contrast_db:.1f} dB from peak to floor): little range")
        lines.append("#   dependence left, or little seabed signal.")
    if fit.sides_disagree:
        lines.append("#   Port and starboard disagree (see metrics); the pooled fit may suit neither side.")
    lines.append("#   Over short windows 20 log R + abs, 30 log R and 40 log R differ by tenths of a dB;")
    lines.append("#   for a survey, compare several lines before fixing one model.")
    lines.append(f"spreading_loss_db: {best.spreading_loss_db:g}")
    lines.append(f"absorption_db_per_m: {best.absorption_db_per_m:.4f}")
    return lines


def render_recommended_config(
    *, source_name: str, freq_key: str, ping_range: tuple, value_var: str,
    values: ValueSummary, nadir: NadirResult, beam: BeamGeometryResult | None,
    usable_range: UsableRangeResult, tvg_fit: TVGFitResult | None, units: UnitsResult,
    geometry: PingGeometry, tvg_check: TVGCheckResult | None = None,
) -> str:
    """YAML text recommending ``sidescan-correct`` / ``sidescan-export chip``
    settings for one section, with the evidence for each as comments."""
    start, stop = ping_range
    lines: list[str] = [
        f"# Recommended by `sidescan-analyze recommend-config` against {source_name} / {freq_key}",
        f"# ({value_var}, pings {start}:{stop if stop is not None else ''}).",
        "# Review before use -- a starting point, not validated ground truth. Copy into",
        "# configs/<survey>.yaml, or point --config directly at this file, and",
        "# edit as needed.",
        "",
        "# ---- sidescan-correct ----",
    ]

    lines.extend(_tvg_lines(tvg_fit, tvg_check))
    lines.append("")

    lines.append(f"# nadir: {nadir.nadir_type} (altitude_sample_median="
                  f"{_fmt(nadir.altitude_sample_median)}, first_bottom_return_sample_median="
                  f"{_fmt(nadir.first_bottom_return_sample_median)})")
    lines.append("#   No sidescan-correct/sidescan-export flag currently consumes this directly --")
    lines.append("#   informational only.")
    lines.append("")

    if beam is not None:
        lines.append(f"# beam_geometry: {beam.confidence} (near_angle_deg="
                      f"{_fmt(beam.near_angle_deg_median)}, far_angle_deg="
                      f"{_fmt(beam.far_angle_deg)}, beam_width_deg="
                      f"{_fmt(beam.beam_width_deg)})")
        lines.append("#   Informational only -- no downstream config flag yet.")
        lines.append("")

    usable_range_m = usable_range.usable_range_m
    how = "noise knee of the median level" if usable_range.method == "knee" else "SNR against the far-range noise floor"
    lines.append(f"# usable_range_m: {_fmt(usable_range_m)} ({how})")
    lines.append("")

    lines.append(f"# units: {units.units} (dtype={values.dtype})")
    lines.append("#   Informational only.")
    lines.append("")

    lines.append("# ---- sidescan-export chip ----")
    spacing = geometry.ping_spacing_m_median
    swath = geometry.swath_width_m
    sample_interval = geometry.sample_interval_m
    if spacing is not None:
        lines.append(f"# Local median ping spacing = {_fmt(spacing, '.4f')} m, swath width = {_fmt(swath)} m")
        lines.append("# (measured over this section's own real navigation/geometry).")
        if sample_interval is not None:
            # Oversample past the finest native axis and let the
            # geometric resample interpolate, rather than stopping at the
            # coarser axis. Half the finer spacing keeps every real
            # sample on that axis in its own pixel.
            finest = min(spacing, sample_interval)
            resolution_m = recommend_resolution(geometry)
            finest_axis = "ping spacing (along-track)" if finest == spacing else "sample interval (across-track)"
            coarse = max(spacing, sample_interval)
            lines.append(f"# Range-axis sample interval = {_fmt(sample_interval, '.4f')} m (across-track).")
            lines.append(
                f"# resolution_m = min(ping spacing, sample interval) / 2 = {_fmt(resolution_m, '.4f')} m "
                f"(half the {finest_axis}). Along the coarser axis most pixels are interpolated, "
                f"not measured: ~{coarse / resolution_m:.0f} output pixels per real sample there."
            )
            lines.append(
                f"# Coarsest-axis alternative (no interpolation, much larger footprint): "
                f"max(ping spacing, sample interval) = {_fmt(coarse, '.4f')} m."
            )
            lines.append(
                f"# Chip footprint = size_px x resolution_m, e.g. 300 px -> {_fmt(300 * resolution_m, '.2f')} m; "
                "set size_px (and box_size) to suit the targets."
            )
        else:
            resolution_m = recommend_resolution(geometry)
            lines.append(
                "# Range-axis sample interval unavailable (no slant_range_m) -- using half the ping "
                "spacing; the sample interval is usually finer, so check it and prefer half of that."
            )
        lines.append(f"resolution_m: {_fmt(resolution_m, '.4f')}")
    else:
        lines.append("# Could not measure ping spacing for this section (fewer than 2 pings, or no")
        lines.append("# along-track navigation) -- resolution_m needs a manually chosen value; see")
        lines.append("# sidescan-export chip --resolution-m's own help text.")
    lines.append("")

    return "\n".join(lines) + "\n"
