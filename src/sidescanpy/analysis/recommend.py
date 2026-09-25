"""
Evidence-based recommended config generation.

Real user request, direct quote: "I want to have the resolution and
crop size for the dataset to be determined by the config, not by auto
calculation... I would even be interested in analyze generating a
config file for each dataset, but with the opportunity for me to edit
it manually, and the analysis should provide evidence for every
decision it suggests." Third increment of the analyze/config redesign
this session (after ``sidescanpy.analysis.segmentation`` and the
``nadir``/``beam-geometry``/``usable-range``/``units`` per-tool
subcommands).

Two pieces:

- ``measure_ping_geometry`` — the actual missing measurement behind
  ``sidescan-export chip --resolution-m`` (see this package's own
  ``CHANGELOG.md``, "config-driven, not auto-calculated" entry): real
  local median ping spacing and swath width for an already-clean
  section, robust (median, percentile-bounded) rather than a plain
  mean/exact-span the way the *removed* per-target auto-calculation
  was -- this module never feeds a per-target runtime path, only a
  one-time-per-section recommendation a human reviews.
- ``render_recommended_config`` — turns a section's already-computed
  ``StageResult``s (Stage 1/2/2.5/3/4/5, whichever ran) plus that
  geometry measurement into one hand-formatted, heavily-commented YAML
  text: every emitted key states the evidence for it directly above,
  and a stage with no confident/actionable finding gets a comment, not
  a guessed key. Deliberately hand-formatted rather than built via
  ``yaml.dump`` -- a plain dump has no way to attach a comment to a
  specific key, and per-key evidence is the entire point here. Output
  keys match real ``sidescan-correct``/``sidescan-export`` CLI flag
  dests exactly, so the result is usable as-is via ``--config`` (the
  same YAML loader every other subcommand already shares, see
  ``cli/_config.py``) -- not a separate format the user has to
  translate by hand.

Only two stages currently map to a real downstream config key:
Stage 4 (TVG term selection -- ``no_spreading``/``no_absorption``, the
same boolean-disable convention already established in this family's
own ``configs/default.yaml``) and the new geometry measurement
(``resolution_m``). Every other stage (nadir, beam geometry, usable
range, units) has no config flag downstream that consumes its finding
today -- reported as evidence/context in a comment, not invented as a
key nothing reads.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sidescanpy.analysis.investigation import StageResult
from sidescanpy.viz.waterfall import along_track_distance_m


def measure_ping_geometry(ds: xr.Dataset, *, low_pct: float = 1.0, high_pct: float = 99.0) -> dict:
    """Real local median ping spacing (meters) + percentile-bounded
    across-track swath width (meters) for one already-clean section.

    Median ping-to-ping along-track distance, not a plain mean/whole-
    span average -- robust to a handful of outlier jumps, the same
    caution ``analysis.segmentation.detect_gaps``/``score_stability``
    already apply elsewhere in this family, and the direct fix for the
    real production bug that motivated exposing ``resolution_m`` as a
    config value in the first place (see this package's own
    ``CHANGELOG.md``). Swath width uses percentiles rather than the
    exact min/max, for the same reason: one far-flung outlier sample
    shouldn't set the number.

    Returns ``{"ping_spacing_m_median": float | None, "swath_width_m":
    float | None}`` -- ``None`` where the input doesn't support the
    measurement (too few pings, no ``across_track_m``) rather than a
    fabricated number.
    """
    result: dict = {"ping_spacing_m_median": None, "swath_width_m": None}
    n_pings = ds.sizes["ping"]
    if n_pings >= 2:
        dist = along_track_distance_m(ds)
        diffs = np.abs(np.diff(dist))
        nonzero = diffs[diffs > 0]
        if nonzero.size:
            result["ping_spacing_m_median"] = float(np.median(nonzero))
    if "across_track_m" in ds:
        across = np.asarray(ds["across_track_m"].values, dtype=np.float64)
        finite = across[np.isfinite(across)]
        if finite.size:
            lo, hi = np.percentile(finite, [low_pct, high_pct])
            result["swath_width_m"] = float(hi - lo)
    return result


_TVG_STATUS_NOTE = {
    "full": "Hardware already compensates both spreading and absorption.",
    "absorption_only": "Hardware already compensates spreading; absorption term still needed.",
    "spreading_only": "Hardware already compensates absorption; spreading term still needed.",
    "none": "Neither term appears compensated by hardware -- full physics needed "
            "(no override -- sidescan-correct's own defaults already apply full physics).",
}


def _fmt(value, spec: str = ".2f") -> str:
    if isinstance(value, (int, float)) and value is not None and np.isfinite(value):
        return format(value, spec)
    return str(value)


def render_recommended_config(
    *, source_name: str, freq_key: str, ping_range: tuple, value_var: str,
    stage1: StageResult, stage2: StageResult, stage2_5: StageResult | None,
    stage3: StageResult, stage4: StageResult | None, stage5: StageResult,
    geometry: dict,
) -> str:
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

    if stage4 is None:
        lines.append("# TVG status: could not be determined -- Stage 3 has no usable_range_m for")
        lines.append("#   this section (see usable_range_m note below). Pass --usable-range-m to")
        lines.append("#   `sidescan-analyze tvg-fit` manually, or investigate the far-range noise")
        lines.append("#   floor, before setting spreading/absorption overrides.")
    else:
        best = stage4.metrics.get("best_candidate")
        rmse = stage4.metrics.get("best_rmse_db")
        lines.append(f"# TVG status: {stage4.classification} (best_candidate={best}, RMSE={_fmt(rmse)} dB)")
        note = _TVG_STATUS_NOTE.get(stage4.classification)
        if note is not None:
            lines.append(f"#   {note}")
            if stage4.classification == "full":
                lines.append("no_spreading: true")
                lines.append("no_absorption: true")
            elif stage4.classification == "absorption_only":
                lines.append("no_spreading: true")
            elif stage4.classification == "spreading_only":
                lines.append("no_absorption: true")
        else:
            lines.append(f"#   {stage4.classification!r} -- not confident enough for an automatic")
            lines.append("#   recommendation (see metrics.json). Investigate manually (re-run tvg-fit")
            lines.append("#   with a different window) before setting spreading/absorption overrides.")
    lines.append("")

    lines.append(f"# nadir: {stage2.classification} (altitude_sample_median="
                  f"{_fmt(stage2.metrics.get('altitude_sample_median'))}, first_bottom_return_sample_median="
                  f"{_fmt(stage2.metrics.get('first_bottom_return_sample_median'))})")
    lines.append("#   No sidescan-correct/sidescan-export flag currently consumes this directly --")
    lines.append("#   informational only.")
    lines.append("")

    if stage2_5 is not None:
        lines.append(f"# beam_geometry: {stage2_5.classification} (near_angle_deg="
                      f"{_fmt(stage2_5.metrics.get('near_angle_deg_median'))}, far_angle_deg="
                      f"{_fmt(stage2_5.metrics.get('far_angle_deg'))}, beam_width_deg="
                      f"{_fmt(stage2_5.metrics.get('beam_width_deg'))})")
        lines.append("#   Informational only -- no downstream config flag yet.")
        lines.append("")

    usable_range_m = stage3.metrics.get("usable_range_m")
    lines.append(f"# usable_range_m: {_fmt(usable_range_m)} "
                 f"(Stage 3, SNR against the far-range noise floor)")
    lines.append("")

    lines.append(f"# units: {stage5.classification} (Stage 1 dtype={stage1.metrics.get('dtype')})")
    lines.append("#   Informational only.")
    lines.append("")

    lines.append("# ---- sidescan-export chip ----")
    spacing = geometry.get("ping_spacing_m_median")
    swath = geometry.get("swath_width_m")
    if spacing is not None:
        lines.append(f"# Local median ping spacing = {_fmt(spacing, '.4f')} m, swath width = {_fmt(swath)} m")
        lines.append("# (measured over this section's own real navigation/geometry).")
        lines.append(f"resolution_m: {_fmt(spacing, '.4f')}")
    else:
        lines.append("# Could not measure ping spacing for this section (fewer than 2 pings, or no")
        lines.append("# along-track navigation) -- resolution_m needs a manually chosen value; see")
        lines.append("# sidescan-export chip --resolution-m's own help text.")
    lines.append("")

    return "\n".join(lines) + "\n"
