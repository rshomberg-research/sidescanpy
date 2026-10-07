"""
Survey segmentation: find clean, homogeneous ping ranges within one
already-converted survey channel and split it into separate ``.ss.h5``
files, one per clean section.

Motivation (real, from a production report): every other analysis tool
in this package (nadir detection, beam geometry, usable range, TVG
decay fit) is only as trustworthy as the data it's pointed at.
Analyzing a file that silently contains a mid-survey gap, a hardware
setting change, a dropout run, or a deployment/turn transient produces
misleading aggregate numbers with nothing in the output signaling that
a blend of good and bad data happened -- the same class of problem this
package's own investigation stages have hit before (see
``analysis/investigation.py``'s ``channel`` parameter history for a
close analogue: pooling port and starboard was "invisible until you
looked for it"). This module exists to cut the bad data out *before*
any of that other analysis ever runs, not to explain a confusing result
after the fact.

Four independent detectors, each producing real evidence (not just a
boolean):

- ``detect_gaps`` -- a jump in along-track distance between consecutive
  pings far beyond the survey's own typical spacing (GPS dropout, a
  logging pause).
- ``detect_setting_changes`` -- a per-ping acquisition-parameter change
  (``valid_samples``, when present; ``sound_vel_ms`` otherwise/also).
  **Known limitation**: a raw sample-count/range change that an
  upstream format-specific converter already padded/reconciled away
  before writing this file is NOT detectable here -- see that
  function's own docstring.
- ``detect_garbage_runs`` -- a sustained run of degenerate (NaN/zero)
  backscatter across the ping axis, distinct from the nadir gap's own
  per-ping leading blanking run (``analysis.investigation.
  detect_nadir_noise_extent`` looks within one ping's sample axis; this
  looks across the ping axis instead).
- ``detect_unstable_spans`` -- a rolling attitude/altitude stability
  check (``score_stability``, )
  flags deployment/turn transients.

``find_segments`` combines all four into a list of ``Segment``s, each
labeled ``"clean"`` or ``"excluded"`` with human-readable ``reasons``.
``write_segments`` slices every frequency channel of the source survey
to each clean segment's ping range (via ``processing.slicing.
slice_by_pings`` -- no new slicing logic) and writes one independent
``.ss.h5`` file per segment.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing.slicing import slice_by_pings
from sidescanpy.viz.waterfall import along_track_distance_m

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    label: str
    start: int
    stop: int
    status: str  # "clean" | "excluded"
    reasons: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "label": self.label, "start": self.start, "stop": self.stop,
            "status": self.status, "reasons": self.reasons,
        }


def detect_gaps(ds: xr.Dataset, *, gap_factor: float = 8.0, time_gap_factor: float = 8.0) -> np.ndarray:
    """Per-ping boolean array; ``True`` at index ``i`` means a boundary
    between ping ``i-1`` and ``i``. Two independent checks, either can
    fire -- a real discontinuity (a paused/resumed recording, a GPS
    glitch) isn't guaranteed to show up in both signals at once:

    - a jump in cumulative along-track distance (``viz.waterfall.
      along_track_distance_m``, the same primitive ``cli/export.py``
      already uses for real ping-spacing math) ``gap_factor`` times the
      survey's own typical (median nonzero) spacing -- catches a
      position discontinuity (e.g. a GPS glitch) even if ping timing
      stayed regular.
    - a jump in the raw per-ping ``time`` coordinate itself
      ``time_gap_factor`` times the survey's own typical (median
      nonzero) ping interval -- catches a logging pause/resume even if
      the position happened to replay a stale-but-plausible value
      through it (so along-track distance alone wouldn't show it).

    Index 0 is never flagged -- nothing before it to jump from.
    """
    n_pings = ds.sizes["ping"]
    boundary = np.zeros(n_pings, dtype=bool)
    if n_pings < 3:
        return boundary

    dist = along_track_distance_m(ds)
    diffs = np.abs(np.diff(dist))
    nonzero = diffs[diffs > 0]
    if nonzero.size:
        typical = float(np.median(nonzero))
        if typical > 0:
            boundary[1:] |= diffs > (gap_factor * typical)

    if "time" in ds:
        t = ds["time"].values
        dt = np.abs(np.diff(t).astype("timedelta64[ms]").astype(np.float64))
        nonzero_dt = dt[dt > 0]
        if nonzero_dt.size:
            typical_dt = float(np.median(nonzero_dt))
            if typical_dt > 0:
                boundary[1:] |= dt > (time_gap_factor * typical_dt)

    return boundary


def detect_setting_changes(ds: xr.Dataset, *, sound_vel_rel_tol: float = 0.05) -> np.ndarray:
    """Per-ping boolean boundary flag for a likely acquisition-setting
    change mid-survey. Two independent checks, either can fire:

    - ``valid_samples`` (ping, channel), when present -- added to some
      formats' converters (e.g. Klein SDF) to preserve each ping's true
      recorded sample count through NaN-padding to a common array
      shape. A change here is a direct, unambiguous range/sample-count
      reconfiguration.
    - ``sound_vel_ms`` -- a real per-ping recorded value guaranteed by
      the core schema (``core/schema.py`` ``REQUIRED_VARS``, every
      format). A relative step beyond ``sound_vel_rel_tol`` between
      consecutive pings is treated as a likely environment/recording
      parameter change.

    **Known limitation, not silently assumed away**: a raw sample-count/
    range change that an upstream format-specific converter already
    padded/reconciled away before writing this file (true for any
    format without its own ``valid_samples``-equivalent, e.g. JSF
    and some vendor HDF5 formats) is NOT detectable here -- that
    information is already gone by the time this function ever sees
    the data. Fixing that needs preserving the boundary at conversion
    time (format-specific), not detecting it after the
    fact, which is this function's own job.
    """
    n_pings = ds.sizes["ping"]
    boundary = np.zeros(n_pings, dtype=bool)
    if n_pings < 2:
        return boundary

    if "valid_samples" in ds:
        vs = np.asarray(ds["valid_samples"].values)
        changed = np.any(vs[1:] != vs[:-1], axis=tuple(range(1, vs.ndim))) if vs.ndim > 1 else (vs[1:] != vs[:-1])
        boundary[1:] |= np.asarray(changed, dtype=bool)

    if "sound_vel_ms" in ds:
        sv = np.asarray(ds["sound_vel_ms"].values, dtype=np.float64)
        finite = np.isfinite(sv)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_change = np.abs(np.diff(sv)) / np.abs(sv[:-1])
        step = finite[1:] & finite[:-1] & (rel_change > sound_vel_rel_tol)
        boundary[1:] |= step

    return boundary


def detect_garbage_runs(
    ds: xr.Dataset, value_var: str = "backscatter_raw", *,
    min_run_pings: int = 20, degenerate_frac: float = 0.999,
) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` -- sustained (>= ``min_run_pings``) runs
    of degenerate (NaN or exactly zero, across >= ``degenerate_frac`` of
    every channel/sample) pings. Distinct from the nadir gap's own
    leading per-ping blanking run (``analysis.investigation.
    detect_nadir_noise_extent`` looks within one ping's own sample axis
    near its start); this looks across the ping axis for a mid-survey
    dropout instead -- same rolling + sustained-run-confirmation pattern
    already validated there, applied to a different axis.
    """
    n_pings = ds.sizes["ping"]
    if n_pings < min_run_pings or value_var not in ds:
        return []
    arr = np.asarray(ds[value_var].values)
    degenerate = np.isnan(arr) | (arr == 0)
    frac_degenerate = degenerate.reshape(n_pings, -1).mean(axis=1)
    is_bad = frac_degenerate >= degenerate_frac

    spans: list[tuple[int, int]] = []
    i = 0
    while i < n_pings:
        if is_bad[i]:
            j = i
            while j < n_pings and is_bad[j]:
                j += 1
            if j - i >= min_run_pings:
                spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def score_stability(altitude: np.ndarray, pitch: np.ndarray, start: int, span: int) -> float:
    """Lower = more stable = a better section candidate.
    (altitude coefficient-of-variation + pitch std, weighted 1:1 by raw
    magnitude -- not first-principles-derived, just a combination where
    both terms land in a similar range for a genuinely calm stretch, and
    blow up by 1-2 orders of magnitude for a real deployment/turn
    transient) -- moved here since it's now shared by both that script
    and ``detect_unstable_spans`` below, rather than kept as two
    independent copies.
    """
    alt_win = altitude[start:start + span]
    pitch_win = pitch[start:start + span]
    alt_win = alt_win[np.isfinite(alt_win)]
    pitch_win = pitch_win[np.isfinite(pitch_win)]
    if alt_win.size < 2 or pitch_win.size < 2:
        return float("inf")
    alt_mean = float(np.mean(alt_win))
    alt_cv = float(np.std(alt_win) / abs(alt_mean)) if abs(alt_mean) > 1e-6 else float("inf")
    return alt_cv + float(np.std(pitch_win))


def detect_unstable_spans(
    ds: xr.Dataset, *, window_pings: int = 100, stability_threshold: float = 1.0,
    min_run_pings: int = 50,
) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` -- non-overlapping ``window_pings``-wide
    windows scored by ``score_stability``; windows scoring above
    ``stability_threshold`` are "unstable" (e.g. a deployment/turn
    transient -- see the real site-c ping 0-500 case in this
    package's own ``CHANGELOG.md``, found by exactly this kind of
    altitude/pitch variability check). Adjacent unstable windows merge
    into one span; a lone unstable window narrower than
    ``min_run_pings`` is dropped (treated as detector noise, not a real
    transient) rather than excluded.

    ``stability_threshold``'s default (1.0) is a coarse, not-yet-
    validated starting point -- ``score_stability``'s own docstring
    describes "a genuinely calm stretch" scoring a few percent / a few
    tenths of a degree, and "a real deployment/turn transient" scoring
    1-2 orders of magnitude higher. Exposed as a parameter rather than
    trusted blindly on a new survey, the same caution this project
    already applies to every other auto-classification.
    """
    if "altitude_m" not in ds or "pitch_deg" not in ds:
        return []
    n_pings = ds.sizes["ping"]
    if n_pings < window_pings:
        return []
    altitude = np.asarray(ds["altitude_m"].values, dtype=np.float64)
    pitch = np.asarray(ds["pitch_deg"].values, dtype=np.float64)

    starts = list(range(0, n_pings, window_pings))
    unstable = [
        score_stability(altitude, pitch, s, min(window_pings, n_pings - s)) > stability_threshold
        for s in starts
    ]

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(starts):
        if unstable[i]:
            j = i
            while j < len(starts) and unstable[j]:
                j += 1
            span_start = starts[i]
            span_stop = min(starts[j - 1] + window_pings, n_pings)
            if span_stop - span_start >= min_run_pings:
                spans.append((span_start, span_stop))
            i = j
        else:
            i += 1
    return spans


def find_segments(
    ds: xr.Dataset,
    *,
    gap_factor: float = 8.0,
    time_gap_factor: float = 8.0,
    sound_vel_rel_tol: float = 0.05,
    garbage_value_var: str = "backscatter_raw",
    garbage_min_run_pings: int = 20,
    garbage_degenerate_frac: float = 0.999,
    stability_window_pings: int = 100,
    stability_threshold: float = 1.0,
    stability_min_run_pings: int = 50,
    min_segment_pings: int = 200,
) -> list[Segment]:
    """Combine all four detectors into a list of ``Segment``s covering
    ``[0, n_pings)`` with no gaps between them. Each is labeled
    ``"clean"`` or ``"excluded"``, with ``reasons`` naming every signal
    that fired inside or at its boundary -- the evidence for the
    decision, not just the decision itself.

    Status is resolved per raw interval *before* any merging, then only
    a too-short **clean** interval gets merged into an adjacent clean
    neighbor (preferring the previous one, matching the family's
    existing "return something smaller rather than nothing" convention
    for a region too short for the configured span -- see
    ``apply_config_pipeline.py:resolve_sections``'s own short-region
    fallback). ``min_segment_pings`` only exists to avoid a clean
    section too short to be practically usable -- a short **excluded**
    interval has no such requirement (nothing downstream reads it) and
    is never merged into a clean neighbor, which would silently taint
    real clean pings as excluded. **Real bug this fixes, found from
    real-data testing**: an earlier version merged by raw interval
    order regardless of status, then computed status from the *merged*
    range's own overlap with the excluded spans -- a 100-ping excluded
    span merging into a 1000-ping clean interval flipped the *entire*
    1100-ping result to ``"excluded"``, not just its own 100 pings.
    A too-short clean interval with no clean neighbor to merge into
    (e.g. it's the whole survey) is kept as its own short segment
    rather than forced into something else.
    """
    n_pings = ds.sizes["ping"]
    if n_pings == 0:
        return []

    gap_boundary = detect_gaps(ds, gap_factor=gap_factor, time_gap_factor=time_gap_factor)
    setting_boundary = detect_setting_changes(ds, sound_vel_rel_tol=sound_vel_rel_tol)
    garbage_spans = detect_garbage_runs(
        ds, garbage_value_var, min_run_pings=garbage_min_run_pings, degenerate_frac=garbage_degenerate_frac,
    )
    unstable_spans = detect_unstable_spans(
        ds, window_pings=stability_window_pings, stability_threshold=stability_threshold,
        min_run_pings=stability_min_run_pings,
    )

    boundary_points: set[int] = {0, n_pings}
    point_reasons: dict[int, list[str]] = {}
    for idx in np.flatnonzero(gap_boundary):
        boundary_points.add(int(idx))
        point_reasons.setdefault(int(idx), []).append("gap")
    for idx in np.flatnonzero(setting_boundary):
        boundary_points.add(int(idx))
        point_reasons.setdefault(int(idx), []).append("setting_change")

    span_reasons: dict[tuple[int, int], list[str]] = {}
    for s, e in garbage_spans:
        span_reasons.setdefault((s, e), []).append("garbage_run")
    for s, e in unstable_spans:
        span_reasons.setdefault((s, e), []).append("unstable")
    for s, e in span_reasons:
        boundary_points.add(s)
        boundary_points.add(min(e, n_pings))

    sorted_points = sorted(p for p in boundary_points if 0 <= p <= n_pings)
    raw_intervals = [
        (sorted_points[i], sorted_points[i + 1])
        for i in range(len(sorted_points) - 1) if sorted_points[i] < sorted_points[i + 1]
    ]

    def _status_and_reasons(s: int, e: int) -> tuple[str, list[str]]:
        reasons: list[str] = []
        status = "clean"
        for (gs, ge), rs in span_reasons.items():
            if max(0, min(e, ge) - max(s, gs)) > 0:
                status = "excluded"
                reasons.extend(rs)
        for p, rs in point_reasons.items():
            if s < p < e:
                reasons.extend(rs)
        return status, reasons

    merged: list[list] = []  # [start, stop, status, reasons]
    for s, e in raw_intervals:
        status, reasons = _status_and_reasons(s, e)
        if status == "clean" and (e - s) < min_segment_pings and merged and merged[-1][2] == "clean":
            merged[-1][1] = e
            merged[-1][3] = sorted(set(merged[-1][3] + reasons))
        else:
            merged.append([s, e, status, reasons])

    segments = [
        Segment(label=f"section{i}", start=s, stop=e, status=status, reasons=sorted(set(reasons)) if reasons else ["clean"])
        for i, (s, e, status, reasons) in enumerate(merged, 1)
    ]
    return segments


def write_segments(survey: SideScanSurvey, segments: list[Segment], output_dir: Path) -> list[Path]:
    """Write one ``.ss.h5`` per ``"clean"`` segment, slicing *every*
    frequency channel this survey has to that segment's ping range (not
    just the channel segmentation was computed from) via
    ``processing.slicing.slice_by_pings`` -- no new slicing logic.
    ``"excluded"`` segments are not written as files (see
    ``write_segments_report`` for their own record). Assumes every
    frequency channel shares the same ping axis/count, true for how
    this family's converters build a ``SideScanSurvey`` (each channel
    already globally ping-renumbered per mission/file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for seg in segments:
        if seg.status != "clean":
            continue
        sub_survey = SideScanSurvey(
            source_files=survey.source_files, source_format=survey.source_format,
            survey_name=f"{survey.survey_name}_{seg.label}",
        )
        for freq_key in survey.frequencies:
            channel = survey.get_frequency(freq_key)
            sliced = slice_by_pings(channel, start=seg.start, stop=seg.stop)
            sub_survey.add_frequency(freq_key, sliced.dataset)
        out_path = output_dir / f"{seg.label}.ss.h5"
        write_survey(sub_survey, out_path, overwrite=True)
        written.append(out_path)
    return written


def write_segments_report(segments: list[Segment], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"segments": [s.to_json_dict() for s in segments]}, f, indent=2)
