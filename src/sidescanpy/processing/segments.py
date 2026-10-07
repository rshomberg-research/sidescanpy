"""
Survey segmentation: find clean, homogeneous ping ranges in one channel.

Aggregate analysis over a ping range that silently mixes a gap, a
setting change, a dropout or a turn transient gives misleading numbers.
These detectors find such stretches so they can be cut out first.

- :func:`detect_gaps` — along-track distance or ping-time jumps.
- :func:`detect_setting_changes` — ``valid_samples`` or sound-velocity
  steps.
- :func:`detect_garbage_runs` — sustained runs of all-NaN/zero pings.
- :func:`detect_unstable_spans` — altitude/pitch instability windows
  (scored by :func:`score_stability`).

:func:`find_segments` combines them into :class:`Segment` objects
labelled ``"clean"`` or ``"excluded"`` with the reasons that fired.
:func:`split_by_segments` slices a survey into one sub-survey per clean
segment.
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.processing.slicing import slice_by_pings

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """One ``[start, stop)`` ping range with its status and the reasons for it."""

    label: str
    start: int
    stop: int
    status: str  # "clean" | "excluded"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return asdict(self)


def detect_gaps(ds: xr.Dataset, *, gap_factor: float = 8.0, time_gap_factor: float = 8.0) -> np.ndarray:
    """Per-ping boundary flags: ``True`` at ``i`` marks a break between pings ``i-1`` and ``i``.

    Fires when the along-track step exceeds ``gap_factor`` times the
    median nonzero step, or the ping-time step exceeds
    ``time_gap_factor`` times the median nonzero interval. Either signal
    alone can show a real discontinuity. Index 0 is never flagged.
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
    """Per-ping boundary flags for a likely acquisition-setting change.

    Fires on any change in ``valid_samples`` (when present) or a relative
    step in ``sound_vel_ms`` larger than ``sound_vel_rel_tol``. A
    sample-count change that the converter already padded away without
    recording ``valid_samples`` can't be detected here.
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
    ds: xr.Dataset, value_var: str = "backscatter_intensity", *,
    min_run_pings: int = 20, degenerate_frac: float = 0.999,
) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` runs of at least ``min_run_pings`` degenerate pings.

    A ping is degenerate when at least ``degenerate_frac`` of its values
    (all channels and samples) are NaN or exactly zero — a mid-survey
    dropout, not the per-ping leading nadir blanking.
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
    """Stability score of ``[start, start + span)``; lower is calmer.

    Altitude coefficient of variation plus pitch standard deviation
    (degrees), weighted 1:1. A calm stretch scores a few hundredths to
    tenths; a turn or deployment transient scores one to two orders of
    magnitude higher. ``inf`` with fewer than two finite values.
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
    """``[(start, stop), ...]`` unstable stretches from altitude and pitch.

    Non-overlapping ``window_pings`` windows scoring above
    ``stability_threshold`` (:func:`score_stability`) are unstable;
    adjacent ones merge, and a merged span shorter than ``min_run_pings``
    is dropped as noise. The default threshold is a coarse starting
    point, not a validated value. Empty without ``altitude_m`` and
    ``pitch_deg``.
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
    garbage_value_var: str = "backscatter_intensity",
    garbage_min_run_pings: int = 20,
    garbage_degenerate_frac: float = 0.999,
    stability_window_pings: int = 100,
    stability_threshold: float = 1.0,
    stability_min_run_pings: int = 50,
    min_segment_pings: int = 200,
) -> list[Segment]:
    """Split ``[0, n_pings)`` into contiguous :class:`Segment` objects.

    Gap and setting-change flags become boundaries; garbage and unstable
    spans become ``"excluded"`` segments. Each segment's ``reasons``
    names every signal that fired inside it (``["clean"]`` if none).

    Status is resolved per raw interval before merging. Only a clean
    interval shorter than ``min_segment_pings`` is merged, and only into
    a preceding clean neighbour, so a short excluded span never taints
    clean pings. A short clean interval with no clean neighbour is kept
    as is.
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
    logger.debug("find_segments: %d pings -> %d segment(s)", n_pings, len(segments))
    return segments


def split_by_segments(survey: SideScanSurvey, segments: list[Segment]) -> dict[str, SideScanSurvey]:
    """One sub-survey per ``"clean"`` segment, keyed by segment label.

    Every frequency channel is sliced to the segment's ping range with
    :func:`~sidescanpy.processing.slicing.slice_by_pings`; all channels
    are assumed to share one ping axis. Excluded segments are skipped.
    """
    out: dict[str, SideScanSurvey] = {}
    for seg in segments:
        if seg.status != "clean":
            continue
        sub_survey = SideScanSurvey(
            source_files=survey.source_files, source_format=survey.source_format,
            survey_name=f"{survey.survey_name}_{seg.label}",
        )
        for freq_key in survey.frequencies:
            ds_f = survey.get_frequency(freq_key).dataset
            sub_survey.add_frequency(freq_key, slice_by_pings(ds_f, start=seg.start, stop=seg.stop))
        out[seg.label] = sub_survey
    return out
