"""
Resolve target positions and clock offsets against converted ``.ss.h5`` files.

Catalog formats that store no absolute position (the Klein ``.tgt`` reader in
:mod:`sidescanpy.io.sdf.targets`) are located here by matching each target's
pick timestamp against per-ping times of converted surveys, with optional
constant clock-offset calibration and geodesic speed estimates.

2026 rshomberg@uri.edu
"""

import logging
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Geod

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.targets.core import Target

logger = logging.getLogger(__name__)


def resolve_target_position(
    target: Target,
    survey: SideScanSurvey,
    freq_key: str,
    *,
    max_time_delta_s: float = 300.0,
) -> Target:
    """Approximate ``target``'s position from the nearest ping (by
    timestamp) in ``survey``'s ``freq_key`` channel.

    This is a coarse nearest-ping lookup, not a stored pick position (no
    absolute lat/lon was found in ``.tgt`` — see module docstring) and
    does **not** upgrade ``position_confidence`` above ``"unknown"``.

    Parameters
    ----------
    target : Target
        Must have ``pick_timestamp`` set.
    survey : SideScanSurvey
    freq_key : str
    max_time_delta_s : float, default 300.0
        If the nearest ping's timestamp is farther than this from
        ``target.pick_timestamp``, position is left unset (``target``
        returned unchanged) rather than guessed from a clearly-wrong file.

    Returns
    -------
    Target
        Copy of ``target`` with ``latitude``/``longitude`` set if a
        close-enough ping was found, otherwise unchanged.
    """
    if target.pick_timestamp is None:
        return target

    ds = survey.get_frequency(freq_key).dataset
    ping_times_s = ds["time"].values.astype("datetime64[s]").astype(np.int64)
    idx = int(np.argmin(np.abs(ping_times_s - target.pick_timestamp)))
    delta_s = abs(float(ping_times_s[idx]) - target.pick_timestamp)
    if delta_s > max_time_delta_s:
        logger.debug(
            f"Target {target.target_id}: nearest ping is {delta_s:.0f}s away "
            f"(> {max_time_delta_s}s), leaving position unset"
        )
        return target

    lat = float(ds["latitude"].values[idx])
    lon = float(ds["longitude"].values[idx])
    extra_attributes = dict(target.extra_attributes)
    extra_attributes["position_source"] = "nearest_ping_timestamp"
    extra_attributes["position_time_delta_s"] = delta_s
    return replace(target, latitude=lat, longitude=lon, extra_attributes=extra_attributes)


# ============================================================
# Clock-offset calibration
# ============================================================
#
# A survey-specific data-quality issue: converted
# .ss.h5 ping timestamps can read a wildly wrong date (e.g. years off
# when every raw filename and every target's own catalog
# timestamp agree on another date) -- present in the source .sdf/.tgt files
# themselves, not a sidescanpy bug. Plain resolve_target_position()
# nearest-ping matching finds no match at all under this condition and
# silently resolves zero positions. The offset turns out to be a
# *constant* per survey, not a drifting clock -- derived here from each
# .ss.h5 file's own filename-embedded true start time vs. its corrupted
# first-ping timestamp. This is a Klein-acquisition-chain characteristic
# that could recur on any Klein SDF survey.

_FILENAME_TS_RE = re.compile(r"(\d{6})(\d{6})")
_GEOD = Geod(ellps="WGS84")


def calibrate_clock_offset(ss_h5_dir: str | Path, *, local_utc_offset_hours: float) -> float:
    """Derive the constant epoch-shift between each ``.ss.h5``'s
    (possibly corrupted) ping clock and its filename-embedded true local
    start time.

    Filenames are expected to embed ``<YYMMDD><HHMMSS>`` local start time
    (matches Klein SDF's own ``sonar_data<YYMMDD><HHMMSS>.sdf`` naming,
    preserved through conversion). Returns the median offset (seconds,
    corrupted minus true) across every file found in ``ss_h5_dir``,
    logging each file's individual estimate so a bad calibration (large
    spread => the offset isn't actually constant for this survey) is
    visible rather than silently trusted.

    Parameters
    ----------
    ss_h5_dir : str or Path
    local_utc_offset_hours : float
        Required keyword. Local-time-to-UTC offset (hours to add to the
        filename's embedded local time to get UTC). Use the offset that was
        actually in effect at the survey's location/date -- there is
        deliberately no default.

    Returns
    -------
    float
        Median offset in seconds (corrupted ping time - true start time).

    Raises
    ------
    RuntimeError
        If no file in ``ss_h5_dir`` matched the expected filename pattern.
    """
    logger.debug(f"Calibrating clock offset with local_utc_offset_hours={local_utc_offset_hours}")
    offsets = []
    for path in sorted(Path(ss_h5_dir).glob("*.ss.h5")):
        m = _FILENAME_TS_RE.search(path.stem)
        if not m:
            logger.warning(f"  {path.name}: filename doesn't match <date><time> pattern, skipping calibration")
            continue
        local_dt = datetime.strptime("20" + m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        # .timestamp() on a naive datetime uses the *system's* local
        # timezone, not UTC -- force UTC explicitly or this silently
        # reintroduces a timezone-sized error into the calibration.
        true_utc = (local_dt + timedelta(hours=local_utc_offset_hours)).replace(tzinfo=UTC)
        survey = read_survey(path)
        freq_key = next(iter(survey.frequencies))
        first_ping = survey.get_frequency(freq_key).dataset["time"].values[0]
        corrupted_utc = first_ping.astype("datetime64[s]").astype(int)
        offset_s = float(corrupted_utc) - true_utc.timestamp()
        offsets.append(offset_s)
        logger.debug(f"  {path.name}: true_start={true_utc} corrupted_first_ping_offset_s={offset_s:.1f}")

    if not offsets:
        raise RuntimeError(f"Could not calibrate offset from any file in {ss_h5_dir}")
    offset_s = float(np.median(offsets))
    spread = max(offsets) - min(offsets)
    logger.debug(f"Calibrated clock offset: {offset_s:.1f}s (median of {len(offsets)} files, spread {spread:.1f}s)")
    if spread > 120:
        logger.warning(
            f"Offset spread across files is {spread:.1f}s -- larger than expected for a fixed "
            "epoch-shift bug. Per-target matching may be less reliable than usual; treat "
            "altitude/speed/position from this calibration with extra caution."
        )
    return offset_s


def _preload_ss_h5_nav(ss_h5_dir: Path) -> list[dict]:
    loaded = []
    for path in sorted(Path(ss_h5_dir).glob("*.ss.h5")):
        survey = read_survey(path)
        freq_key = next(iter(survey.frequencies))
        ds = survey.get_frequency(freq_key).dataset
        loaded.append({
            "stem": path.stem,
            "times_s": ds["time"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64),
            "alt": ds["altitude_m"].values,
            "lat": ds["latitude"].values,
            "lon": ds["longitude"].values,
        })
    return loaded


def _local_speed_mps(rec: dict, idx: int) -> tuple[float | None, float | None]:
    """Geodesic speed (m/s) and meters/ping near ``idx``, widening the
    window adaptively -- file-start/end pings sometimes carry a repeated
    stale nav fix (dt=0 between consecutive entries) that a fixed window
    can land entirely inside."""
    n = len(rec["lat"])
    for half_window in (5, 15, 30, 60, 150):
        lo = max(0, idx - half_window)
        hi = min(n - 1, idx + half_window)
        if hi <= lo:
            continue
        dt_s = rec["times_s"][hi] - rec["times_s"][lo]
        if dt_s <= 0:
            continue
        _, _, dist_m = _GEOD.inv(rec["lon"][lo], rec["lat"][lo], rec["lon"][hi], rec["lat"][hi])
        return dist_m / dt_s, dist_m / (hi - lo)
    return None, None


def _resolve_one_offset_corrected(target: Target, preloaded: list[dict], offset_s: float, max_time_delta_s: float) -> Target:
    if target.pick_timestamp is None:
        return target
    shifted = target.pick_timestamp + offset_s

    best = None
    for rec in preloaded:
        times_s = rec["times_s"]
        idx = int(np.argmin(np.abs(times_s - shifted)))
        delta = abs(times_s[idx] - shifted)
        if best is None or delta < best[2]:
            best = (rec, idx, delta)
    if best is None or best[2] > max_time_delta_s * 20:
        # 20x the plain-match tolerance: gaps between recording segments
        # (vessel repositioning) are expected and still useful for
        # altitude/speed, unlike a genuinely unmatched target.
        return target

    rec, idx, delta = best
    altitude_m = float(rec["alt"][idx])
    lat, lon = float(rec["lat"][idx]), float(rec["lon"][idx])
    speed_mps, meters_per_ping = _local_speed_mps(rec, idx)

    extra_attributes = dict(target.extra_attributes)
    extra_attributes["position_source"] = "offset_corrected_nearest_ping_timestamp"
    extra_attributes["position_time_delta_s"] = delta
    extra_attributes["altitude_m"] = altitude_m
    if speed_mps is not None:
        extra_attributes["speed_mps"] = speed_mps
        extra_attributes["meters_per_ping"] = meters_per_ping

    range_m = target.extra_attributes.get("range_m")
    if range_m is not None:
        extra_attributes["ground_range_m"] = float(np.sqrt(max(range_m**2 - altitude_m**2, 0.0)))
        extra_attributes["ground_range_clipped"] = altitude_m >= range_m

    return replace(target, latitude=lat, longitude=lon, extra_attributes=extra_attributes)


def resolve_targets_offset_corrected(
    targets: list[Target],
    ss_h5_dir: str | Path,
    offset_s: float,
    *,
    max_time_delta_s: float = 300.0,
) -> list[Target]:
    """Resolve position (plus altitude/speed) for every target in
    ``targets`` using a known constant clock offset (see
    :func:`calibrate_clock_offset`).

    Enriches each matched target with ``altitude_m``/``speed_mps``/
    ``meters_per_ping``/``ground_range_m`` in ``extra_attributes`` — all
    approximations from the nearest matched ping, not per-sample
    measurements, and constant across a target's ~256-ping window.
    ``position_confidence`` stays ``"unknown"`` regardless of match
    quality, matching :func:`resolve_target_position`'s convention.
    Targets with no ``pick_timestamp``, or whose nearest match (across
    every ``.ss.h5`` file in ``ss_h5_dir``) is farther than
    ``20 * max_time_delta_s`` away, are returned unchanged.
    """
    preloaded = _preload_ss_h5_nav(Path(ss_h5_dir))
    return [_resolve_one_offset_corrected(t, preloaded, offset_s, max_time_delta_s) for t in targets]


def resolve_targets_nearest_ping(
    targets: list[Target],
    ss_h5_dir: str | Path,
    *,
    max_time_delta_s: float = 300.0,
) -> list[Target]:
    """Resolve position for every target in ``targets`` via plain
    (no clock-offset) nearest-ping matching, checking every ``.ss.h5``
    file and every frequency channel in ``ss_h5_dir`` and keeping
    whichever match lands closest in time.

    Thin batch wrapper around :func:`resolve_target_position` — use
    :func:`resolve_targets_offset_corrected` instead when the survey is
    known to need clock-offset correction (see that function and
    :func:`calibrate_clock_offset`).
    """
    ss_h5_dir = Path(ss_h5_dir)
    resolved = []
    for target in targets:
        best, best_delta = target, None
        for path in sorted(ss_h5_dir.glob("*.ss.h5")):
            survey = read_survey(path)
            for freq_key in survey.frequencies:
                candidate = resolve_target_position(target, survey, freq_key, max_time_delta_s=max_time_delta_s)
                if candidate.latitude is not None:
                    delta = candidate.extra_attributes.get("position_time_delta_s", float("inf"))
                    if best_delta is None or delta < best_delta:
                        best, best_delta = candidate, delta
        resolved.append(best)
    return resolved
