"""
Klein/SonarWiz native target catalog reader (``Target_<unix_ts>.tgt`` /
``.tgd`` pairs).

**Reverse-engineered from real files; not vendor-documented.**
No independent reference exists to check field meanings against —
validation consisted of rendering the raw window and checking that a
candidate field lands on a visible sonar contact, not a byte-for-byte
spec match. Every ``Target`` produced here gets
``position_confidence="unknown"`` unconditionally, regardless of which
fields below are confirmed — that flag is about absolute geographic
position (never found in this format at all, see below), not about
whether the on-image box placement is trustworthy.

What's confirmed (byte-exact, cross-checked across multiple real files):

- ``.tgt`` is not a documented catalog format itself, but ``file(1)``
  identifies it as a dBase-IV memo (``.DBT``) block file, and its content
  is consistent with that: a small fixed header, then a run of
  length-prefixed UTF-16LE string fields (each starting with the 3-byte
  marker ``FF FE FF`` then a 1-byte character count), then a fixed-layout
  binary trailer once the strings end.
- The file's own Unix timestamp (from its filename, ``Target_<ts>.tgt``)
  appears verbatim as a little-endian int32 at byte offset 8 — confirms
  the header is being read correctly and the timestamp is the pick time,
  not an arbitrary filename.
- ``.tgd`` is a fixed-size raw sonar window: an 8-byte header
  (``int32 n_channels, int32 n_samples`` — read as ``(2, 2500)`` on the
  one real survey checked, matching that survey's own converted
  ``.ss.h5`` sample count exactly), followed by
  ``(n_pings, n_channels, n_samples)`` int16 amplitude — real backscatter
  data, confirmed by a plausible leading-saturation run (same shape as
  the leading-blanking-run artifact `sidescanpy` already knows about from
  JSF data) and by-eye inspection of the decoded window showing genuine
  seafloor texture.

What's confirmed by visual cross-check:

- ``window_row``: the raw trailer int32 (``ints[5]``) is **2x the true
  row** — this module halves it before returning. Confirmed against
  six real targets by rendering each one's full decoded window with no
  box, locating the visible sonar contact by eye, and checking
  which candidate row landed on it. Five of six checks landed cleanly on
  a visible feature; one landed in a relatively low-contrast area (the
  column could be off for that target, or it is a weak contact).
  *Why* the field is 2x is unknown.
- ``slant_range_m`` and a constant ``max_window_range_m`` (varies by
  survey file, e.g. 40.0 or 60.0 seen so far): a float32 slot that
  varies per target in a plausible tens-of-meters range, paired with a
  float32 slot that's identical across every target in one file.
  ``max_window_range_m / n_samples`` gives a plausible meters-per-sample
  constant, used to convert ``slant_range_m`` into a window column
  index. Not contradicted by any of the six row-validation checks above
  (the column landed in the right vicinity in every case), but never
  independently verified the way ``window_row`` has been.

What's inferred, not confirmed (best-effort field mapping in the binary
trailer that follows the string fields, found by comparing several real
files and noting which int32/float32 slots vary plausibly vs. stay
constant):

- ``channel`` (0/1): an int32 slot that's always exactly 0 or 1. Not
  independently verified — the six-target row check above used whatever
  channel this field selected and found real features there, which is
  indirect support, but the alternate channel per target was never
  compared.
- The string fields ahead of the trailer are read and kept verbatim in
  ``extra_attributes`` (a short numeric-looking code, a decimal number,
  a line-ID-like string, and a notes field) but not interpreted --
  their meaning wasn't determined. The notes field is the *unedited
  placeholder text* ("The operator places target notes here. Multiple
  lines OK.") on every real target checked so far, not a real label —
  passed through as-is rather than guessed at.

No absolute latitude/longitude was found anywhere in ``.tgt`` (searched
exhaustively for plausible float32/float64 pairs in the survey area's
lat/lon range — none found). Position must be resolved externally, e.g.
by matching ``pick_timestamp`` against a converted survey's own per-ping
timestamps (see ``resolve_target_position``) — this is a *nearest-ping*
approximation, not a stored pick position, and does not upgrade
``position_confidence`` above ``"unknown"``.

2026 rshomberg@uri.edu
"""

import logging
import re
import struct
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Geod

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.targets.core import Target

logger = logging.getLogger(__name__)

_SOURCE_FORMAT = "sdf_klein"
_STRING_FIELD_MARKER = b"\xff\xfe\xff"
_STRING_FIELDS_START = 0x28
_MAX_STRING_FIELDS = 8
_TRAILER_INT_COUNT = 20

# Trailer slot indices (0-based, after the string fields end) -- see
# module docstring for how these were inferred, and the "not confirmed"
# caveat that applies to every one of them.
_IDX_CHANNEL = 2
_IDX_RANGE_M = 3
_IDX_ROW = 5
_IDX_MAX_RANGE_M = 8


def _parse_string_fields(data: bytes, start: int = _STRING_FIELDS_START) -> tuple[list[str], int]:
    """Parse the run of length-prefixed UTF-16LE fields starting at
    ``start``. Returns (field_values, offset_immediately_after_fields)."""
    off = start
    fields = []
    for _ in range(_MAX_STRING_FIELDS):
        if data[off:off + 3] != _STRING_FIELD_MARKER:
            break
        length = data[off + 3]
        text_start = off + 4
        text_bytes = data[text_start:text_start + length * 2]
        text = text_bytes.decode("utf-16-le", errors="replace")
        fields.append(text)
        off = text_start + length * 2
    return fields, off


def _target_id_from_filename(path: Path) -> str:
    # "Target_1536923484.tgt" -> "1536923484"
    return path.stem.removeprefix("Target_")


def _read_one_target(tgt_path: Path) -> Target:
    data = tgt_path.read_bytes()
    pick_timestamp = float(struct.unpack_from("<i", data, 8)[0])

    fields, end_off = _parse_string_fields(data)
    trailer = data[end_off:end_off + _TRAILER_INT_COUNT * 4]
    ints = struct.unpack_from(f"<{_TRAILER_INT_COUNT}i", trailer)
    floats = struct.unpack_from(f"<{_TRAILER_INT_COUNT}f", trailer)

    channel = ints[_IDX_CHANNEL]
    # The raw trailer value is 2x the true row -- confirmed empirically
    # (see module docstring), cause unknown.
    window_row = ints[_IDX_ROW] // 2
    range_m = floats[_IDX_RANGE_M]
    max_range_m = floats[_IDX_MAX_RANGE_M]

    tgd_path = tgt_path.with_suffix(".tgd")
    sample = None
    if tgd_path.exists() and max_range_m > 0:
        n_samples = struct.unpack_from("<i", tgd_path.read_bytes(), 4)[0]
        meters_per_sample = max_range_m / n_samples
        sample = int(round(range_m / meters_per_sample))

    extra_attributes = {
        "window_row": window_row,
        "max_window_range_m": max_range_m,
        "range_m": range_m,
        "string_fields": fields,
        "notes": fields[-1] if fields else None,
    }

    return Target(
        target_id=_target_id_from_filename(tgt_path),
        source_format=_SOURCE_FORMAT,
        catalog_source=str(tgt_path),
        pick_timestamp=pick_timestamp,
        channel=channel,
        sample=sample,
        position_confidence="unknown",
        extra_attributes=extra_attributes,
    )


def read_klein_targets(directory: str | Path) -> list[Target]:
    """Read every ``Target_<unix_ts>.tgt`` file in ``directory``.

    Returns one :class:`Target` per file, sorted by filename. Files that
    fail to parse are skipped with a warning rather than aborting the
    whole batch (matches ``io/jsf/targets.py: read_discover_targets``'s
    convention). ``position_confidence`` is unconditionally ``"unknown"``
    on every target returned — see the module docstring.
    """
    targets = []
    for path in sorted(Path(directory).glob("Target_*.tgt")):
        try:
            targets.append(_read_one_target(path))
        except Exception:
            logger.warning(f"Failed to read target catalog {path}", exc_info=True)
    return targets


def decode_target_window(target: Target) -> np.ndarray:
    """Re-decode the raw ``(n_pings, n_samples)`` window for
    ``target.channel`` from its paired ``.tgd`` file.

    Returns real int16 sonar amplitude (not a rendered/processed image),
    cast to float64. Row ``target.extra_attributes["window_row"]`` is the
    target's own ping within this array (confirmed by visual cross-check
    — see module docstring); column ``target.sample`` is its inferred
    position within each ping (not independently verified to the same
    degree).
    """
    tgd_path = Path(target.catalog_source).with_suffix(".tgd")
    data = tgd_path.read_bytes()
    n_channels, n_samples = struct.unpack_from("<2i", data, 0)
    rest = data[8:]
    n_pings = len(rest) // (n_channels * n_samples * 2)
    window = np.frombuffer(rest, dtype="<i2").reshape(n_pings, n_channels, n_samples)
    channel = target.channel if target.channel is not None else 0
    return window[:, channel, :].astype(np.float64)


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


def calibrate_clock_offset(ss_h5_dir: str | Path, local_utc_offset_hours: float = 4.0) -> float:
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
    local_utc_offset_hours : float, default 4.0
        Local-time-to-UTC offset for the filename's embedded local time
        (default 4.0 = EDT). Pick the offset that was actually in effect
        at the survey's location/date, not a hemisphere/season assumption.

    Returns
    -------
    float
        Median offset in seconds (corrupted ping time - true start time).

    Raises
    ------
    RuntimeError
        If no file in ``ss_h5_dir`` matched the expected filename pattern.
    """
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
