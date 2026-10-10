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

- ``channel`` (0/1): an int32 slot that's always exactly 0 or 1; 0 =
  port, 1 = starboard. Confirmed on real catalogs: the ``.tgd`` window
  matches the survey's own pings on that side.
- The string fields ahead of the trailer are kept verbatim in
  ``extra_attributes["string_fields"]``: ping time, ping number (both
  interpreted below), a decimal number (not interpreted), the target
  name, and a notes field. The notes field is the *unedited placeholder
  text* ("The operator places target notes here. Multiple lines OK.") on
  every real target checked so far, not a real label.

Catalog ping time and number (confirmed on real catalogs by matching
every ``.tgd`` window back to the survey's own pings):

- String field 0 is the date-time (``M/D/YYYY h:mm:ss``) and field 1 the
  ping number of the **target's own ping**, both on the sonar's own clock,
  the same clock as the pings in the ``.sdf`` files (which can be years off
  real time). Parsed into ``extra_attributes["catalog_ping_time"]`` (Unix
  seconds on that clock, 1 s resolution) and
  ``extra_attributes["catalog_ping_number"]``. Ping counters can restart
  after a recording gap, so the number alone is not unique; the time
  disambiguates. Use
  :func:`sidescanpy.targets.correlate.resolve_targets_by_catalog_ping` to
  place targets: it needs no clock offset.
- ``pick_timestamp`` (the ``Target_<unix>`` file name) is when the pick
  was saved, on the acquisition PC's clock: typically 30-90 s *after* the
  target ping, and hours later for picks made during playback. Matching
  pings by it places targets tens to hundreds of metres along track from
  where they are.
- The ``.tgd`` window holds the catalog channel of the high-frequency
  data, one row per **second** ping, **newest ping first**. ``window_row``
  counts rows from the top, so its ping is ``newest - 2 * window_row``.
  On real catalogs this lands on the catalog ping for about three
  quarters of the targets; for the rest it is several to ~100 pings
  older, so ``window_row`` is not a reliable target ping. ``sample`` (from
  the range field) has been consistent.
- Some picks are made from the live display while logging is off; their
  catalog time falls in a recording gap and the ``.tgd`` is the only copy
  of that data.

No absolute latitude/longitude was found anywhere in ``.tgt`` (searched
exhaustively for plausible float32/float64 pairs in the survey area's
lat/lon range — none found). Position must be resolved externally, e.g.
by matching ``pick_timestamp`` against a converted survey's own per-ping
timestamps (see ``sidescanpy.targets.correlate.resolve_target_position``) — this is a *nearest-ping*
approximation, not a stored pick position, and does not upgrade
``position_confidence`` above ``"unknown"``.

Parsing only: position resolution and clock-offset calibration against
converted ``.ss.h5`` files live in :mod:`sidescanpy.targets.correlate`.

2026 rshomberg@uri.edu
"""

import logging
import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sidescanpy.core.target import Target

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


def _parse_catalog_time(text: str) -> float | None:
    """``"5/13/2026 14:51:05"`` -> Unix seconds, read as UTC on the sonar's clock."""
    try:
        return datetime.strptime(text.strip(), "%m/%d/%Y %H:%M:%S").replace(tzinfo=UTC).timestamp()
    except ValueError:
        logger.debug(f"Unparseable catalog ping time {text!r}")
        return None


def _parse_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        logger.debug(f"Unparseable catalog ping number {text!r}")
        return None


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
    meters_per_sample = None
    if tgd_path.exists() and max_range_m > 0:
        n_samples = struct.unpack_from("<i", tgd_path.read_bytes(), 4)[0]
        meters_per_sample = max_range_m / n_samples
        sample = int(round(range_m / meters_per_sample))

    catalog_ping_time = _parse_catalog_time(fields[0]) if fields else None
    catalog_ping_number = _parse_int(fields[1]) if len(fields) > 1 else None

    extra_attributes = {
        "catalog_ping_time": catalog_ping_time,
        "catalog_ping_number": catalog_ping_number,
        "window_row": window_row,
        "max_window_range_m": max_range_m,
        # The window's sample spacing: ``sample`` counts samples at this
        # spacing, which picks the matching frequency when targets are
        # correlated to a survey (frequencies can differ in spacing).
        "ping_meters_per_sample": meters_per_sample,
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
    cast to float64. Rows are every second ping, newest first. Row
    ``target.extra_attributes["window_row"]`` is the catalog's box row,
    which is not always the target's own ping (see module docstring);
    column ``target.sample`` is the target's sample within each ping.
    """
    tgd_path = Path(target.catalog_source).with_suffix(".tgd")
    data = tgd_path.read_bytes()
    n_channels, n_samples = struct.unpack_from("<2i", data, 0)
    rest = data[8:]
    n_pings = len(rest) // (n_channels * n_samples * 2)
    window = np.frombuffer(rest, dtype="<i2").reshape(n_pings, n_channels, n_samples)
    channel = target.channel if target.channel is not None else 0
    return window[:, channel, :].astype(np.float64)
