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
timestamps (see ``sidescanpy.targets.correlate.resolve_target_position``) — this is a *nearest-ping*
approximation, not a stored pick position, and does not upgrade
``position_confidence`` above ``"unknown"``.

Parsing only: position resolution and clock-offset calibration against
converted ``.ss.h5`` files live in :mod:`sidescanpy.targets.correlate`.

2026 rshomberg@uri.edu
"""

import logging
import struct
from pathlib import Path

import numpy as np

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
