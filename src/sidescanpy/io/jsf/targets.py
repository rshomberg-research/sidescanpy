"""
EdgeTech Discover SQLite ``.target`` catalog reader.

Discover writes one SQLite database per target pick (``Target-NNNN.target``),
holding a raw JSF ping-sample window plus position metadata. Schema
(``Target``, ``PingChannel``, ``MutableTarget``, ``MutableTag`` tables) was
reverse-engineered by hand and verified against real Discover target files.

Known gotchas this reader accounts for:

- ``Target.TargetId`` is a per-file primary key that always resets to 1 —
  never a cross-file identifier. The real identity is the filename's own
  number (``Target-NNNN.target``), used as ``Target.target_id`` here.
- ``PingChannel``'s ~200-row window is *usually* a clean contiguous run of
  ``PingSequenceNumber`` values, but not always. One real file seen so far
  (``Target-0002.target``) has two things wrong at once: (1) a uint32
  rollover artifact — 104 of its rows, *and the target's own
  ``MutableTarget.PingSequenceNumber``*, are stored as
  ``true_value + 2**32`` (e.g. raw ``4294971779`` for a true ``4483``,
  confirmed by the corrected value's neighbors forming a clean contiguous
  run); (2) an unrelated, disjoint cluster of 94 rows (``4846``-``4939``)
  mixed into the same file's ``PingChannel`` table, nowhere near the
  target's real window. Correcting the rollover and then walking outward
  from the target's own (corrected) ping to find its contiguous run
  handles both at once — no fixed distance threshold from the target's
  raw ping is reliable here, since the target's own raw value is exactly
  what's corrupted. ``position_confidence`` downgrades to
  ``"approximate"`` (recording the exclusion in ``extra_attributes``)
  whenever this filtering actually removes rows.
- ``MutableTarget.Description``/``MutableTag.Tag`` are commonly empty —
  this is normal (no classification was ever recorded), not a read failure;
  ``label`` stays ``None`` rather than a placeholder string.

Position formula (validated visually against 6+ real targets): within the ping window sorted by
sequence number, the target's row is the index of its own
``PingSequenceNumber``; its column is
``round(SlantRangeDistanceMeters / PingMetersPerSample)`` using the
*target's own ping's* ``PingMetersPerSample`` (not a window-wide constant,
though it typically doesn't vary within one window).

Decode (:func:`decode_target_window`; raw values, native orientation --
TVG and display orientation live in ``export.target_window``): ``SampleBytes`` -> ``np.frombuffer(dtype="<i2").astype(float) *
2 ** (-JsfWeightingFactor)`` — real JSF sample amplitudes, not a rendered
image. Expect a leading run of exact-zero samples (blanking/ring-down);
not a bug.

2026 rshomberg@uri.edu
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np

from sidescanpy.core.target import Target

logger = logging.getLogger(__name__)

_SOURCE_FORMAT = "jsf_discover"

# A PingSequenceNumber this large is treated as a uint32 rollover artifact
# (true_value + 2**32) rather than a real ping index — real surveys are
# nowhere near this many pings.
_ROLLOVER_THRESHOLD = 2**31
_UINT32_ROLLOVER = 2**32


def _corrected_seq(seq: int) -> int:
    return seq - _UINT32_ROLLOVER if seq > _ROLLOVER_THRESHOLD else seq


def _contiguous_window(raw_seqs: list[int], target_raw_seq: int) -> tuple[list[int], bool]:
    """Find the contiguous run of ``PingSequenceNumber`` values containing
    ``target_raw_seq``, correcting the uint32 rollover artifact first.

    Returns ``(run_raw_seqs, was_filtered)`` — ``run_raw_seqs`` holds the
    original *raw* (uncorrected) sequence numbers, ordered by their
    corrected chronological position, restricted to the contiguous run
    containing the target's own ping; ``was_filtered`` is True whenever
    that run is smaller than ``raw_seqs`` (rows excluded, whether due to
    rollover correction, an unrelated disjoint cluster, or both).
    """
    by_corrected: dict[int, int] = {}
    for raw in raw_seqs:
        by_corrected.setdefault(_corrected_seq(raw), raw)

    corrected_target = _corrected_seq(target_raw_seq)
    if corrected_target not in by_corrected:
        return [], True

    corrected_sorted = sorted(by_corrected)
    idx = corrected_sorted.index(corrected_target)
    lo = idx
    while lo > 0 and corrected_sorted[lo - 1] == corrected_sorted[lo] - 1:
        lo -= 1
    hi = idx
    while hi < len(corrected_sorted) - 1 and corrected_sorted[hi + 1] == corrected_sorted[hi] + 1:
        hi += 1

    run_raw = [by_corrected[c] for c in corrected_sorted[lo : hi + 1]]
    return run_raw, len(run_raw) != len(raw_seqs)


def read_discover_targets(directory: str | Path) -> list[Target]:
    """Read every ``Target-NNNN.target`` file in ``directory``.

    Returns one :class:`Target` per file, sorted by filename. Files that
    fail to parse are skipped with a warning rather than aborting the
    whole batch (matches the family's "clip/skip rather than error on one
    bad row" convention elsewhere in this library).
    """
    targets = []
    for path in sorted(Path(directory).glob("Target-*.target")):
        try:
            targets.append(_read_one_target(path))
        except Exception:
            logger.warning(f"Failed to read target catalog {path}", exc_info=True)
    return targets


def _target_id_from_filename(path: Path) -> str:
    # "Target-0001.target" -> "0001"
    return path.stem.removeprefix("Target-")


def _read_one_target(path: Path) -> Target:
    conn = sqlite3.connect(path)
    try:
        channel = conn.execute("select ChannelNumber from Target").fetchone()[0]

        ping_seq, slant_range_m, description = conn.execute(
            "select PingSequenceNumber, SlantRangeDistanceMeters, Description from MutableTarget"
        ).fetchone()
        tags = [r[0] for r in conn.execute("select Tag from MutableTag")]
        label = tags[0] if tags else description

        window_rows = conn.execute(
            "select PingSequenceNumber, PingMetersPerSample, PositionX, PositionY "
            "from PingChannel order by rowid"
        ).fetchall()
    finally:
        conn.close()

    all_seqs = [r[0] for r in window_rows]
    window_seqs, was_filtered = _contiguous_window(all_seqs, ping_seq)
    n_excluded = len(all_seqs) - len(window_seqs)

    by_seq = {r[0]: r for r in window_rows}
    own_row = by_seq.get(ping_seq)
    _, mps, position_x, position_y = own_row if own_row is not None else (None, None, None, None)

    sample_col = round(slant_range_m / mps) if (slant_range_m is not None and mps is not None) else None
    window_row = window_seqs.index(ping_seq) if ping_seq in window_seqs else None

    extra_attributes: dict = {
        "window_ping_sequence": window_seqs,
        "window_row": window_row,
        "ping_meters_per_sample": mps,
    }
    if was_filtered:
        extra_attributes["window_filtered"] = True
        extra_attributes["n_window_rows_excluded"] = n_excluded

    if window_row is None:
        position_confidence = "unknown"
    elif was_filtered:
        position_confidence = "approximate"
    else:
        position_confidence = "verified"

    return Target(
        target_id=_target_id_from_filename(path),
        source_format=_SOURCE_FORMAT,
        catalog_source=str(path),
        ping=ping_seq,
        channel=channel,
        sample=sample_col,
        latitude=position_y,
        longitude=position_x,
        position_confidence=position_confidence,
        label=label,
        extra_attributes=extra_attributes,
    )


def decode_target_window(target: Target) -> np.ndarray:
    """Re-decode the raw ping-sample window for ``target`` from its source
    SQLite catalog.

    Returns a ``(n_pings, n_samples)`` float array of real JSF sample
    amplitudes (not a rendered/processed image) — samples per ping are
    truncated to the shortest row in the window, in the rare case they
    differ. Row ``target.extra_attributes["window_row"]`` is the target's
    own ping within this array; column ``target.sample`` is its computed
    position within each ping.
    """
    seq_order = target.extra_attributes["window_ping_sequence"]
    conn = sqlite3.connect(target.catalog_source)
    try:
        rows = []
        for seq in seq_order:
            sample_bytes, weighting_factor = conn.execute(
                "select SampleBytes, JsfWeightingFactor from PingChannel where PingSequenceNumber = ?",
                (seq,),
            ).fetchone()
            arr = np.frombuffer(sample_bytes, dtype="<i2").astype(np.float64) * 2.0 ** (-weighting_factor)
            rows.append(arr)
    finally:
        conn.close()

    n_samples = min(len(r) for r in rows)
    return np.stack([r[:n_samples] for r in rows])
