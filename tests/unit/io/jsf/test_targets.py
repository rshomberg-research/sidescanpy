"""
Unit tests for sidescanpy.io.jsf.targets, against small synthetic SQLite
fixtures built here (not real data) — shaped after the real schema and
the real Target-0002.target anomaly confirmed on real data.

2026 rshomberg@uri.edu
"""

import sqlite3
from pathlib import Path

import numpy as np

from sidescanpy.io.jsf.targets import (
    decode_target_window,
    read_discover_targets,
)

_SCHEMA = """
CREATE TABLE Target (
    TargetId INTEGER, SubsystemNumber INTEGER, ChannelNumber INTEGER,
    SpatialReferenceSystemIdentifier INTEGER
);
CREATE TABLE PingChannel (
    PingSequenceNumber INTEGER, PingTimestamp INTEGER, PingStartMeters REAL,
    PingEndMeters REAL, PingMetersPerSample REAL, SamplesCount INTEGER,
    JsfDataFormat INTEGER, JsfWeightingFactor INTEGER, SampleBytes BLOB,
    AltitudeMeters REAL, Course REAL, Heading REAL, PositionX REAL,
    PositionY REAL, MetersSinceLastPingHeuristic REAL, TargetId INTEGER
);
CREATE TABLE MutableTarget (
    Description TEXT, Thumbnail BLOB, Palette BLOB, TvgA REAL, TvgB REAL,
    TvgC REAL, Gamma REAL, LengthMeters REAL, WidthMeters REAL,
    HeightMeters REAL, AltitudeMeters REAL, NavOffsetXMeters REAL,
    NavOffsetYMeters REAL, NavOffsetLatitudeDegrees REAL,
    NavOffsetLongitudeDegrees REAL, NavOffsetDepthMeters REAL,
    NavOffsetHeadingDegrees REAL, NavOffsetPitchDegrees REAL,
    NavOffsetRollDegrees REAL, NavOffsetAftMeters REAL,
    NavOffsetStarboardMeters REAL, PingSequenceNumber INTEGER,
    SlantRangeDistanceMeters REAL, PositionCalculationDirectionOption INTEGER,
    TargetId INTEGER
);
CREATE TABLE MutableTag (Tag TEXT, TargetId INTEGER);
"""


def _make_target_file(
    path: Path,
    *,
    ping_seq: int = 105,
    channel: int = 0,
    slant_range_m: float = 5.0,
    mps: float = 0.1,
    window_seqs: list[int] | None = None,
    corrupt_seqs: list[int] | None = None,
    n_samples: int = 20,
    weighting_factor: int = 0,
    tag: str | None = None,
) -> None:
    if window_seqs is None:
        window_seqs = list(range(100, 110))
    corrupt_seqs = corrupt_seqs or []

    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute(
        "insert into Target values (1, 1, ?, 0)", (channel,)
    )
    for seq in list(window_seqs) + corrupt_seqs:
        samples = np.arange(n_samples, dtype=np.int16).tobytes()
        conn.execute(
            "insert into PingChannel values (?, 0, 0, 0, ?, ?, 0, ?, ?, 1.0, 0, 0, -71.0, 41.0, 0, 1)",
            (seq, mps, n_samples, weighting_factor, samples),
        )
    conn.execute(
        "insert into MutableTarget values (?, null, null, 0,0,0,0, null,null,null, null,"
        " 0,0,0,0,0,0,0,0,0,0, ?, ?, 0, 1)",
        (None, ping_seq, slant_range_m),
    )
    if tag is not None:
        conn.execute("insert into MutableTag values (?, 1)", (tag,))
    conn.commit()
    conn.close()


def test_reads_clean_target(tmp_path):
    _make_target_file(tmp_path / "Target-0001.target", ping_seq=105, channel=0, slant_range_m=1.0, mps=0.1)
    targets = read_discover_targets(tmp_path)
    assert len(targets) == 1
    t = targets[0]
    assert t.target_id == "0001"  # filename number, not internal TargetId
    assert t.source_format == "jsf_discover"
    assert t.ping == 105
    assert t.channel == 0
    assert t.sample == 10  # round(1.0 / 0.1)
    assert t.position_confidence == "verified"
    assert t.label is None
    assert t.latitude == 41.0
    assert t.longitude == -71.0
    assert t.extra_attributes["window_row"] == 5  # index of 105 in [100..109]


def test_filename_is_used_over_internal_target_id(tmp_path):
    # Every file's internal Target.TargetId is 1 (confirmed on real data) —
    # target_id must come from the filename, not collide across files.
    _make_target_file(tmp_path / "Target-0001.target")
    _make_target_file(tmp_path / "Target-0002.target")
    targets = read_discover_targets(tmp_path)
    ids = {t.target_id for t in targets}
    assert ids == {"0001", "0002"}


def test_corrupted_window_rows_are_excluded_and_confidence_downgraded(tmp_path):
    # Modeled precisely on the real Target-0002.target anomaly: the
    # target's own MutableTarget.PingSequenceNumber *and* its real window
    # rows are stored as (true_value + 2**32) — a uint32 rollover
    # artifact — while an unrelated, disjoint cluster of plausible-looking
    # (but wrong) sequence numbers is also mixed into the same file's
    # PingChannel table.
    rollover = 2**32
    true_window = list(range(4460, 4564))  # 104 rows, true target ping 4483
    true_target_seq = 4483
    unrelated_cluster = list(range(4846, 4940))  # 94 rows, nothing to do with this target
    _make_target_file(
        tmp_path / "Target-0002.target",
        ping_seq=true_target_seq + rollover,
        window_seqs=[s + rollover for s in true_window],
        corrupt_seqs=unrelated_cluster,
    )
    targets = read_discover_targets(tmp_path)
    t = targets[0]
    assert t.position_confidence == "approximate"
    assert t.extra_attributes["window_filtered"] is True
    assert t.extra_attributes["n_window_rows_excluded"] == len(unrelated_cluster)
    assert len(t.extra_attributes["window_ping_sequence"]) == len(true_window)
    assert t.ping == true_target_seq + rollover  # native field keeps the raw catalog value
    assert t.extra_attributes["window_row"] == true_window.index(true_target_seq)


def test_empty_description_and_tag_leave_label_none(tmp_path):
    _make_target_file(tmp_path / "Target-0001.target", tag=None)
    t = read_discover_targets(tmp_path)[0]
    assert t.label is None


def test_decode_target_window_shape_and_values(tmp_path):
    _make_target_file(
        tmp_path / "Target-0001.target",
        ping_seq=105,
        window_seqs=list(range(100, 110)),
        n_samples=8,
        weighting_factor=1,
    )
    t = read_discover_targets(tmp_path)[0]
    window = decode_target_window(t)
    assert window.shape == (10, 8)
    # weighting_factor=1 halves the raw int16 samples
    np.testing.assert_allclose(window[0], np.arange(8) * 0.5)


def test_target_ping_missing_from_own_window_is_unknown_confidence(tmp_path):
    _make_target_file(
        tmp_path / "Target-0003.target",
        ping_seq=999,  # not present in window_seqs at all
        window_seqs=list(range(100, 110)),
    )
    t = read_discover_targets(tmp_path)[0]
    assert t.position_confidence == "unknown"
    assert t.extra_attributes["window_row"] is None


def test_decode_and_correct_window_matches_manual_chain(tmp_path):
    from sidescanpy.export.orientation import orient_for_display
    from sidescanpy.export.target_window import decode_and_correct_window
    from sidescanpy.processing.radiometric import apply_tvg_to_amplitude

    _make_target_file(
        tmp_path / "Target-0001.target",
        ping_seq=105,
        channel=0,
        window_seqs=list(range(100, 110)),
        n_samples=8,
        weighting_factor=1,
        mps=0.1,
    )
    t = read_discover_targets(tmp_path)[0]

    window, corrected_target = decode_and_correct_window(t, freq_hz=1_495_000)

    expected = decode_target_window(t)
    expected = apply_tvg_to_amplitude(expected, t.extra_attributes["ping_meters_per_sample"], freq_hz=1_495_000)
    expected, expected_target = orient_for_display(expected, t)

    np.testing.assert_allclose(window, expected)
    assert corrected_target.sample == expected_target.sample
    assert corrected_target.extra_attributes["window_row"] == expected_target.extra_attributes["window_row"]


def test_unreadable_file_is_skipped_not_raised(tmp_path):
    bad = tmp_path / "Target-9999.target"
    bad.write_text("not a sqlite file")
    _make_target_file(tmp_path / "Target-0001.target")
    targets = read_discover_targets(tmp_path)
    assert len(targets) == 1
    assert targets[0].target_id == "0001"
