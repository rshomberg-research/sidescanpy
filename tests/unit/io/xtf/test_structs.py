"""
Unit tests for sidescanpy.io.xtf.structs.

The dtypes are checked field for field (name, offset, size) against
pyxtf's ctypes structures when pyxtf is installed.

2026 rshomberg@uri.edu
"""

import ctypes

import numpy as np
import pytest

from sidescanpy.io.xtf.structs import (
    CHAN_INFO_DTYPE,
    FILE_HEADER_DTYPE,
    PACKET_START_DTYPE,
    PING_CHAN_HEADER_DTYPE,
    PING_HEADER_DTYPE,
    sample_dtype,
)
from tests.unit.io.xtf.conftest import chan_info


class TestSizes:
    @pytest.mark.parametrize(
        "dtype, size",
        [
            (CHAN_INFO_DTYPE, 128),
            (FILE_HEADER_DTYPE, 1024),
            (PACKET_START_DTYPE, 14),
            (PING_HEADER_DTYPE, 256),
            (PING_CHAN_HEADER_DTYPE, 64),
        ],
    )
    def test_itemsize_matches_spec(self, dtype, size):
        assert dtype.itemsize == size

    def test_ping_header_starts_with_packet_start(self):
        for name in PACKET_START_DTYPE.names:
            assert PING_HEADER_DTYPE.fields[name][1] == PACKET_START_DTYPE.fields[name][1]


def _ctypes_layout(cls):
    return [(name, getattr(cls, name).offset, getattr(cls, name).size) for name, *_ in cls._fields_]


def _dtype_layout(dtype):
    return [(name, dtype.fields[name][1], dtype.fields[name][0].itemsize) for name in dtype.names]


@pytest.fixture(scope="module")
def pyxtf_ctypes():
    return pytest.importorskip("pyxtf.xtf_ctypes")


class TestAgainstPyxtf:
    def test_chan_info(self, pyxtf_ctypes):
        assert _dtype_layout(CHAN_INFO_DTYPE) == _ctypes_layout(pyxtf_ctypes.XTFChanInfo)

    def test_file_header(self, pyxtf_ctypes):
        assert _dtype_layout(FILE_HEADER_DTYPE) == _ctypes_layout(pyxtf_ctypes.XTFFileHeader)
        assert ctypes.sizeof(pyxtf_ctypes.XTFFileHeader) == FILE_HEADER_DTYPE.itemsize

    def test_packet_start(self, pyxtf_ctypes):
        assert _dtype_layout(PACKET_START_DTYPE) == _ctypes_layout(pyxtf_ctypes.XTFPacketStart)

    def test_ping_header(self, pyxtf_ctypes):
        expected = _ctypes_layout(pyxtf_ctypes.XTFPacketStart) + _ctypes_layout(
            pyxtf_ctypes.XTFPingHeader
        )
        assert _dtype_layout(PING_HEADER_DTYPE) == expected

    def test_ping_chan_header(self, pyxtf_ctypes):
        assert _dtype_layout(PING_CHAN_HEADER_DTYPE) == _ctypes_layout(
            pyxtf_ctypes.XTFPingChanHeader
        )

    def test_ping_header_values_match_ctypes_read(self, pyxtf_ctypes):
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, PING_HEADER_DTYPE.itemsize, dtype=np.uint8).tobytes()
        ours = np.frombuffer(raw, PING_HEADER_DTYPE)[0]
        theirs = pyxtf_ctypes.XTFPingHeader.from_buffer_copy(raw)
        for name in ("PingNumber", "SensorXcoordinate", "SensorHeading", "OptionalOffset",
                     "FishPositionDeltaY", "CableOutHundredths", "ShipDepth"):
            a, b = ours[name], getattr(theirs, name)
            assert a == b or (np.isnan(a) and np.isnan(b)), name


class TestSampleDtype:
    @pytest.mark.parametrize(
        "fmt, bps, expected",
        [
            (0, 1, "u1"),
            (0, 2, "<u2"),
            (0, 4, "<u4"),
            (2, 4, "<u4"),
            (3, 2, "<u2"),
            (5, 4, "<f4"),
            (8, 1, "u1"),
        ],
    )
    def test_supported_formats(self, fmt, bps, expected):
        assert sample_dtype(chan_info(1, 100.0, bps, fmt)) == np.dtype(expected)

    @pytest.mark.parametrize("fmt, bps", [(1, 4), (4, 4), (0, 3), (0, 8)])
    def test_unsupported_raises(self, fmt, bps):
        with pytest.raises(ValueError, match="Unsupported"):
            sample_dtype(chan_info(1, 100.0, bps, fmt))

    def test_format_overrides_disagreeing_bytes_per_sample(self, caplog):
        assert sample_dtype(chan_info(1, 100.0, 2, 8)) == np.dtype("u1")
        assert "disagrees" in caplog.text
