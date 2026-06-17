"""
Unit tests for sidescanpy.io.sdf.reader.

Builds synthetic SDF files (multi-page byte streams) and verifies the
reader correctly iterates pages, advances by the right offsets, yields
both sonar and unknown pages, and recovers from bad markers.

2026 rshomberg@uri.edu
"""

import logging
import struct

import numpy as np
import pytest

from sidescanpy.io.sdf.pages import (
    HEADER_TOTAL_BYTES,
    MARKER_BYTES,
    PAGE_HEADER_BYTES,
    SDFSonarPage,
    SDFUnknownPage,
)
from sidescanpy.io.sdf.reader import (
    read_sdf,
    read_sdf_header,
    read_sdf_iter,
)
from tests.unit.io.sdf.conftest import (
    MARKER,
    make_sdf_header_bytes,
    make_sdf_page_bytes,
)


# ============================================================
# EAGER READ
# ============================================================

@pytest.mark.unit
class TestEagerRead:
    def test_read_sdf_returns_list(self, tmp_path):
        path = tmp_path / "single.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=100, n_stbd=100))

        pages = read_sdf(path)
        assert isinstance(pages, list)
        assert len(pages) == 1

    def test_read_sdf_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.sdf"
        path.write_bytes(b"")
        pages = read_sdf(path)
        assert pages == []

    def test_read_sdf_two_sonar_pages(self, tmp_path):
        path = tmp_path / "two.sdf"
        page1 = make_sdf_page_bytes(n_port=100, n_stbd=100, page_version=3501)
        page2 = make_sdf_page_bytes(n_port=200, n_stbd=200, page_version=3502)
        path.write_bytes(page1 + page2)

        pages = read_sdf(path)
        assert len(pages) == 2
        assert pages[0].page_version == 3501
        assert pages[1].page_version == 3502
        assert all(isinstance(p, SDFSonarPage) for p in pages)


# ============================================================
# LAZY READ
# ============================================================

@pytest.mark.unit
class TestLazyRead:
    def test_read_sdf_iter_yields_pages(self, tmp_path):
        path = tmp_path / "iter.sdf"
        page1 = make_sdf_page_bytes(n_port=50, n_stbd=50, ping_number=10)
        page2 = make_sdf_page_bytes(n_port=50, n_stbd=50, ping_number=20)
        path.write_bytes(page1 + page2)

        gen = read_sdf_iter(path)
        first = next(gen)
        assert first.ping_number == 10
        second = next(gen)
        assert second.ping_number == 20

    def test_read_sdf_iter_terminates_at_eof(self, tmp_path):
        path = tmp_path / "single.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        gen = read_sdf_iter(path)
        next(gen)
        with pytest.raises(StopIteration):
            next(gen)


# ============================================================
# PAGE DATA
# ============================================================

@pytest.mark.unit
class TestPageData:
    def test_port_data_correct_size(self, tmp_path):
        path = tmp_path / "data.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=512, n_stbd=512))

        pages = read_sdf(path)
        assert len(pages[0].port_data) == 512

    def test_stbd_data_correct_size(self, tmp_path):
        path = tmp_path / "data.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=200, n_stbd=300))

        pages = read_sdf(path)
        assert len(pages[0].stbd_data) == 300

    def test_port_data_values_match_input(self, tmp_path):
        """The synthetic builder uses np.arange — verify recovery."""
        path = tmp_path / "values.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=100, n_stbd=100))

        pages = read_sdf(path)
        np.testing.assert_array_equal(
            pages[0].port_data,
            np.arange(100, dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            pages[0].stbd_data,
            np.arange(100, dtype=np.uint32) + 10000,
        )

    def test_port_data_dtype_uint32(self, tmp_path):
        path = tmp_path / "dtype.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        pages = read_sdf(path)
        assert pages[0].port_data.dtype == np.uint32
        assert pages[0].stbd_data.dtype == np.uint32

    def test_extension_bytes_empty_by_default(self, tmp_path):
        """No extension bytes in the synthetic builder by default."""
        path = tmp_path / "no_ext.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        pages = read_sdf(path)
        assert pages[0].extension_bytes == b""

    def test_extension_bytes_preserved(self, tmp_path):
        """If extension is present, capture it in extension_bytes."""
        ext = b"\xDE\xAD\xBE\xEF" * 5  # 20 bytes
        path = tmp_path / "ext.sdf"
        path.write_bytes(make_sdf_page_bytes(
            n_port=10, n_stbd=10, extension_bytes=ext,
        ))

        pages = read_sdf(path)
        assert pages[0].extension_bytes == ext


# ============================================================
# SONAR + UNKNOWN PAGE TYPES
# ============================================================

@pytest.mark.unit
class TestPageTypes:
    def test_unknown_page_yielded(self, tmp_path):
        """Non-sonar page versions yield SDFUnknownPage (not skipped)."""
        # Build an unknown-version page: header + 100 bytes of body
        header_bytes = make_sdf_header_bytes(
            page_version=9999,
            page_size=PAGE_HEADER_BYTES + 100,
        )
        marker = struct.pack("<I", MARKER)
        unknown_page = marker + header_bytes + b"\x00" * 100

        path = tmp_path / "unknown.sdf"
        path.write_bytes(unknown_page)

        pages = read_sdf(path)
        assert len(pages) == 1
        assert isinstance(pages[0], SDFUnknownPage)
        assert pages[0].page_version == 9999
        assert len(pages[0].body_bytes) == 100

    def test_mixed_sonar_and_unknown(self, tmp_path):
        sonar1 = make_sdf_page_bytes(
            n_port=10, n_stbd=10, ping_number=1, page_version=3501,
        )

        # Build unknown page
        unknown_header = make_sdf_header_bytes(
            page_version=9999,
            page_size=PAGE_HEADER_BYTES + 50,
        )
        marker = struct.pack("<I", MARKER)
        unknown_page = marker + unknown_header + b"\x00" * 50

        sonar2 = make_sdf_page_bytes(
            n_port=10, n_stbd=10, ping_number=2, page_version=3502,
        )

        path = tmp_path / "mixed.sdf"
        path.write_bytes(sonar1 + unknown_page + sonar2)

        pages = read_sdf(path)
        assert len(pages) == 3
        assert isinstance(pages[0], SDFSonarPage)
        assert pages[0].ping_number == 1
        assert isinstance(pages[1], SDFUnknownPage)
        assert pages[1].page_version == 9999
        assert isinstance(pages[2], SDFSonarPage)
        assert pages[2].ping_number == 2

    def test_all_sonar(self, tmp_path):
        path = tmp_path / "sonar.sdf"
        page1 = make_sdf_page_bytes(n_port=10, n_stbd=10, page_version=3501)
        page2 = make_sdf_page_bytes(n_port=10, n_stbd=10, page_version=3502)
        path.write_bytes(page1 + page2)

        pages = read_sdf(path)
        assert all(isinstance(p, SDFSonarPage) for p in pages)
        assert len(pages) == 2

# ============================================================
# MARKER VALIDATION AND RECOVERY
# ============================================================

@pytest.mark.unit
class TestMarkerRecovery:
    def test_bad_marker_logs_and_recovers(self, tmp_path, caplog):
        """A bad marker should log a warning and resume at the next valid one."""
        good_page = make_sdf_page_bytes(n_port=10, n_stbd=10, ping_number=1)
        # Insert 100 bytes of garbage before another good page
        garbage = b"\x00" * 100
        good_page2 = make_sdf_page_bytes(n_port=10, n_stbd=10, ping_number=2)

        path = tmp_path / "bad.sdf"
        path.write_bytes(good_page + garbage + good_page2)

        with caplog.at_level(logging.WARNING):
            pages = read_sdf(path)

        # Both good pages should be recovered
        assert len(pages) == 2
        assert pages[0].ping_number == 1
        assert pages[1].ping_number == 2
        # At least one warning should have been logged about the bad data
        assert any("marker" in r.message.lower() or "bad" in r.message.lower()
                   for r in caplog.records)

    MAX_MARKER_SCAN_BYTES = 1_048_576
    from typing import BinaryIO
    def _scan_for_next_marker(
        self,
        f: BinaryIO,
        file_size: int,
        max_scan: int = MAX_MARKER_SCAN_BYTES,
    ) -> int | None:
        """Scan forward from current position to find the next valid marker."""
        target = struct.pack("<I", MARKER)
        start_offset = f.tell()
        end_offset = min(start_offset + max_scan, file_size)

        chunk_size = 65536
        overlap = 3   # so markers spanning chunks aren't missed
        pos = start_offset

        while pos < end_offset:
            f.seek(pos)
            to_read = min(chunk_size, end_offset - pos)
            chunk = f.read(to_read)
            if not chunk:
                return None

            idx = chunk.find(target)
            if idx >= 0:
                marker_offset = pos + idx
                f.seek(marker_offset)
                return marker_offset

            # If chunk is smaller than the overlap, we've reached the end
            # of scannable space without finding a marker
            if len(chunk) <= overlap:
                break

            pos += len(chunk) - overlap

        return None
    
    @pytest.mark.unit
    def test_scan_terminates_on_pathological_size(self, tmp_path):
        """Marker scan must terminate even when scan range hits exact tail edge."""
        import time
        from sidescanpy.io.sdf.reader import _scan_for_next_marker

        # Construct a file where scan hits the chunk-boundary edge case
        data = b"\x00" * 1_048_576
        path = tmp_path / "edge.sdf"
        path.write_bytes(data)

        with open(path, "rb") as f:
            start = time.time()
            result = _scan_for_next_marker(f, len(data))
            elapsed = time.time() - start

        assert result is None
        assert elapsed < 1.0, f"scan took {elapsed:.1f}s — should be fast"

    def test_only_bad_data_returns_empty(self, tmp_path, caplog):
        """A file with no valid markers should return empty (no exception)."""
        path = tmp_path / "all_bad.sdf"
        path.write_bytes(b"\x00" * 1000)

        with caplog.at_level(logging.WARNING):
            pages = read_sdf(path)

        assert pages == []


# ============================================================
# READ_SDF_HEADER (LOW-LEVEL HELPER)
# ============================================================

@pytest.mark.unit
class TestReadSdfHeader:
    def test_reads_marker_and_header(self, tmp_path):
        """read_sdf_header consumes 516 bytes and returns parsed header."""
        path = tmp_path / "single.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        from sidescanpy.io.sdf.pages import SDFPageHeader
        with open(path, "rb") as f:
            header = read_sdf_header(f)

        assert isinstance(header, SDFPageHeader)
        assert header.ping_number == 1000  # default

    def test_advances_file_pointer_516_bytes(self, tmp_path):
        path = tmp_path / "single.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        with open(path, "rb") as f:
            read_sdf_header(f)
            assert f.tell() == HEADER_TOTAL_BYTES

    def test_invalid_marker_raises(self, tmp_path):
        """read_sdf_header raises on bad marker (recovery is reader-level)."""
        # Create 516 bytes with bad marker
        bad_data = b"\x00" * HEADER_TOTAL_BYTES
        path = tmp_path / "bad.sdf"
        path.write_bytes(bad_data)

        with open(path, "rb") as f:
            with pytest.raises(ValueError, match="marker"):
                read_sdf_header(f)

    def test_truncated_input_raises(self, tmp_path):
        """Less than 516 bytes raises ValueError."""
        path = tmp_path / "trunc.sdf"
        path.write_bytes(b"\xFF\xFF\xFF\xFF" + b"\x00" * 100)

        with open(path, "rb") as f:
            with pytest.raises(ValueError, match="EOF|bytes"):
                read_sdf_header(f)


# ============================================================
# ERROR HANDLING (NON-RECOVERY)
# ============================================================

@pytest.mark.unit
class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_sdf(tmp_path / "nonexistent.sdf")

    def test_truncated_data_section_terminates(self, tmp_path, caplog):
        """A page header claiming more data than exists should terminate
        gracefully with a warning, not crash."""
        # Build a header claiming a normal-size page
        header = make_sdf_header_bytes(
            page_version=3502,
            page_size=20000,  # claims a large page
            num_samples=2500,
        )
        # But only provide a tiny fake data section
        marker = struct.pack("<I", MARKER)
        tiny_data = b"\x00" * 50

        path = tmp_path / "truncated.sdf"
        path.write_bytes(marker + header + tiny_data)

        with caplog.at_level(logging.WARNING):
            pages = read_sdf(path)

        # Should not raise; just stop early with warning
        assert pages == []
        assert any("file_size" in r.message.lower() or "eof" in r.message.lower()
                   for r in caplog.records)


# ============================================================
# OFFSET ADVANCEMENT
# ============================================================

@pytest.mark.unit
class TestOffsetAdvancement:
    def test_reading_advances_through_pages(self, tmp_path):
        """Reader should correctly advance past each page's full extent."""
        p1 = make_sdf_page_bytes(n_port=100, n_stbd=100, ping_number=1)
        p2 = make_sdf_page_bytes(n_port=500, n_stbd=500, ping_number=2)
        p3 = make_sdf_page_bytes(n_port=200, n_stbd=200, ping_number=3)

        path = tmp_path / "varied.sdf"
        path.write_bytes(p1 + p2 + p3)

        pages = read_sdf(path)
        assert len(pages) == 3
        assert [p.ping_number for p in pages] == [1, 2, 3]
        assert [len(p.port_data) for p in pages] == [100, 500, 200]

# ============================================================
# HIGH-BIT MASKING (Klein metadata bits)
# ============================================================

@pytest.mark.unit
class TestHighBitMasking:
    def test_normal_data_unchanged(self, tmp_path):
        """Samples with no high bits set should pass through unchanged."""
        path = tmp_path / "normal.sdf"
        path.write_bytes(make_sdf_page_bytes(n_port=10, n_stbd=10))

        pages = read_sdf(path)
        assert len(pages) == 1
        # Default builder uses np.arange — all values < 16 bits
        np.testing.assert_array_equal(
            pages[0].port_data,
            np.arange(10, dtype=np.uint32),
        )

    def test_bit_17_masked_off(self, tmp_path):
        """Samples with bit 17 set should be masked to low 16 bits."""
        # Build a page with one sample having bit 17 set
        # (raw value = 131,200 = 0x00020080; low16 = 0x80 = 128)
        n_samples = 5
        port_data = np.array([100, 200, 131_200, 400, 500], dtype=np.uint32)
        stbd_data = np.array([10, 20, 30, 40, 50], dtype=np.uint32)

        from tests.unit.io.sdf.conftest import (
            MARKER, PAGE_HEADER_BYTES, make_sdf_data_section,
            make_sdf_header_bytes,
        )
        data_section = make_sdf_data_section(port_data, stbd_data)
        page_size = PAGE_HEADER_BYTES + len(data_section)
        header_bytes = make_sdf_header_bytes(
            page_version=3502,
            page_size=page_size,
            num_samples=n_samples,
        )
        marker = struct.pack("<I", MARKER)

        path = tmp_path / "high_bit.sdf"
        path.write_bytes(marker + header_bytes + data_section)

        pages = read_sdf(path)
        assert len(pages) == 1
        # Sample at index 2 had bit 17 set; low16 = 128
        np.testing.assert_array_equal(
            pages[0].port_data,
            np.array([100, 200, 128, 400, 500], dtype=np.uint32),
        )

    def test_multiple_high_bits_logged(self, tmp_path, caplog):
        """Multiple high-bit samples are logged at DEBUG level."""
        import logging

        # 3 samples have bit 17 set
        port_data = np.array(
            [100, 131_200, 200, 131_200, 300, 131_200, 400],
            dtype=np.uint32,
        )
        stbd_data = np.zeros(7, dtype=np.uint32)

        from tests.unit.io.sdf.conftest import (
            MARKER, PAGE_HEADER_BYTES, make_sdf_data_section,
            make_sdf_header_bytes,
        )
        data_section = make_sdf_data_section(port_data, stbd_data)
        page_size = PAGE_HEADER_BYTES + len(data_section)
        header_bytes = make_sdf_header_bytes(
            page_version=3502,
            page_size=page_size,
            num_samples=7,
        )
        marker = struct.pack("<I", MARKER)

        path = tmp_path / "many_high.sdf"
        path.write_bytes(marker + header_bytes + data_section)

        with caplog.at_level(logging.DEBUG):
            pages = read_sdf(path)

        # Verify masking
        np.testing.assert_array_equal(
            pages[0].port_data,
            np.array([100, 128, 200, 128, 300, 128, 400], dtype=np.uint32),
        )

        # Verify debug log mentions the count
        assert any(
            "masked 3" in r.message.lower() or "3 port" in r.message.lower()
            for r in caplog.records
        )

    def test_stbd_also_masked(self, tmp_path):
        """Starboard channel is also masked (independently of port)."""
        port_data = np.zeros(5, dtype=np.uint32)
        stbd_data = np.array(
            [10, 131_200, 30, 40, 50], dtype=np.uint32
        )

        from tests.unit.io.sdf.conftest import (
            MARKER, PAGE_HEADER_BYTES, make_sdf_data_section,
            make_sdf_header_bytes,
        )
        data_section = make_sdf_data_section(port_data, stbd_data)
        page_size = PAGE_HEADER_BYTES + len(data_section)
        header_bytes = make_sdf_header_bytes(
            page_version=3502,
            page_size=page_size,
            num_samples=5,
        )
        marker = struct.pack("<I", MARKER)

        path = tmp_path / "stbd_high.sdf"
        path.write_bytes(marker + header_bytes + data_section)

        pages = read_sdf(path)
        np.testing.assert_array_equal(
            pages[0].stbd_data,
            np.array([10, 128, 30, 40, 50], dtype=np.uint32),
        )

    def test_max_low16_value_preserved(self, tmp_path):
        """Value 0xFFFF (max uint16) is preserved unchanged."""
        port_data = np.array([0xFFFF, 100], dtype=np.uint32)
        stbd_data = np.zeros(2, dtype=np.uint32)

        from tests.unit.io.sdf.conftest import (
            MARKER, PAGE_HEADER_BYTES, make_sdf_data_section,
            make_sdf_header_bytes,
        )
        data_section = make_sdf_data_section(port_data, stbd_data)
        page_size = PAGE_HEADER_BYTES + len(data_section)
        header_bytes = make_sdf_header_bytes(
            page_version=3502,
            page_size=page_size,
            num_samples=2,
        )
        marker = struct.pack("<I", MARKER)

        path = tmp_path / "max_lo.sdf"
        path.write_bytes(marker + header_bytes + data_section)

        pages = read_sdf(path)
        # 0xFFFF should pass through unchanged (no high bits)
        assert pages[0].port_data[0] == 0xFFFF