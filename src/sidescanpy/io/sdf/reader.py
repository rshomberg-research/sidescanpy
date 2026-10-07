"""
Klein SDF file reader.

Iterates pages in a Klein SDF file. Each page has:
- A 4-byte sync marker (always 0xFFFFFFFF)
- A 512-byte typed header (described by SDFPageHeader)
- Variable-length body (sonar samples for sonar pages, raw bytes otherwise)

The reader handles file-level concerns: marker validation, header parsing,
page-version dispatch, and sample data extraction. The pages module defines
what a page IS (typed structures); the reader handles WHERE pages are.

Three entry points:
- read_sdf_header(f): low-level, reads marker + header at current position
- read_sdf(path): eager, returns list of all pages (sonar + unknown)
- read_sdf_iter(path): lazy, yields pages one at a time

On a bad marker, logs a warning and scans forward (up to 1 MB) looking
for the next valid marker. This is robust to small corruption, mid-page
truncation, or our own header parsing errors.

For sonar pages, the body is parsed into port and starboard sample arrays,
with any remaining bytes preserved in `extension_bytes` for future decoding.
For non-sonar pages, the entire body is preserved in `body_bytes`.

2026 rshomberg@uri.edu
"""

import logging
import struct
from pathlib import Path
from typing import BinaryIO, Iterator, Union

import numpy as np

from sidescanpy.io.sdf.pages import (
    HEADER_TOTAL_BYTES,
    MARKER,
    MARKER_BYTES,
    PAGE_HEADER_BYTES,
    PAGE_VERSIONS_SONAR,
    SDFPageHeader,
    SDFSonarPage,
    SDFUnknownPage,
)

logger = logging.getLogger(__name__)


# Maximum bytes to scan when searching for the next marker after a parsing
# error or invalid marker. 1 MB is enough to skip past most corruption,
# while not so large that we'd waste minutes scanning a fully-broken file.
MAX_MARKER_SCAN_BYTES = 1_048_576


# Type alias: pages iterated from a file are either sonar or unknown
SDFPage = Union[SDFSonarPage, SDFUnknownPage]


# ============================================================
# PUBLIC API
# ============================================================

def read_sdf_header(f: BinaryIO) -> SDFPageHeader:
    """Read marker + header from the current file position.

    Validates the marker, then parses the next 512 bytes as an
    SDFPageHeader. Advances the file pointer by 516 bytes total.
    Caller is responsible for handling sonar/non-sonar dispatch
    and for reading the page body.

    Parameters
    ----------
    f : BinaryIO
        Open file with pointer at the start of a page (marker position).

    Returns
    -------
    SDFPageHeader

    Raises
    ------
    ValueError
        If the marker is invalid, the input is truncated, or header
        parsing fails.
    """
    marker_bytes = f.read(MARKER_BYTES)
    if len(marker_bytes) < MARKER_BYTES:
        raise ValueError(f"EOF reading marker (got {len(marker_bytes)} bytes)")
    marker = struct.unpack("<I", marker_bytes)[0]
    if marker != MARKER:
        raise ValueError(
            f"Invalid SDF marker: 0x{marker:08X} (expected 0x{MARKER:08X})"
        )

    header_bytes = f.read(PAGE_HEADER_BYTES)
    if len(header_bytes) < PAGE_HEADER_BYTES:
        raise ValueError(
            f"EOF reading header (got {len(header_bytes)}/{PAGE_HEADER_BYTES} bytes)"
        )

    return SDFPageHeader.from_bytes(header_bytes)


def read_sdf(path: Path | str) -> list[SDFPage]:
    """Read all pages from a Klein SDF file (eager).

    Includes both sonar pages (SDFSonarPage) and non-sonar pages
    (SDFUnknownPage). Caller can filter by type if needed.

    Parameters
    ----------
    path : Path or str
        Path to a Klein SDF file.

    Returns
    -------
    list[SDFPage]
        All pages in file order.
    """
    return list(read_sdf_iter(path))


def read_sdf_iter(path: Path | str) -> Iterator[SDFPage]:
    """Iterate pages from a Klein SDF file (lazy).

    Yields both sonar pages (SDFSonarPage with port/stbd data) and
    non-sonar pages (SDFUnknownPage with raw body bytes). On bad
    markers or parsing errors, logs a warning and scans forward to
    find the next valid marker.

    Parameters
    ----------
    path : Path or str
        Path to a Klein SDF file.

    Yields
    ------
    SDFPage
        Either SDFSonarPage or SDFUnknownPage.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SDF file not found: {path}")

    file_size = path.stat().st_size
    logger.info(f"Reading SDF file: {path} ({file_size:,} bytes)")

    n_total = 0
    n_sonar = 0
    n_unknown = 0
    n_recovered = 0

    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            if offset >= file_size:
                break

            # Try to read marker + header at current position
            try:
                header = read_sdf_header(f)
            except ValueError as e:
                # Bad marker or truncated header. Scan forward to the next
                # valid marker.
                f.seek(offset)
                new_offset = _scan_for_next_marker(f, file_size)
                if new_offset is None:
                    logger.warning(
                        f"No valid marker found within {MAX_MARKER_SCAN_BYTES:,} "
                        f"bytes after offset {offset}. Stopping iteration."
                    )
                    break
                else:
                    skipped = new_offset - offset
                    logger.warning(
                        f"Bad data at offset {offset} ({e}). "
                        f"Resuming at next marker at offset {new_offset} "
                        f"(skipped {skipped:,} bytes)."
                    )
                    n_recovered += 1
                    f.seek(new_offset)
                    continue

            n_total += 1

            # Validate page size sanity
            full_page_size = header.page_size + MARKER_BYTES
            if full_page_size < HEADER_TOTAL_BYTES:
                logger.warning(
                    f"Page at offset {offset} claims size {header.page_size} "
                    f"which is smaller than header. Skipping ahead."
                )
                # Try to recover by scanning forward
                new_offset = _scan_for_next_marker(f, file_size)
                if new_offset is None:
                    break
                f.seek(new_offset)
                continue

            body_size = full_page_size - HEADER_TOTAL_BYTES
            page_end_offset = offset + full_page_size

            if page_end_offset > file_size:
                logger.warning(
                    f"Page at offset {offset} claims to extend past EOF "
                    f"(end={page_end_offset:,}, file_size={file_size:,}). "
                    f"Stopping iteration."
                )
                break

            # Dispatch by page version
            if header.page_version in PAGE_VERSIONS_SONAR:
                # Read sonar page body
                try:
                    body_bytes = f.read(body_size)
                    if len(body_bytes) < body_size:
                        logger.warning(
                            f"Truncated body at offset {offset} "
                            f"(got {len(body_bytes)}/{body_size}). Stopping."
                        )
                        break

                    port_data, stbd_data, extension_bytes = _parse_sonar_body(
                        body_bytes, header
                    )
                    page = SDFSonarPage(
                        header=header,
                        port_data=port_data,
                        stbd_data=stbd_data,
                        extension_bytes=extension_bytes,
                    )
                    n_sonar += 1
                    yield page

                except (ValueError, struct.error) as e:
                    logger.warning(
                        f"Failed to parse sonar body at offset {offset}: {e}. "
                        f"Skipping page."
                    )
                    f.seek(page_end_offset)
                    continue

            else:
                # Non-sonar page: preserve raw body bytes
                body_bytes = f.read(body_size)
                if len(body_bytes) < body_size:
                    logger.warning(
                        f"Truncated body at offset {offset} "
                        f"(got {len(body_bytes)}/{body_size}). Stopping."
                    )
                    break

                page = SDFUnknownPage(header=header, body_bytes=body_bytes)
                n_unknown += 1
                yield page

    logger.info(
        f"Read SDF: {n_sonar} sonar, {n_unknown} unknown, "
        f"{n_total} total, {n_recovered} marker recoveries"
    )


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _scan_for_next_marker(
    f: BinaryIO,
    file_size: int,
    max_scan: int = MAX_MARKER_SCAN_BYTES,
    ) -> int | None:
    """Scan forward from the current position to find the next valid marker.

    Reads the file in chunks and searches for the 4-byte marker pattern.
    Returns the byte offset of the marker, or None if not found within
    `max_scan` bytes.

    Parameters
    ----------
    f : BinaryIO
        Open file. Pointer is restored on return.
    file_size : int
        Total size of the file in bytes (for bounds checking).
    max_scan : int
        Maximum bytes to scan before giving up.

    Returns
    -------
    int or None
        Byte offset of the next marker, or None.
    """
    target = struct.pack("<I", MARKER)
    start_offset = f.tell()
    end_offset = min(start_offset + max_scan, file_size)

    chunk_size = 65536
    overlap = 3
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

        if len(chunk) <= overlap:
            break

        pos += len(chunk) - overlap

    return None


def _parse_sonar_body(
    body_bytes: bytes,
    header: SDFPageHeader,
) -> tuple[np.ndarray, np.ndarray, bytes]:
    """Parse a sonar page body into port/stbd arrays + extension bytes.

    Body layout per Klein spec:
        [n_port (uint32)]
        [port_data (n_port × uint32)]
        [n_stbd (uint32)]
        [stbd_data (n_stbd × uint32)]
        [extension_bytes (variable, optional)]

    Sample masking:
        Klein samples are stored as uint32, but only the low 16 bits
        carry amplitude information. Bit 17 is occasionally set as a
        metadata flag (purpose undocumented; ~0.001% of samples in
        observed Klein 4900 data). We mask to the low 16 bits so the
        amplitude is clean uint32 values 0-65535.

    Parameters
    ----------
    body_bytes : bytes
        Page body (after the 516-byte header section).
    header : SDFPageHeader
        Parsed header (used for sanity checks).

    Returns
    -------
    port_data, stbd_data : np.ndarray (uint32, low 16 bits only)
    extension_bytes : bytes
        Anything after the stbd array; empty if no extension.

    Raises
    ------
    ValueError
        On structural parsing failure (truncated counts or arrays).
    """
    body_size = len(body_bytes)
    if body_size < 8:
        raise ValueError(f"Body too small for port+stbd headers: {body_size}")

    # Port count + data
    n_port = struct.unpack_from("<I", body_bytes, 0)[0]
    port_byte_size = n_port * 4
    if 4 + port_byte_size > body_size:
        raise ValueError(
            f"Port array claims {port_byte_size} bytes but body has "
            f"{body_size - 4} after count"
        )

    port_raw = body_bytes[4:4 + port_byte_size]
    port_u32 = np.frombuffer(port_raw, dtype="<u4").copy()

    # Mask high 16 bits (Klein metadata flags; not amplitude data)
    port_high_count = int((port_u32 > 0xFFFF).sum())
    if port_high_count > 0:
        logger.debug(
            f"Masked {port_high_count} port sample(s) with metadata bits "
            f"at ping {header.ping_number}"
        )
    port_data = (port_u32 & 0xFFFF).astype(np.uint32)

    # Sanity check vs header.num_samples
    if n_port != header.num_samples:
        logger.warning(
            f"Port n_samples mismatch at ping {header.ping_number}: "
            f"header={header.num_samples}, body={n_port}"
        )

    # Stbd count + data
    stbd_offset = 4 + port_byte_size
    if stbd_offset + 4 > body_size:
        raise ValueError(
            f"Body has no room for stbd count (offset={stbd_offset}, "
            f"body_size={body_size})"
        )

    n_stbd = struct.unpack_from("<I", body_bytes, stbd_offset)[0]
    stbd_byte_size = n_stbd * 4
    if stbd_offset + 4 + stbd_byte_size > body_size:
        raise ValueError(
            f"Stbd array claims {stbd_byte_size} bytes but body only has "
            f"{body_size - (stbd_offset + 4)} remaining"
        )

    stbd_raw = body_bytes[stbd_offset + 4:stbd_offset + 4 + stbd_byte_size]
    stbd_u32 = np.frombuffer(stbd_raw, dtype="<u4").copy()

    # Mask high 16 bits
    # HACK: I cannot find anything about this in the spec sheet, but it appears to be the case
    stbd_high_count = int((stbd_u32 > 0xFFFF).sum())
    if stbd_high_count > 0:
        logger.debug(
            f"Masked {stbd_high_count} stbd sample(s) with metadata bits "
            f"at ping {header.ping_number}"
        )
    stbd_data = (stbd_u32 & 0xFFFF).astype(np.uint32)

    if n_stbd != header.num_samples:
        logger.warning(
            f"Stbd n_samples mismatch at ping {header.ping_number}: "
            f"header={header.num_samples}, body={n_stbd}"
        )

    # Extension: anything left over after port + stbd
    ext_start = stbd_offset + 4 + stbd_byte_size
    extension_bytes = body_bytes[ext_start:]

    return port_data, stbd_data, extension_bytes
