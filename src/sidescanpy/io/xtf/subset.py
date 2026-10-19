"""
Cut a run of pings out of an XTF file, byte for byte.

:func:`write_subset` copies the file header and every packet from the
first kept sonar ping up to (not including) the sonar ping after the last
kept one, unchanged. Non-sonar packets inside that span (navigation,
attitude, notes) are kept; those before or after it are not. Used to make
small test fixtures from public recordings.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.io.xtf.reader import iter_packets, read_xtf_header
from sidescanpy.io.xtf.structs import HEADER_TYPE_SONAR

logger = logging.getLogger(__name__)


def write_subset(src: str | Path, dst: str | Path, *, n_pings: int, start_ping: int = 0) -> int:
    """Write sonar pings ``start_ping`` to ``start_ping + n_pings`` of ``src`` to ``dst``.

    Parameters
    ----------
    src, dst : str or Path
        Source XTF file and output file (overwritten).
    n_pings : int
        Sonar packets (pings) to keep.
    start_ping : int, default 0
        Index of the first sonar packet kept.

    Returns
    -------
    int
        Number of sonar pings written (fewer than ``n_pings`` if the file
        ends first).
    """
    if n_pings < 1 or start_ping < 0:
        raise ValueError("n_pings must be >= 1 and start_ping >= 0")
    src, dst = Path(src), Path(dst)
    header = read_xtf_header(src)
    begin = end = None
    seen = 0
    for pkt in iter_packets(src):
        if pkt.header_type != HEADER_TYPE_SONAR:
            continue
        if seen == start_ping:
            begin = pkt.offset
        if seen == start_ping + n_pings:
            end = pkt.offset
            break
        seen += 1
    if begin is None:
        raise ValueError(f"{src}: fewer than {start_ping + 1} sonar pings")
    written = min(seen, start_ping + n_pings) - start_ping
    with open(src, "rb") as f:
        head = f.read(header.size)
        f.seek(begin)
        body = f.read() if end is None else f.read(end - begin)
    dst.write_bytes(head + body)
    logger.debug("%s: wrote %d sonar pings (%d bytes) to %s", src.name, written, len(head) + len(body), dst)
    return written
