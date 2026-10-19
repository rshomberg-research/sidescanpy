"""
Cut a run of pings out of a JSF file, byte for byte.

:func:`write_subset` copies every message from the first sonar data
message (type 80) of the first kept ping up to (not including) the first
sonar data message of the ping after the last kept one, unchanged. A ping
is a distinct ``ping_number`` among sonar data messages, so every channel
and subsystem of a kept ping is kept. Other messages inside that span
(navigation, attitude, system information) are kept; those before or
after it are not. Used to make small test fixtures from public recordings.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.io.jsf.messages import JSFMessageHeader, SonarDataMessage

logger = logging.getLogger(__name__)


def write_subset(src: str | Path, dst: str | Path, *, n_pings: int, start_ping: int = 0) -> int:
    """Write pings ``start_ping`` to ``start_ping + n_pings`` of ``src`` to ``dst``.

    Parameters
    ----------
    src, dst : str or Path
        Source JSF file and output file (overwritten).
    n_pings : int
        Pings (distinct sonar ``ping_number`` values) to keep.
    start_ping : int, default 0
        Index of the first ping kept, counting distinct ping numbers in
        file order.

    Returns
    -------
    int
        Number of pings written (fewer than ``n_pings`` if the file ends
        first).
    """
    if n_pings < 1 or start_ping < 0:
        raise ValueError("n_pings must be >= 1 and start_ping >= 0")
    src, dst = Path(src), Path(dst)
    size = JSFMessageHeader.STRUCT.size
    pings: list[int] = []
    begin = end = None
    with open(src, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.read(size)
            if len(raw) < size:
                break
            header = JSFMessageHeader.from_bytes(raw)
            body = f.read(header.message_size)
            if len(body) < header.message_size:
                logger.warning("%s: truncated message at byte %d; stopping", src.name, offset)
                break
            if header.message_type != SonarDataMessage.MESSAGE_TYPE:
                continue
            ping = SonarDataMessage.from_bytes(header, body).ping_number
            if ping in pings:
                continue
            pings.append(ping)
            if len(pings) == start_ping + 1:
                begin = offset
            if len(pings) == start_ping + n_pings + 1:
                end = offset
                break
        if begin is None:
            raise ValueError(f"{src}: fewer than {start_ping + 1} pings")
        f.seek(begin)
        data = f.read() if end is None else f.read(end - begin)
    dst.write_bytes(data)
    written = min(len(pings), start_ping + n_pings) - start_ping
    logger.debug("%s: wrote %d pings (%d bytes) to %s", src.name, written, len(data), dst)
    return written
