"""
Cut a run of pings out of a Humminbird recording, byte for byte.

:func:`write_subset` copies the ``.DAT`` file unchanged and, from every
``.SON`` file, the ping frames recorded between the first and the last
kept side imaging ping, unchanged. The ``.IDX`` files are not written
(the reader does not use them), and the ``.DAT`` still states the whole
recording's ping count and duration. Used to make small test fixtures
from public recordings.

2026 rshomberg@uri.edu
"""

import logging
import shutil
from pathlib import Path

import numpy as np

from sidescanpy.io.humminbird.reader import (
    BEAM_PORT,
    BEAM_STARBOARD,
    read_humminbird,
    son_directory,
)

logger = logging.getLogger(__name__)


def write_subset(src: str | Path, dst: str | Path, *, n_pings: int, start_ping: int = 0) -> int:
    """Write side imaging pings ``start_ping`` to ``start_ping + n_pings`` of ``src`` to ``dst``.

    Parameters
    ----------
    src, dst : str or Path
        Source and output ``.DAT`` files. The output's SON directory
        (``dst`` without its suffix) is created; SON files already in it
        are overwritten.
    n_pings : int
        Side imaging pings to keep, counted on the port beam (the
        starboard beam if there is no port beam).
    start_ping : int, default 0
        Index of the first kept side imaging ping.

    Returns
    -------
    int
        Number of side imaging pings written (fewer than ``n_pings`` if
        the recording ends first).
    """
    if n_pings < 1 or start_ping < 0:
        raise ValueError("n_pings must be >= 1 and start_ping >= 0")
    src, dst = Path(src), Path(dst)
    rec = read_humminbird(src)
    side = rec.beams.get(BEAM_PORT) or rec.beams.get(BEAM_STARBOARD)
    if side is None:
        raise ValueError(f"{src}: no side imaging beam (B002/B003)")
    if start_ping >= side.n_pings:
        raise ValueError(f"{src}: fewer than {start_ping + 1} side imaging pings")
    stop = min(side.n_pings, start_ping + n_pings)
    t0 = side.fields["time_ms"][start_ping]
    t1 = side.fields["time_ms"][stop - 1]
    out_dir = dst.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    src_dir = son_directory(src)
    for beam in rec.beams.values():
        keep = np.flatnonzero((beam.fields["time_ms"] >= t0) & (beam.fields["time_ms"] <= t1))
        name = beam.path.relative_to(src_dir)
        raw = beam.path.read_bytes()
        chunks = [raw[beam.frame_offset[i]: beam.sample_offset[i] + beam.n_samples[i]] for i in keep]
        (out_dir / name).write_bytes(b"".join(chunks))
        logger.debug("%s: wrote %d frames to %s", beam.path.name, keep.size, out_dir / name)
    return int(stop - start_ping)
