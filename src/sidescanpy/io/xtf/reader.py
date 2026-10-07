"""XTF reader stub. See :mod:`sidescanpy.io.xtf`."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_xtf(path: str | Path):
    """Read an XTF file into a survey (not yet implemented).

    Raises
    ------
    NotImplementedError
        Always. An XTF reader is planned; no test data exists yet.
    """
    raise NotImplementedError(
        "XTF reading is not implemented yet (planned; no test data available)."
    )
