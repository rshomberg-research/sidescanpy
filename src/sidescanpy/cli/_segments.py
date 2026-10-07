"""
File output for ``sidescan-analyze segment``: one ``.ss.h5`` per clean
segment plus a JSON report of every segment.
"""

import json
import logging
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing.segments import Segment, split_by_segments

logger = logging.getLogger(__name__)


def write_segments(survey: SideScanSurvey, segments: list[Segment], output_dir: Path) -> list[Path]:
    """Write ``<label>.ss.h5`` for every clean segment; return the paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label, sub_survey in split_by_segments(survey, segments).items():
        out_path = output_dir / f"{label}.ss.h5"
        write_survey(sub_survey, out_path, overwrite=True)
        written.append(out_path)
    return written


def write_segments_report(segments: list[Segment], path: Path) -> None:
    """Write ``{"segments": [...]}`` for every segment, clean or excluded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"segments": [s.to_dict() for s in segments]}, f, indent=2)
