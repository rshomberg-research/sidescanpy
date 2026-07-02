"""
Format-dispatching helpers for sonar file ingestion.

Given a list of file paths, pick the right per-format reader/converter
by extension and produce a validated SideScanSurvey. Used by any script
or caller that wants "just give me a survey from these files" without
hand-rolling the .jsf/.sdf branch itself.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.jsf import convert as jsf_convert
from sidescanpy.io.jsf import read_jsf
from sidescanpy.io.sdf import convert as sdf_convert
from sidescanpy.io.sdf import read_sdf

logger = logging.getLogger(__name__)


def detect_format(input_paths: list[Path]) -> str:
    """Detect sonar file format from extension; raise on mixed extensions.

    Parameters
    ----------
    input_paths : list[Path]
        Candidate sonar files. All must share the same extension — this
        function is for the "merge these into one survey" case; batch
        callers processing files independently should call it once per
        file instead of once for a whole directory.

    Returns
    -------
    str
        One of: "jsf", "sdf".

    Raises
    ------
    ValueError
        If extensions are mixed, or the (single) extension isn't
        supported.
    """
    extensions = {p.suffix.lower() for p in input_paths}
    if len(extensions) > 1:
        raise ValueError(
            f"All input files must have the same format. "
            f"Got mixed extensions: {sorted(extensions)}"
        )
    ext = next(iter(extensions))
    if ext == ".jsf":
        return "jsf"
    if ext == ".sdf":
        return "sdf"
    raise ValueError(
        f"Unsupported format: '{ext}'. Supported: .jsf, .sdf"
    )


def read_and_convert(
    input_paths: list[Path],
    fmt: str,
    survey_name: str | None = None,
) -> SideScanSurvey:
    """Read sonar files and convert to a SideScanSurvey.

    Parameters
    ----------
    input_paths : list[Path]
        Paths to .jsf or .sdf files (all same format).
    fmt : str
        "jsf" or "sdf".
    survey_name : str, optional
        Survey name. Defaults to first input file's stem.

    Returns
    -------
    SideScanSurvey
    """
    if fmt == "jsf":
        all_messages = []
        for p in input_paths:
            logger.debug(f"  Reading JSF: {p}")
            msgs = read_jsf(p)
            all_messages.extend(msgs)
        return jsf_convert(
            all_messages,
            source_files=input_paths,
            survey_name=survey_name or input_paths[0].stem,
        )

    if fmt == "sdf":
        all_pages = []
        for p in input_paths:
            logger.debug(f"  Reading SDF: {p}")
            pages = read_sdf(p)
            all_pages.extend(pages)
        return sdf_convert(
            all_pages,
            source_files=input_paths,
            survey_name=survey_name or input_paths[0].stem,
        )

    raise ValueError(f"Unknown format: {fmt}")
