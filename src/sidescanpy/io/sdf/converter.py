"""
Convert Klein SDF pages into a SideScanSurvey.

The converter takes pages (typically from read_sdf or read_sdf_iter) and
builds a validated SideScanSurvey with:

- One SideScanFrequencyChannel per distinct center_freq_hz
- Per-ping fields (latitude, longitude, heading, pitch, roll, depth,
  altitude, fathometer_reading) extracted from each page's header
- Navigation aux record from per-page (timestamp, lat, lon)
- Attitude aux record from per-page (timestamp, heading, pitch, roll)
- Empty DVL record (Klein doesn't have DVL)
- No sensor records (Klein doesn't have ancillary sensor streams)

The reader yields both SDFSonarPage (sonar versions 3501/3502) and
SDFUnknownPage (everything else). The converter filters to sonar pages
only with an info log; non-sonar pages are dropped silently.

Klein-specific design decisions:

- depth_m is set to NaN. The spec stores depth in Volts (field 31), and
  conversion to meters requires calibration data from the INI file that
  we don't currently parse. Users with calibration can override.
- sound_vel_ms per ping is NaN (Klein doesn't provide sound velocity).
- heave_m per ping is NaN (Klein doesn't provide heave).
- altitude_m is currently a copy of fathometer_reading_m (no attitude
  correction yet).
- sample_rate_hz is computed from the first page's (num_samples, range_m)
  assuming 1500 m/s sound velocity.
- pulse_length_us is 0.0 (not provided by Klein).
- freq_hz_start == freq_hz_end == center_freq_hz.
- Pages with different num_samples are padded to max with NaN, logged as
  a warning when it happens (was previously silent). Each ping's true
  recorded sample count survives padding via the new
  `valid_samples` (ping, channel) variable, so downstream code (e.g.
  `processing/farrange.py`) can distinguish real short pings from
  padded filler rather than treating the padded array shape as if every
  ping genuinely recorded that many samples.
- `convert()` builds one array per call across every page it's handed.
  Current callers (`io/dispatch.py: read_and_convert()`) invoke it once
  per source file (`01_convert.py` calls `read_and_convert([single_path], ...)`
  in a loop), so in practice padding is currently scoped to pages within
  one physical `.sdf` file. `read_and_convert()`'s own multi-file-merge
  capability would pad across files too if a caller ever used it that
  way — worth knowing before doing so, not a hidden trap given the
  warning above, but a single outlier-length page anywhere in whatever
  set of pages reaches this function does inflate every other page's
  stored array.

2026 rshomberg@uri.edu
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from sidescanpy.core.schema import (
    make_empty_attitude_record,
    make_empty_dvl_record,
    make_empty_navigation_record,
)
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.sdf.pages import SDFSonarPage, SDFUnknownPage

logger = logging.getLogger(__name__)


_DEFAULT_SOUND_VEL_MS = 1500.0


# ============================================================
# PUBLIC API
# ============================================================

def convert(
    pages: Iterable,
    *,
    source_files: list[Path] | None = None,
    survey_name: str | None = None,
) -> SideScanSurvey:
    """Convert SDF pages into a validated SideScanSurvey.

    Pages can include both SDFSonarPage and SDFUnknownPage (as yielded
    by read_sdf). Non-sonar pages are filtered out with an info log.

    Parameters
    ----------
    pages : Iterable
        SDF pages from read_sdf or read_sdf_iter.
    source_files : list[Path], optional
        Paths to the source SDF files.
    survey_name : str, optional
        Survey name. Defaults to first source file's stem.

    Returns
    -------
    SideScanSurvey
        Validated survey with one or more frequency channels and aux records.

    Raises
    ------
    ValueError
        If `pages` contains no sonar pages.
    """
    pages_list = list(pages)
    sonar_pages = [p for p in pages_list if isinstance(p, SDFSonarPage)]
    n_unknown = len(pages_list) - len(sonar_pages)

    if n_unknown > 0:
        logger.info(
            f"Filtered {n_unknown} non-sonar pages from input "
            f"(kept {len(sonar_pages)} sonar pages)"
        )

    if not sonar_pages:
        raise ValueError(
            "Cannot convert empty page list (no sonar pages in input)"
        )

    source_files = source_files or []
    if survey_name is None and source_files:
        survey_name = source_files[0].stem
    elif survey_name is None:
        survey_name = "untitled_survey"

    logger.info(
        f"Converting {len(sonar_pages)} SDF sonar pages → SideScanSurvey "
        f"(name={survey_name!r})"
    )

    # Group pages by center frequency
    freq_groups: dict[int, list[SDFSonarPage]] = defaultdict(list)
    for page in sonar_pages:
        freq_groups[page.center_freq_hz].append(page)

    # Sort each group by ping_number
    for freq_hz in freq_groups:
        freq_groups[freq_hz].sort(key=lambda p: p.ping_number)

    logger.info(f"  {len(freq_groups)} frequency channel(s)")

    # Build a frequency channel per group
    survey = SideScanSurvey(
        source_files=source_files,
        source_format="sdf",
        survey_name=survey_name,
    )

    for freq_hz, group_pages in freq_groups.items():
        freq_key = f"hz_{freq_hz}"
        ds = _build_channel_dataset(freq_hz, group_pages)
        survey.add_frequency(freq_key, ds)
        ch = survey.get_frequency(freq_key)
        logger.info(
            f"  {freq_key}: {ch.n_pings} pings × 2 channels × "
            f"{ch.n_samples} samples"
        )

    # Build aux records from per-page values
    survey.navigation = _build_navigation_record(sonar_pages)
    survey.attitude = _build_attitude_record(sonar_pages)
    survey.dvl = make_empty_dvl_record(source="sdf_no_dvl")

    return survey


# ============================================================
# CHANNEL DATASET BUILDER
# ============================================================

def _build_channel_dataset(
    freq_hz: int,
    pages: list[SDFSonarPage],
) -> xr.Dataset:
    """Build the (ping, channel, sample) dataset for one frequency."""
    n_pings = len(pages)

    port_lengths = [len(p.port_data) for p in pages]
    stbd_lengths = [len(p.stbd_data) for p in pages]
    max_samples = max(max(port_lengths, default=0),
                      max(stbd_lengths, default=0))
    if max_samples == 0:
        raise ValueError(
            f"All pages at {freq_hz} Hz have zero samples"
        )

    n_padded = sum(
        1 for n_p, n_s in zip(port_lengths, stbd_lengths)
        if n_p != max_samples or n_s != max_samples
    )
    if n_padded > 0:
        logger.warning(
            f"  {freq_hz} Hz: {n_padded}/{n_pings} ping(s) have fewer samples "
            f"than this group's max ({max_samples}) and are NaN-padded to "
            f"match — real per-ping length is preserved in the "
            f"'valid_samples' variable. Real length range: "
            f"{min(min(port_lengths, default=0), min(stbd_lengths, default=0))}"
            f"-{max_samples}."
        )

    # NaN-padded backscatter array
    bs = np.full((n_pings, 2, max_samples), np.nan, dtype=np.float32)
    for i, page in enumerate(pages):
        n_p = len(page.port_data)
        n_s = len(page.stbd_data)
        if n_p > 0:
            bs[i, 0, :n_p] = page.port_data.astype(np.float32)
        if n_s > 0:
            bs[i, 1, :n_s] = page.stbd_data.astype(np.float32)

    # Per-ping, per-channel true recorded sample count — preserved so
    # padding (above) never silently discards how much of a padded
    # ping's tail is real vs. filler.
    valid_samples = np.stack(
        [np.array(port_lengths, dtype=np.int32),
         np.array(stbd_lengths, dtype=np.int32)],
        axis=1,
    )

    # Per-ping arrays
    times_ns = np.array([p.timestamp_ns for p in pages], dtype="int64")
    times = times_ns.astype("datetime64[ns]")
    ping_numbers = np.array([p.ping_number for p in pages], dtype=np.int32)

    latitudes = np.array([p.latitude for p in pages], dtype=np.float64)
    longitudes = np.array([p.longitude for p in pages], dtype=np.float64)
    headings = np.array([p.heading_deg for p in pages], dtype=np.float32)
    pitches = np.array([p.pitch_deg for p in pages], dtype=np.float32)
    rolls = np.array([p.roll_deg for p in pages], dtype=np.float32)
    fathometer = np.array(
        [p.fathometer_reading_m for p in pages], dtype=np.float32
    )

    # depth_m is NaN: Klein stores depth in Volts, requires INI calibration
    depths = np.full(n_pings, np.nan, dtype=np.float32)

    # altitude_m == fathometer_reading_m for now (until attitude correction)
    altitude = fathometer.copy()

    # NaN per-ping fields (Klein doesn't provide these)
    sound_vel = np.full(n_pings, np.nan, dtype=np.float32)
    heave = np.full(n_pings, np.nan, dtype=np.float32)

    # Compute sample_rate_hz from first page; warn if subsequent differ
    sample_rate_hz = _compute_sample_rate(pages, freq_hz)

    attrs = {
        "freq_hz_start": int(freq_hz),
        "freq_hz_end": int(freq_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "pulse_length_us": 0.0,
        "processing_level": "raw",
        "processing_history": json.dumps([]),
    }

    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(max_samples),
        "ping_number": (["ping"], ping_numbers),
        "time": (["ping"], times),
        "channel_id": (["channel"], np.array(["port", "starboard"])),
    }

    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], bs.copy()),
        "backscatter_raw": (["ping", "channel", "sample"], bs.copy()),
        "valid_samples": (["ping", "channel"], valid_samples),
        "latitude": (["ping"], latitudes),
        "longitude": (["ping"], longitudes),
        "heading_deg": (["ping"], headings),
        "pitch_deg": (["ping"], pitches),
        "roll_deg": (["ping"], rolls),
        "depth_m": (["ping"], depths),
        "altitude_m": (["ping"], altitude),
        "fathometer_reading_m": (["ping"], fathometer),
        "heave_m": (["ping"], heave),
        "sound_vel_ms": (["ping"], sound_vel),
    }

    return xr.Dataset(data_vars, coords, attrs)


def _compute_sample_rate(
    pages: list[SDFSonarPage],
    freq_hz: int,
) -> float:
    """Compute sample_rate_hz from the first page; warn on inconsistency.

    Formula: sample_rate_hz = (num_samples × sound_vel) / (2 × range_m),
    using a fixed sound velocity of 1500 m/s.
    """
    first = pages[0]
    n_samples = first.num_samples
    range_m = first.range_m
    if range_m <= 0:
        logger.warning(
            f"  {freq_hz} Hz: first page has range_m={range_m}; "
            f"using 0 for sample_rate_hz"
        )
        return 0.0
    sample_rate = (n_samples * _DEFAULT_SOUND_VEL_MS) / (2.0 * range_m)

    # Check consistency with subsequent pages
    for page in pages[1:]:
        if page.num_samples != n_samples or page.range_m != range_m:
            logger.warning(
                f"  {freq_hz} Hz: inconsistent num_samples/range_m across "
                f"pages (sample_rate computed from first page; downstream "
                f"may need per-ping sample_rate)"
            )
            break

    return sample_rate


# ============================================================
# AUX RECORDS
# ============================================================

def _build_navigation_record(pages: list[SDFSonarPage]) -> xr.Dataset:
    """Build a navigation record from per-page (timestamp, lat, lon)."""
    if not pages:
        return make_empty_navigation_record(source="sdf_no_pages")

    sorted_pages = sorted(pages, key=lambda p: p.timestamp_ns)
    times_ns = np.array(
        [p.timestamp_ns for p in sorted_pages], dtype="int64"
    )
    times = times_ns.astype("datetime64[ns]")
    lats = np.array([p.latitude for p in sorted_pages], dtype=np.float64)
    lons = np.array([p.longitude for p in sorted_pages], dtype=np.float64)

    return xr.Dataset(
        data_vars={
            "latitude": (("obs",), lats),
            "longitude": (("obs",), lons),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "extracted_from_sdf_pages"},
    )


def _build_attitude_record(pages: list[SDFSonarPage]) -> xr.Dataset:
    """Build an attitude record from per-page (timestamp, heading, pitch, roll).

    Klein attitude data lives in each ping's header rather than as a
    separate stream, so the attitude record matches the per-ping density.
    """
    if not pages:
        return make_empty_attitude_record(source="sdf_no_pages")

    sorted_pages = sorted(pages, key=lambda p: p.timestamp_ns)
    times_ns = np.array(
        [p.timestamp_ns for p in sorted_pages], dtype="int64"
    )
    times = times_ns.astype("datetime64[ns]")
    headings = np.array(
        [p.heading_deg for p in sorted_pages], dtype=np.float32
    )
    pitches = np.array(
        [p.pitch_deg for p in sorted_pages], dtype=np.float32
    )
    rolls = np.array([p.roll_deg for p in sorted_pages], dtype=np.float32)
    heaves = np.full(len(sorted_pages), np.nan, dtype=np.float32)

    return xr.Dataset(
        data_vars={
            "heading_deg": (("obs",), headings),
            "pitch_deg": (("obs",), pitches),
            "roll_deg": (("obs",), rolls),
            "heave_m": (("obs",), heaves),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "extracted_from_sdf_pages"},
    )        