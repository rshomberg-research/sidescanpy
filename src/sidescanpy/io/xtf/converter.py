"""
Convert XTF sonar pings into a SideScanSurvey.

Takes the per-file output of :func:`sidescanpy.io.xtf.read_xtf` (one
:class:`~sidescanpy.io.xtf.reader.XtfPingData` per file) and builds a
validated survey shaped like the JSF and SDF ones:

- One frequency channel per port/starboard pair sharing a ``ChanInfo``
  frequency, keyed ``hz_<Hz>``. ``ChanInfo.Frequency`` is in kHz (the
  per-channel header's ``Frequency`` is a 16-bit kHz value too); a value
  of 10 000 or more cannot be a sidescan frequency in kHz and is taken
  as Hz (logged at DEBUG).
- ``backscatter_intensity`` is ``(ping, channel, sample)`` float32 with
  channel 0 = port, 1 = starboard. Samples beyond a ping's
  ``NumSamples`` and a side absent from a ping are NaN;
  ``valid_samples`` (ping, channel) holds the recorded count (0 where
  absent), as in the SDF converter. Sample 0 is nearest the sonar on
  both sides: XTF stores port samples far -> near (so ``[port | stbd]``
  reads as one across-track line), and the converter reverses each
  port ping's valid samples. Verified on real EdgeTech 4200 files: the
  first bottom return of the reversed port data tracks
  ``SensorPrimaryAltitude`` the way the starboard data does as stored.
- Samples are multiplied by ``2**-Weight`` from each channel header.
  EdgeTech Discover writes the raw block-floating-point samples and puts
  the JSF weighting factor N in ``Weight``, which can change from ping to
  ping; scaling makes the samples equal to the JSF ones for the same
  ping (checked on a public Discover 7.22 recording with both files).
  Writers that leave ``Weight`` at 0 are unaffected. The range seen is
  recorded in ``xtf_weight_min`` / ``xtf_weight_max``.
- Each frequency has its own ping axis. A ping holding neither channel
  of a frequency (some files' early packets carry only the low-frequency
  pair, see :mod:`sidescanpy.io.xtf.reader`) is left out of that
  frequency, logged at DEBUG and counted in the ``xtf_pings_absent``
  attribute.
- Per-ping fields come from the ping header: ``SensorYcoordinate`` /
  ``SensorXcoordinate`` → ``latitude`` / ``longitude`` (degrees with
  ``NavUnits`` 3; with ``NavUnits`` 0 they are northing / easting in
  metres and are converted, see below), ``SensorHeading``, ``SensorPitch`` (positive
  nose up), ``SensorRoll`` (positive roll to starboard, i.e. port up,
  the JSF convention), ``Heave``, ``SensorDepth`` →
  ``heading_deg``, ``pitch_deg``, ``roll_deg``, ``heave_m``, ``depth_m``.
  ``SensorPrimaryAltitude`` → ``altitude_m`` and
  ``fathometer_reading_m``, with 0 (no bottom lock) as NaN, the way the
  JSF converter turns an invalid altitude into NaN.
- Times from ``Year`` … ``Second`` + ``HSeconds`` (10 ms resolution, the
  recorder's time base, normally UTC).

Sound velocity and sample rate
------------------------------
Some recorders write half the sound velocity in ``SoundVelocity`` (the
one-way factor ``c / 2``). A value below 1000 m/s is not a plausible
sound speed in water, so it is doubled (logged at DEBUG); a plausible
value is kept. Where the channel header gives both ``SlantRange`` and
``TimeDuration`` the choice is cross-checked against ``SlantRange /
TimeDuration`` (which is ``c / 2``) and a disagreement over 5 % logged at
WARNING. Zero or negative values become NaN, so downstream code falls
back to 1500 m/s as for SDF.

``sample_rate_hz`` is ``NumSamples / TimeDuration`` (falling back to
``NumSamples * c / (2 * SlantRange)`` when ``TimeDuration`` is 0), so
the slant range of the last sample, ``n * c / (2 * fs)``, equals
``TimeDuration * c / 2``. The survey model has one sample rate per
frequency. Files whose range setting changes mid-file (the sample count
staying fixed) are meant to be split first with
:func:`sidescanpy.io.xtf.split_by_sample_rate` and each run converted
on its own. When unsplit pings still mix rates, the most common rate is
used and the pings at other rates are reported at WARNING and counted in
``xtf_pings_other_sample_rate``; their ranges are then wrong by the ratio
of the rates. For split runs the tolerance is twice the split's
``rtol``, so the WARNING does not fire for them.

Projected navigation
--------------------
With ``NavUnits`` 0 the file header says ``SensorXcoordinate`` /
``SensorYcoordinate`` (and ``ShipXcoordinate`` / ``ShipYcoordinate``, not
used) are easting / northing in metres, but not in which projection. The
caller names it with ``nav_crs`` (an EPSG code, a string pyproj accepts,
or a ``pyproj.CRS``); the sensor positions are converted to WGS84
longitude / latitude with pyproj, a position of exactly (0, 0) becomes
NaN (no fix), and the CRS is recorded in ``xtf_nav_crs``. Without
``nav_crs`` such a file raises ``ValueError``: the zone is not guessed.
``nav_crs`` is ignored for files with ``NavUnits`` 3.

Layback and position source
---------------------------
``Layback`` → ``layback_m`` and ``CableOut`` (+ ``CableOutHundredths``)
→ ``cable_out_m``, per ping, as written (some writers leave ``Layback``
NaN or 0). ``position_source`` says whether ``latitude`` / ``longitude``
already are the towfish: ``towfish`` when ``SensorX/Ycoordinate`` sit
more than ``SHIP_SENSOR_TOL_M`` from ``ShipX/Ycoordinate`` on at least
``SHIP_SENSOR_FRACTION`` of the pings (the writer offset them), otherwise
``unknown``. Ship equal to Sensor does not mean the ship: EdgeTech
Discover writes the laid-back towfish position in both fields. The
``layback`` correction step does not move ``unknown`` positions.

Not converted: ``pulse_length_us`` is not in the XTF ping or channel
headers and is 0.0 (as for SDF); gain codes are not interpreted.

Provenance attributes on every frequency dataset (``xtf_`` prefix):
recording program and version, sonar name and type, channel numbers
and names, whether sound velocity was doubled, the ``Weight`` range, and counts of reader
anomalies summed over the converted files (packets with fewer channel
blocks than declared, clipped channel blocks, channel blocks without
ChanInfo, unknown and non-sonar packets, resyncs, truncated files).
A single split run (``XtfPingData.segment`` set) adds
``xtf_segment_index`` and ``xtf_segment_count`` (1-based, kept runs of
the file), ``xtf_segment_ping_start`` / ``xtf_segment_ping_stop`` (its
half-open range in the file's sonar pings), ``xtf_file_pings``, and
``xtf_runs_dropped_short`` / ``xtf_pings_dropped_short_runs`` (runs of
the file shorter than the split's ``min_pings``, and their pings).

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from sidescanpy.core.schema import (
    OPTIONAL_PER_PING_VARS,
    POSITION_SOURCE_ATTR,
    POSITION_SOURCE_REASON_ATTR,
    POSITION_TOWFISH,
    POSITION_UNKNOWN,
    combine_position_sources,
    make_empty_dvl_record,
)
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.xtf.reader import XtfChannel, XtfPingData, ping_times
from sidescanpy.io.xtf.structs import HEADER_TYPE_SONAR, NAV_UNITS_LATLON, NAV_UNITS_METERS

logger = logging.getLogger(__name__)

#: ``SoundVelocity`` values below this (m/s) are taken as halved and doubled.
HALVED_SOUND_VELOCITY_BELOW_MS = 1000.0
#: Relative tolerance for treating two pings' sample rates as equal.
SAMPLE_RATE_RTOL = 1e-3
_FREQ_HZ_THRESHOLD = 10_000.0
_SV_CROSSCHECK_RTOL = 0.05
_NAV_SOURCE = "extracted_from_xtf_ping_headers"
#: Sensor further than this (m) from Ship counts as a separate (towfish) position.
SHIP_SENSOR_TOL_M = 0.5
#: Fraction of pings with a Ship position that must be apart for "towfish".
SHIP_SENSOR_FRACTION = 0.95
_METRES_PER_DEGREE = 111_320.0


# ============================================================
# PUBLIC API
# ============================================================

def convert(
    pings: XtfPingData | Iterable[XtfPingData],
    *,
    source_files: list[Path] | None = None,
    survey_name: str | None = None,
    nav_crs: int | str | CRS | None = None,
) -> SideScanSurvey:
    """Convert XTF sonar pings into a validated SideScanSurvey.

    Parameters
    ----------
    pings : XtfPingData or iterable of XtfPingData
        Output of :func:`sidescanpy.io.xtf.read_xtf`, one per file.
        Several files are merged into one survey: pings are concatenated
        in the given order and stably sorted by time if that order is
        not chronological; frequencies are matched by value and shorter
        pings NaN-padded to the longest.
    source_files : list[Path], optional
        Source paths stored on the survey. Defaults to the paths in
        ``pings``.
    survey_name : str, optional
        Survey name. Defaults to the first source file's stem.
    nav_crs : int, str or pyproj.CRS, optional
        Projected CRS of the coordinates in files with ``NavUnits`` 0
        (easting / northing in metres), e.g. ``32619`` or ``"EPSG:32619"``
        for UTM zone 19N. Required for such files; ignored for files with
        ``NavUnits`` 3. Positions are converted to WGS84 longitude /
        latitude (see the module docstring).

    Returns
    -------
    SideScanSurvey
        One frequency channel per port/starboard frequency, navigation and
        attitude records from the ping headers, an empty DVL record.

    Raises
    ------
    ValueError
        If there are no sonar pings, no port/starboard channels, more
        than one channel pair of unknown frequency, a frequency whose
        sample rate cannot be derived, a file with ``NavUnits`` 0 and no
        ``nav_crs``, or a ``nav_crs`` that is not projected.
    NotImplementedError
        If a file's ``NavUnits`` is neither 0 nor 3.
    """
    files = [pings] if isinstance(pings, XtfPingData) else list(pings)
    if not files or sum(f.n_pings for f in files) == 0:
        raise ValueError("Cannot convert XTF input with no sonar pings")
    crs = _nav_crs(nav_crs)
    projected = [_check_nav_units(f, crs) for f in files]

    if source_files is None:
        source_files = [Path(f.path) for f in files]
    if survey_name is None:
        survey_name = source_files[0].stem if source_files else "untitled_survey"
    logger.debug(
        "Converting %d XTF file(s), %d pings -> SideScanSurvey (name=%r)",
        len(files), sum(f.n_pings for f in files), survey_name,
    )

    position_attrs = _position_source_attrs(files, projected)
    headers = np.concatenate([
        _geographic_headers(f, crs) if proj else f.ping_headers for f, proj in zip(files, projected)
    ])
    times = ping_times(headers)
    order = _time_order(times)
    sound_vel, doubled = _sound_velocity(files)
    provenance = {**_provenance(files, doubled), **position_attrs}
    if any(projected):
        provenance["xtf_nav_crs"] = crs.to_string()
    rate_rtol = _rate_rtol(files)

    survey = SideScanSurvey(
        source_files=list(source_files),
        source_format="xtf",
        survey_name=survey_name,
    )
    pieces = _collect_frequencies(files)
    if not pieces:
        raise ValueError("XTF input has no port/starboard sonar channels")
    for freq_hz in sorted(pieces):
        ds = _build_frequency_channel(
            freq_hz, pieces[freq_hz], headers, times, sound_vel, order, provenance, rate_rtol
        )
        freq_key = f"hz_{freq_hz}"
        survey.add_frequency(freq_key, ds)
        ch = survey.get_frequency(freq_key)
        logger.debug(
            "  %s: %d pings x 2 channels x %d samples", freq_key, ch.n_pings, ch.n_samples
        )

    survey.navigation = _build_navigation_record(headers[order], times[order])
    survey.attitude = _build_attitude_record(headers[order], times[order])
    survey.dvl = make_empty_dvl_record(source="xtf_no_dvl")
    return survey


# ============================================================
# FILE-LEVEL CHECKS
# ============================================================

def _nav_crs(nav_crs: int | str | CRS | None) -> CRS | None:
    if nav_crs is None:
        return None
    crs = CRS.from_user_input(nav_crs)
    if not crs.is_projected:
        raise ValueError(
            f"nav_crs {crs.to_string()} is not a projected CRS; XTF NavUnits 0 "
            "coordinates are easting/northing in metres"
        )
    return crs


def _check_nav_units(data: XtfPingData, crs: CRS | None) -> bool:
    """True if the file's coordinates are projected (NavUnits 0) and ``crs`` is set."""
    units = int(data.file_header["NavUnits"])
    if units == NAV_UNITS_LATLON:
        if crs is not None:
            logger.debug("%s: NavUnits 3 (degrees); nav_crs ignored", data.path)
        return False
    if units == NAV_UNITS_METERS:
        if crs is None:
            raise ValueError(
                f"{data.path}: NavUnits 0 (projected easting/northing in metres) and the "
                "file does not say which projection; pass nav_crs (an EPSG code such as "
                "32619 for UTM zone 19N, or a pyproj CRS) to convert the positions"
            )
        return True
    raise NotImplementedError(
        f"{data.path}: NavUnits {units} is not defined by the XTF specification; "
        "only NavUnits 0 (projected metres, with nav_crs) and 3 (degrees) are converted"
    )


def _geographic_headers(data: XtfPingData, crs: CRS) -> np.ndarray:
    """Copy of the ping headers with sensor easting/northing replaced by WGS84 lon/lat."""
    h = data.ping_headers.copy()
    east = h["SensorXcoordinate"].astype(np.float64)
    north = h["SensorYcoordinate"].astype(np.float64)
    no_fix = (east == 0) & (north == 0)
    lon, lat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True).transform(east, north)
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    lon[no_fix] = np.nan
    lat[no_fix] = np.nan
    if no_fix.any():
        logger.debug("%s: %d ping(s) at easting/northing (0, 0) set to NaN", data.path, int(no_fix.sum()))
    h["SensorXcoordinate"] = lon
    h["SensorYcoordinate"] = lat
    logger.debug("%s: NavUnits 0 positions converted from %s to WGS84", data.path, crs.to_string())
    return h


def _ship_sensor_offset_m(data: XtfPingData, projected: bool) -> np.ndarray:
    """Per-ping distance (m) from Ship to Sensor coordinates; NaN where Ship is (0, 0)."""
    h = data.ping_headers
    sx, sy = h["SensorXcoordinate"].astype(np.float64), h["SensorYcoordinate"].astype(np.float64)
    shx, shy = h["ShipXcoordinate"].astype(np.float64), h["ShipYcoordinate"].astype(np.float64)
    if projected:
        dist = np.hypot(sx - shx, sy - shy)
    else:
        dist = _METRES_PER_DEGREE * np.hypot(sy - shy, (sx - shx) * np.cos(np.deg2rad(sy)))
    dist[(shx == 0) & (shy == 0)] = np.nan
    return dist


def _position_source_attrs(files: list[XtfPingData], projected: list[bool]) -> dict[str, str]:
    """What ``SensorX/Ycoordinate`` are the position of, from Ship vs Sensor.

    Per file: Sensor away from Ship (more than ``SHIP_SENSOR_TOL_M``) on at
    least ``SHIP_SENSOR_FRACTION`` of the pings with a Ship position means
    the writer already placed the towfish (towfish). Ship equal to Sensor
    or absent says nothing (unknown): some writers (EdgeTech Discover)
    put the laid-back towfish position in both fields.
    """
    sources, notes = [], []
    for data, proj in zip(files, projected):
        dist = _ship_sensor_offset_m(data, proj)
        with_ship = np.isfinite(dist)
        n_ship = int(with_ship.sum())
        n_apart = int((dist[with_ship] > SHIP_SENSOR_TOL_M).sum())
        if n_ship and n_apart >= SHIP_SENSOR_FRACTION * n_ship:
            sources.append(POSITION_TOWFISH)
        else:
            sources.append(POSITION_UNKNOWN)
        median = float(np.median(dist[with_ship])) if n_ship else float("nan")
        notes.append(
            f"{Path(data.path).name}: Ship position in {n_ship} of {data.n_pings} pings, "
            f"Sensor more than {SHIP_SENSOR_TOL_M:g} m from it in {n_apart} (median {median:.2f} m)"
        )
    source = combine_position_sources(sources)
    reason = (
        "XTF Sensor vs Ship coordinates (apart: Sensor is the towfish; equal or no Ship: unknown). "
        + "; ".join(notes)
    )
    return {POSITION_SOURCE_ATTR: source, POSITION_SOURCE_REASON_ATTR: reason}


def _rate_rtol(files: list[XtfPingData]) -> float:
    """Sample-rate tolerance: twice the split's ``rtol`` when every input is a split run."""
    if all(f.segment is not None for f in files):
        return max(SAMPLE_RATE_RTOL, 2.0 * max(f.segment.rtol for f in files))
    return SAMPLE_RATE_RTOL


def _time_order(times: np.ndarray) -> np.ndarray:
    """Stable time order of the merged pings (identity when already ordered)."""
    n = len(times)
    t = times.astype("int64")
    valid = ~np.isnat(times)
    if np.all(np.diff(t[valid]) >= 0):
        return np.arange(n)
    order = np.argsort(times, kind="stable")
    moved = int(np.sum(order != np.arange(n)))
    logger.debug("XTF pings not in time order; %d pings reordered by time", moved)
    return order


def _decode(value) -> str:
    return bytes(value).decode("ascii", "replace").rstrip("\x00 ")


# ============================================================
# SOUND VELOCITY
# ============================================================

def _sound_velocity(files: list[XtfPingData]) -> tuple[np.ndarray, bool]:
    """Per-ping sound velocity (m/s, float32) over the merged pings."""
    out = []
    any_doubled = False
    for f in files:
        sv = f.ping_headers["SoundVelocity"].astype(np.float64)
        valid = np.isfinite(sv) & (sv > 0)
        halved = valid & (sv < HALVED_SOUND_VELOCITY_BELOW_MS)
        real = np.where(halved, 2.0 * sv, sv)
        real[~valid] = np.nan
        if halved.any():
            any_doubled = True
            logger.debug(
                "%s: SoundVelocity below %g m/s in %d of %d pings (median %g); "
                "taken as c/2 and doubled",
                f.path, HALVED_SOUND_VELOCITY_BELOW_MS, int(halved.sum()), len(sv),
                float(np.median(sv[halved])),
            )
        if (~valid).any():
            logger.debug(
                "%s: SoundVelocity missing or non-positive in %d of %d pings; set to NaN",
                f.path, int((~valid).sum()), len(sv),
            )
        _crosscheck_sound_velocity(f, real)
        out.append(real)
    return np.concatenate(out).astype(np.float32), any_doubled


def _crosscheck_sound_velocity(data: XtfPingData, real: np.ndarray) -> None:
    """Compare c with 2 * SlantRange / TimeDuration from the channel headers."""
    implied = np.full(data.n_pings, np.nan)
    for ch in data.channels.values():
        h = ch.headers
        ok = ch.present & (h["TimeDuration"] > 0) & (h["SlantRange"] > 0) & np.isnan(implied)
        implied[ok] = 2.0 * h["SlantRange"][ok] / h["TimeDuration"][ok]
    both = np.isfinite(implied) & np.isfinite(real)
    if not both.any():
        return
    bad = both & (np.abs(real - implied) > _SV_CROSSCHECK_RTOL * real)
    if bad.any():
        logger.warning(
            "%s: sound velocity disagrees by more than %d%% with 2*SlantRange/TimeDuration "
            "in %d of %d pings (median %.1f vs %.1f m/s); kept the ping-header value",
            data.path, int(_SV_CROSSCHECK_RTOL * 100), int(bad.sum()), int(both.sum()),
            float(np.median(real[bad])), float(np.median(implied[bad])),
        )


# ============================================================
# FREQUENCY GROUPS
# ============================================================

@dataclass
class _Piece:
    """One frequency group of one file, placed in the merged ping index."""

    offset: int
    port: XtfChannel | None
    stbd: XtfChannel | None


@dataclass
class _FreqPieces:
    pieces: list[_Piece] = field(default_factory=list)
    channel_numbers: set = field(default_factory=set)
    channel_names: set = field(default_factory=set)


def _freq_hz(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value >= _FREQ_HZ_THRESHOLD:
        logger.debug("XTF frequency %g >= %g taken as Hz, not kHz", value, _FREQ_HZ_THRESHOLD)
        return int(round(value))
    return int(round(value * 1000.0))


def _collect_frequencies(files: list[XtfPingData]) -> dict[int, _FreqPieces]:
    out: dict[int, _FreqPieces] = {}
    offset = 0
    for f in files:
        unknown = 0
        for group in f.frequency_groups:
            freq_hz = _freq_hz(group.frequency)
            if freq_hz == 0:
                unknown += 1
                if unknown > 1:
                    raise ValueError(
                        f"{f.path}: more than one port/starboard pair of unknown frequency"
                    )
            entry = out.setdefault(freq_hz, _FreqPieces())
            entry.pieces.append(_Piece(offset=offset, port=group.port, stbd=group.stbd))
            for ch in (group.port, group.stbd):
                if ch is not None:
                    entry.channel_numbers.add(ch.channel_number)
                    entry.channel_names.add(ch.name)
        offset += f.n_pings
    return out


def _side_arrays(
    ch: XtfChannel | None, n_ping: int
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    """(samples, num_samples, present, headers) for one side, zeros if missing."""
    if ch is None:
        return None, np.zeros(n_ping, np.int64), np.zeros(n_ping, bool), None
    return ch.samples, ch.num_samples.astype(np.int64), ch.present, ch.headers


def _build_frequency_channel(
    freq_hz: int,
    entry: _FreqPieces,
    headers: np.ndarray,
    times: np.ndarray,
    sound_vel: np.ndarray,
    order: np.ndarray,
    provenance: dict,
    rate_rtol: float = SAMPLE_RATE_RTOL,
) -> xr.Dataset:
    n_total = len(headers)
    max_n = 0
    for piece in entry.pieces:
        for ch in (piece.port, piece.stbd):
            if ch is not None and ch.samples.shape[1] > max_n:
                max_n = ch.samples.shape[1]

    # Merged per-ping bookkeeping over every ping of every file.
    present = np.zeros((n_total, 2), dtype=bool)
    valid = np.zeros((n_total, 2), dtype=np.int64)
    duration = np.zeros(n_total)
    slant = np.zeros(n_total)
    nsamp = np.zeros(n_total)
    for piece in entry.pieces:
        for side, ch in enumerate((piece.port, piece.stbd)):
            if ch is None:
                continue
            sl = slice(piece.offset, piece.offset + len(ch.present))
            present[sl, side] = ch.present
            valid[sl, side] = ch.num_samples
            take = ch.present & (duration[sl] == 0) & (slant[sl] == 0)
            idx = np.flatnonzero(take) + piece.offset
            duration[idx] = ch.headers["TimeDuration"][take]
            slant[idx] = ch.headers["SlantRange"][take]
            nsamp[idx] = ch.num_samples[take]

    keep_mask = present.any(axis=1)
    keep = order[keep_mask[order]]
    n_absent = int((~keep_mask).sum())
    if n_absent:
        logger.debug(
            "hz_%d: %d of %d pings hold neither channel of this frequency; left out",
            freq_hz, n_absent, n_total,
        )
    if keep.size == 0:
        raise ValueError(f"hz_{freq_hz}: no pings hold this frequency")
    one_sided = int(np.sum(present[keep].sum(axis=1) == 1))
    if one_sided:
        logger.warning(
            "hz_%d: %d ping(s) hold only one side; the other side is NaN", freq_hz, one_sided
        )

    sample_rate, n_other = _sample_rate(
        freq_hz, nsamp[keep], duration[keep], slant[keep], sound_vel[keep], rate_rtol
    )

    n = len(keep)
    pos = np.empty(n_total, dtype=np.int64)
    pos[keep] = np.arange(n)
    bs = np.full((n, 2, max_n), np.nan, dtype=np.float32)
    weights: list[np.ndarray] = []
    for piece in entry.pieces:
        for side, ch in enumerate((piece.port, piece.stbd)):
            if ch is None or ch.samples.shape[1] == 0:
                continue
            rows = np.flatnonzero(ch.present)
            dst = pos[rows + piece.offset]
            m = ch.samples.shape[1]
            block = ch.samples[rows].astype(np.float32)
            # Block floating point: samples are scaled by 2**-Weight, as
            # JSF samples are by their weighting factor.
            weight = ch.headers["Weight"][rows].astype(np.int64)
            weights.append(weight)
            if np.any(weight != 0):
                block *= np.exp2(-weight).astype(np.float32)[:, None]
            k = ch.num_samples[rows, None].astype(np.int64)
            j = np.arange(m)[None, :]
            block[j >= k] = np.nan
            if side == 0:
                # XTF stores port far -> near; reverse each ping's valid
                # samples so sample 0 is nearest the sonar on both sides.
                block = np.take_along_axis(block, np.where(j < k, k - 1 - j, j), axis=1)
            bs[dst, side, :m] = block

    weight_all = np.concatenate(weights) if weights else np.zeros(0, dtype=np.int64)
    n_weighted = int(np.sum(weight_all != 0))
    if n_weighted:
        logger.debug(
            "hz_%d: %d channel blocks scaled by 2**-Weight (Weight %d to %d)",
            freq_hz, n_weighted, weight_all.min(), weight_all.max(),
        )

    h = headers[keep]
    altitude = h["SensorPrimaryAltitude"].astype(np.float32)
    n_alt0 = int(np.sum(altitude == 0))
    if n_alt0:
        logger.debug(
            "hz_%d: SensorPrimaryAltitude 0 (no value) in %d of %d pings; set to NaN",
            freq_hz, n_alt0, n,
        )
    altitude[altitude == 0] = np.nan

    attrs = {
        "freq_hz_start": int(freq_hz),
        "freq_hz_end": int(freq_hz),
        "sample_rate_hz": float(sample_rate),
        "pulse_length_us": 0.0,
        "processing_history": json.dumps([]),
        **provenance,
        "xtf_channel_numbers": ",".join(str(c) for c in sorted(entry.channel_numbers)),
        "xtf_channel_names": ",".join(sorted(entry.channel_names)),
        "xtf_pings_absent": n_absent,
        "xtf_pings_other_sample_rate": n_other,
        "xtf_weight_min": int(weight_all.min()) if weight_all.size else 0,
        "xtf_weight_max": int(weight_all.max()) if weight_all.size else 0,
    }
    coords = {
        "sample": (["sample"], np.arange(max_n)),
        "ping_number": (["ping"], h["PingNumber"].astype(np.int32)),
        "time": (["ping"], times[keep]),
        "channel_id": (["channel"], np.array(["port", "starboard"])),
    }
    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], bs),
        "valid_samples": (["ping", "channel"], valid[keep].astype(np.int32)),
        "latitude": (["ping"], h["SensorYcoordinate"].astype(np.float64)),
        "longitude": (["ping"], h["SensorXcoordinate"].astype(np.float64)),
        "heading_deg": (["ping"], h["SensorHeading"].astype(np.float32)),
        "depth_m": (["ping"], h["SensorDepth"].astype(np.float32)),
        "roll_deg": (["ping"], h["SensorRoll"].astype(np.float32)),
        "pitch_deg": (["ping"], h["SensorPitch"].astype(np.float32)),
        "heave_m": (["ping"], h["Heave"].astype(np.float32)),
        "altitude_m": (["ping"], altitude),
        "sound_vel_ms": (["ping"], sound_vel[keep]),
        "fathometer_reading_m": (["ping"], altitude.copy()),
        "layback_m": (
            ["ping"], h["Layback"].astype(np.float32),
            {"units": "m", "long_name": OPTIONAL_PER_PING_VARS["layback_m"]},
        ),
        "cable_out_m": (
            ["ping"], (h["CableOut"] + h["CableOutHundredths"] / 100.0).astype(np.float32),
            {"units": "m", "long_name": OPTIONAL_PER_PING_VARS["cable_out_m"]},
        ),
    }
    return xr.Dataset(data_vars, coords, attrs)


def _sample_rate(
    freq_hz: int,
    nsamp: np.ndarray,
    duration: np.ndarray,
    slant: np.ndarray,
    sound_vel: np.ndarray,
    rtol: float = SAMPLE_RATE_RTOL,
) -> tuple[float, int]:
    """Most common per-ping sample rate (Hz) and the count of pings at other rates."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(duration > 0, nsamp / duration, np.nan)
        c = np.where(np.isfinite(sound_vel), sound_vel, np.nan)
        from_range = np.where(slant > 0, nsamp * c / (2.0 * slant), np.nan)
    n_fallback = int(np.sum(~np.isfinite(rate) & np.isfinite(from_range)))
    if n_fallback:
        logger.debug(
            "hz_%d: TimeDuration 0 in %d pings; sample rate from SlantRange and sound velocity",
            freq_hz, n_fallback,
        )
    rate = np.where(np.isfinite(rate), rate, from_range)
    rate = rate[np.isfinite(rate) & (rate > 0)]
    if rate.size == 0:
        raise ValueError(
            f"hz_{freq_hz}: cannot derive sample_rate_hz (TimeDuration and SlantRange are 0)"
        )
    values, counts = np.unique(rate, return_counts=True)
    modal = float(values[np.argmax(counts)])
    other = np.abs(rate - modal) > rtol * modal
    n_other = int(other.sum())
    if n_other:
        hist = Counter(np.round(rate[other], 1).tolist())
        logger.warning(
            "hz_%d: sample rate changes within the converted pings (range setting changed); "
            "%d of %d pings at %.1f Hz used for sample_rate_hz, %d pings at other rates %s "
            "get ranges scaled by the ratio of the rates",
            freq_hz, rate.size - n_other, rate.size, modal, n_other, dict(hist.most_common(5)),
        )
    return modal, n_other


# ============================================================
# PROVENANCE
# ============================================================

def _provenance(files: list[XtfPingData], sound_vel_doubled: bool) -> dict:
    programs, sonars, sonar_types = [], [], []
    fewer = clipped = no_ci = resyncs = truncated = 0
    unknown = other = 0
    for f in files:
        fh = f.file_header
        prog = " ".join(
            p for p in (_decode(fh["RecordingProgramName"]),
                        _decode(fh["RecordingProgramVersion"])) if p
        )
        for lst, val in ((programs, prog), (sonars, _decode(fh["SonarName"])),
                         (sonar_types, str(int(fh["SonarType"])))):
            if val not in lst:
                lst.append(val)
        bs = f.block_stats
        fewer += bs.fewer_channels
        clipped += bs.clipped
        no_ci += bs.no_chan_info
        resyncs += f.stats.resyncs
        truncated += int(f.stats.truncated)
        unknown += sum(f.stats.unknown.values())
        other += sum(n for t, n in f.stats.counts.items() if t != HEADER_TYPE_SONAR)
    out = {
        "xtf_recording_program": "; ".join(programs),
        "xtf_sonar_name": "; ".join(sonars),
        "xtf_sonar_type": "; ".join(sonar_types),
        "xtf_sound_velocity_doubled": int(sound_vel_doubled),
        "xtf_packets_fewer_channels": fewer,
        "xtf_channel_blocks_clipped": clipped,
        "xtf_channel_blocks_no_chan_info": no_ci,
        "xtf_packets_unknown_type": unknown,
        "xtf_packets_non_sonar": other,
        "xtf_resyncs": resyncs,
        "xtf_files_truncated": truncated,
    }
    if len(files) == 1 and files[0].segment is not None:
        seg = files[0].segment
        out.update({
            "xtf_segment_index": seg.index,
            "xtf_segment_count": seg.count,
            "xtf_segment_ping_start": seg.ping_start,
            "xtf_segment_ping_stop": seg.ping_stop,
            "xtf_file_pings": seg.n_pings_file,
            "xtf_runs_dropped_short": seg.n_runs_dropped,
            "xtf_pings_dropped_short_runs": seg.n_pings_dropped,
        })
    return out


# ============================================================
# AUX RECORDS
# ============================================================

def _build_navigation_record(h: np.ndarray, times: np.ndarray) -> xr.Dataset:
    """Sensor position, heading and speed from every ping header."""
    return xr.Dataset(
        data_vars={
            "latitude": (("obs",), h["SensorYcoordinate"].astype(np.float64)),
            "longitude": (("obs",), h["SensorXcoordinate"].astype(np.float64)),
            "heading_deg": (("obs",), h["SensorHeading"].astype(np.float32)),
            "speed_knots": (("obs",), h["SensorSpeed"].astype(np.float32)),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": _NAV_SOURCE},
    )


def _build_attitude_record(h: np.ndarray, times: np.ndarray) -> xr.Dataset:
    """Heading, pitch, roll and heave from every ping header."""
    return xr.Dataset(
        data_vars={
            "heading_deg": (("obs",), h["SensorHeading"].astype(np.float32)),
            "pitch_deg": (("obs",), h["SensorPitch"].astype(np.float32)),
            "roll_deg": (("obs",), h["SensorRoll"].astype(np.float32)),
            "heave_m": (("obs",), h["Heave"].astype(np.float32)),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": _NAV_SOURCE},
    )
