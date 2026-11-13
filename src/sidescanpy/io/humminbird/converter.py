"""
Convert Humminbird side imaging recordings into a SideScanSurvey.

Takes the output of :func:`sidescanpy.io.humminbird.read_humminbird`
(one :class:`~sidescanpy.io.humminbird.reader.HumRecording` per
recording) and builds a validated survey shaped like the JSF, SDF and
XTF ones:

- The side imaging beams, 2 (port) and 3 (starboard), make one frequency
  channel per recorded frequency (tag 146, in Hz), keyed ``hz_<Hz>``.
  Port and starboard pings are paired by their time (tag 129): the unit
  records both sides of a ping at the same millisecond, or the starboard
  one 1 ms later, so times within :data:`PAIR_TOLERANCE_MS` pair (the
  ping takes the port time). A ping
  recorded on one side only keeps NaN on the other (``valid_samples``
  0) and is counted in ``humminbird_pings_one_side``.
- ``backscatter_intensity`` is ``(ping, channel, sample)`` float32,
  the recorded 8-bit values unscaled. Both SON files store samples near
  to far, so neither side is reversed (checked on real Helix files: the
  water column and first bottom return line up on both sides and with
  the recorded depth). Pings hold slightly different sample counts; the
  shorter ones are NaN-padded and ``valid_samples`` holds each count.
- The down-looking beams (0, 1 and 4: 2D sonar and down imaging) have one
  channel each and are not converted; one WARNING per recording names
  them, and ``humminbird_beams_skipped`` lists them.

Sample rate
-----------
The files do not record the sample interval or range. The samples are
taken at a fixed rate: on real Helix files the first bottom return sits
at the same depth-to-sample ratio (about 0.019 m per sample) whatever
the ping's sample count. ``sample_rate_hz`` defaults to
:data:`HUMMINBIRD_SAMPLE_RATE_HZ`, the rate implied by the pixel size
PyHum and PING-Mapper use (``2 * asin(c / (L f)) / pi`` metres with
``L`` = 0.108 m and ``f`` = 455 kHz, which is ``c / (2 * fs)`` with
``fs = pi L f / 4``). At 1500 m/s that is 0.0194 m per sample, within
2 % of the first-bottom estimate. Pass ``sample_rate_hz`` to override it;
``humminbird_sample_rate_source`` records which was used.

Per-ping fields
---------------
- ``latitude`` / ``longitude`` from tags 130/131 with
  :func:`~sidescanpy.io.humminbird.reader.humminbird_to_lonlat`. They
  are the head unit's GPS position, repeated between fixes.
- ``heading_deg`` from tag 132 (tenths of a degree). Without a compass
  connected this is the GPS course over ground.
- ``altitude_m`` and ``fathometer_reading_m`` from tag 135, the depth
  below the transducer in decimetres (0, no bottom lock, is NaN).
- ``depth_m`` (transducer below the surface), ``roll_deg``,
  ``pitch_deg``, ``heave_m`` and ``sound_vel_ms`` are not recorded and
  are NaN, so downstream code uses 1500 m/s.
- ``time``: the ``.DAT`` start time (Unix seconds, UTC) plus tag 129
  (milliseconds since the start).
- ``pulse_length_us`` is not recorded and is 0.0.

``position_source`` is ``unknown``: the file holds the GPS antenna's
position and says nothing about where the transducer is.

The navigation record holds every side imaging ping's position, heading
and speed (tag 133, decimetres per second, stored as knots). There is
no attitude or DVL record.

Provenance attributes on every frequency dataset (``humminbird_``
prefix): the recording name and sonar id from the ``.DAT``, its water
type, the SON header sizes, beams skipped, pings on one side only, and
reader counts of truncated frames and resyncs.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from sidescanpy.core.schema import (
    POSITION_SOURCE_ATTR,
    POSITION_SOURCE_REASON_ATTR,
    POSITION_UNKNOWN,
    make_empty_attitude_record,
    make_empty_dvl_record,
)
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.humminbird.reader import (
    BEAM_NAMES,
    BEAM_PORT,
    BEAM_STARBOARD,
    HumBeam,
    HumRecording,
    humminbird_to_lonlat,
)

logger = logging.getLogger(__name__)

#: Transducer length (m) and frequency (Hz) in the PyHum / PING-Mapper pixel size.
_PIXEL_SIZE_LENGTH_M = 0.108
_PIXEL_SIZE_FREQ_HZ = 455_000.0
#: Default sample rate (Hz): ``pi * L * f / 4`` from the pixel size above.
HUMMINBIRD_SAMPLE_RATE_HZ = float(np.pi * _PIXEL_SIZE_LENGTH_M * _PIXEL_SIZE_FREQ_HZ / 4.0)
#: Largest time difference (ms) between a port and a starboard ping of
#: the same transmission (the unit writes them up to 1 ms apart; pings are
#: tens of milliseconds apart).
PAIR_TOLERANCE_MS = 5
_NAV_SOURCE = "extracted_from_humminbird_son_headers"
_MS_TO_KNOTS = 3600.0 / 1852.0


# ============================================================
# PUBLIC API
# ============================================================

def convert(
    recordings: HumRecording | Iterable[HumRecording],
    *,
    source_files: list[Path] | None = None,
    survey_name: str | None = None,
    sample_rate_hz: float | None = None,
) -> SideScanSurvey:
    """Convert Humminbird recordings into a validated SideScanSurvey.

    Parameters
    ----------
    recordings : HumRecording or iterable of HumRecording
        Output of :func:`sidescanpy.io.humminbird.read_humminbird`.
        Several recordings are merged into one survey, sorted by time.
    source_files : list[Path], optional
        Source paths stored on the survey. Defaults to the ``.DAT`` paths.
    survey_name : str, optional
        Survey name. Defaults to the first source file's stem.
    sample_rate_hz : float, optional
        Sample rate of the side imaging samples. Defaults to
        :data:`HUMMINBIRD_SAMPLE_RATE_HZ` (see the module docstring);
        the files do not record it.

    Returns
    -------
    SideScanSurvey
        One frequency channel per side imaging frequency, a navigation
        record from the ping headers, empty attitude and DVL records.

    Raises
    ------
    ValueError
        If no recording has a side imaging beam, or ``sample_rate_hz``
        is not positive.
    """
    recs = [recordings] if isinstance(recordings, HumRecording) else list(recordings)
    if not recs:
        raise ValueError("Cannot convert an empty list of Humminbird recordings")
    if sample_rate_hz is not None and not sample_rate_hz > 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    rate = HUMMINBIRD_SAMPLE_RATE_HZ if sample_rate_hz is None else float(sample_rate_hz)
    rate_source = "assumed (PyHum / PING-Mapper pixel size)" if sample_rate_hz is None else "given by caller"
    if source_files is None:
        source_files = [r.path for r in recs]
    if survey_name is None:
        survey_name = Path(source_files[0]).stem if source_files else "untitled_survey"

    pings = [_side_pings(r) for r in recs]
    if not any(p for p in pings):
        raise ValueError("Humminbird input has no side imaging beams (B002/B003)")
    for r in recs:
        skipped = sorted(b for b in r.beams if b not in (BEAM_PORT, BEAM_STARBOARD))
        if skipped:
            logger.warning(
                "%s: down-looking beam(s) %s not converted (one channel each; only side imaging is read)",
                r.path.name, ", ".join(f"{b} ({BEAM_NAMES.get(b, 'unknown')})" for b in skipped),
            )

    survey = SideScanSurvey(source_files=list(source_files), source_format="humminbird",
                            survey_name=survey_name)
    freqs = sorted({f for per_rec in pings for f in per_rec})
    provenance = _provenance(recs, rate_source)
    nav_parts = []
    for freq in freqs:
        parts = [per_rec[freq] for per_rec in pings if freq in per_rec]
        ds = _build_frequency_channel(freq, parts, rate, provenance)
        key = f"hz_{freq}"
        nav_parts.append(ds)
        ds = ds.drop_vars("speed_ms")
        survey.add_frequency(key, ds)
        logger.debug("  %s: %d pings x 2 channels x %d samples", key, ds.sizes["ping"], ds.sizes["sample"])

    survey.navigation = _build_navigation_record(nav_parts)
    survey.attitude = make_empty_attitude_record(source="humminbird_no_attitude")
    survey.dvl = make_empty_dvl_record(source="humminbird_no_dvl")
    return survey


# ============================================================
# PAIRING
# ============================================================

class _Paired:
    """Port/starboard pings of one recording at one frequency, paired by time."""

    def __init__(self, rec: HumRecording, port: HumBeam | None, stbd: HumBeam | None,
                 port_idx: np.ndarray, stbd_idx: np.ndarray):
        self.rec, self.port, self.stbd = rec, port, stbd
        self.port_idx, self.stbd_idx = port_idx, stbd_idx  # -1 where the side is absent

    @property
    def n(self) -> int:
        return int(self.port_idx.size)

    def field(self, name: str) -> np.ndarray:
        """Per-ping header field, from port where present, else starboard."""
        out = np.full(self.n, -1, dtype=np.int64)
        for beam, idx in ((self.stbd, self.stbd_idx), (self.port, self.port_idx)):
            if beam is None or name not in beam.fields:
                continue
            has = idx >= 0
            out[has] = beam.fields[name][idx[has]]
        return out


def _side_pings(rec: HumRecording) -> dict[int, _Paired]:
    """The recording's side imaging pings by frequency (Hz), port and starboard paired by time."""
    port, stbd = rec.beams.get(BEAM_PORT), rec.beams.get(BEAM_STARBOARD)
    freqs = set()
    for beam in (port, stbd):
        if beam is not None:
            freqs.update(np.unique(beam.fields["frequency_hz"]).tolist())
    out = {}
    for freq in sorted(freqs):
        sel = {}
        for side, beam in (("port", port), ("stbd", stbd)):
            if beam is None:
                sel[side] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
                continue
            idx = np.flatnonzero(beam.fields["frequency_hz"] == freq)
            sel[side] = (beam.fields["time_ms"][idx], idx)
        port_idx, stbd_idx = _pair_by_time(*sel["port"], *sel["stbd"])
        out[int(freq)] = _Paired(rec, port, stbd, port_idx, stbd_idx)
    return out


def _pair_by_time(
    t_port: np.ndarray, i_port: np.ndarray, t_stbd: np.ndarray, i_stbd: np.ndarray,
    tol_ms: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair port and starboard pings whose times differ by at most ``tol_ms``.

    Returns per-ping indices into each beam (-1 where that side has no
    ping), in time order. Each ping pairs at most once, with the nearest
    unpaired ping of the other side.
    """
    tol = PAIR_TOLERANCE_MS if tol_ms is None else tol_ms
    op, os_ = np.argsort(t_port, kind="stable"), np.argsort(t_stbd, kind="stable")
    t_port, i_port, t_stbd, i_stbd = t_port[op], i_port[op], t_stbd[os_], i_stbd[os_]
    port_out, stbd_out = [], []
    a = b = 0
    while a < t_port.size or b < t_stbd.size:
        if b >= t_stbd.size or (a < t_port.size and t_port[a] < t_stbd[b] - tol):
            port_out.append(i_port[a])
            stbd_out.append(-1)
            a += 1
        elif a >= t_port.size or t_stbd[b] < t_port[a] - tol:
            port_out.append(-1)
            stbd_out.append(i_stbd[b])
            b += 1
        else:
            port_out.append(i_port[a])
            stbd_out.append(i_stbd[b])
            a += 1
            b += 1
    return np.array(port_out, dtype=np.int64), np.array(stbd_out, dtype=np.int64)


# ============================================================
# FREQUENCY CHANNEL
# ============================================================

def _build_frequency_channel(freq: int, parts: list[_Paired], rate: float, provenance: dict) -> xr.Dataset:
    times = np.concatenate([_ping_times(p) for p in parts])
    order = np.argsort(times, kind="stable")
    n = times.size
    widths = []
    for p in parts:
        for beam, idx in ((p.port, p.port_idx), (p.stbd, p.stbd_idx)):
            if beam is not None and (idx >= 0).any():
                widths.append(int(beam.n_samples[idx[idx >= 0]].max()))
    width = max(widths)
    bs = np.full((n, 2, width), np.nan, dtype=np.float32)
    valid = np.zeros((n, 2), dtype=np.int32)
    row = 0
    for p in parts:
        for ch, (beam, idx) in enumerate(((p.port, p.port_idx), (p.stbd, p.stbd_idx))):
            has = np.flatnonzero(idx >= 0)
            if beam is None or not has.size:
                continue
            samples = beam.samples(idx[has])
            bs[row + has, ch, :samples.shape[1]] = samples
            valid[row + has, ch] = beam.n_samples[idx[has]]
        row += p.n
    one_side = int(np.sum((valid > 0).sum(axis=1) == 1))
    if one_side:
        logger.debug("hz_%d: %d ping(s) recorded on one side only", freq, one_side)

    def cat(name):
        return np.concatenate([p.field(name) for p in parts])

    lon, lat = humminbird_to_lonlat(cat("easting"), cat("northing"))
    depth = cat("depth_dm").astype(np.float32) / 10.0
    no_depth = depth <= 0
    depth[no_depth] = np.nan
    if no_depth.any():
        logger.debug("hz_%d: depth 0 (no bottom lock) in %d of %d pings; set to NaN", freq, int(no_depth.sum()), n)
    heading = (cat("heading_ddeg").astype(np.float32) / 10.0) % 360.0
    speed = cat("speed_dms").astype(np.float32) / 10.0
    record = cat("record_number")
    nan = np.full(n, np.nan, dtype=np.float32)

    attrs = {
        "freq_hz_start": int(freq),
        "freq_hz_end": int(freq),
        "sample_rate_hz": float(rate),
        "pulse_length_us": 0.0,
        "processing_history": json.dumps([]),
        **provenance,
        "humminbird_pings_one_side": one_side,
    }
    coords = {
        "sample": (["sample"], np.arange(width)),
        "ping_number": (["ping"], record[order].astype(np.int32)),
        "time": (["ping"], times[order]),
        "channel_id": (["channel"], np.array(["port", "starboard"])),
    }
    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], bs[order]),
        "valid_samples": (["ping", "channel"], valid[order]),
        "latitude": (["ping"], lat[order]),
        "longitude": (["ping"], lon[order]),
        "heading_deg": (["ping"], heading[order]),
        "depth_m": (["ping"], nan.copy()),
        "roll_deg": (["ping"], nan.copy()),
        "pitch_deg": (["ping"], nan.copy()),
        "heave_m": (["ping"], nan.copy()),
        "altitude_m": (["ping"], depth[order]),
        "sound_vel_ms": (["ping"], nan.copy()),
        "fathometer_reading_m": (["ping"], depth[order].copy()),
        "speed_ms": (["ping"], speed[order], {"units": "m/s", "long_name": "GPS speed over ground, as recorded"}),
    }
    return xr.Dataset(data_vars, coords, attrs)


def _ping_times(p: _Paired) -> np.ndarray:
    start = np.datetime64(int(p.rec.dat.unix_time), "s").astype("datetime64[ns]")
    return start + p.field("time_ms").astype("timedelta64[ms]").astype("timedelta64[ns]")


# ============================================================
# PROVENANCE + AUX
# ============================================================

def _provenance(recs: list[HumRecording], rate_source: str) -> dict:
    sizes = sorted({s for r in recs for b in r.beams.values() for s in b.header_sizes})
    skipped = sorted({b for r in recs for b in r.beams if b not in (BEAM_PORT, BEAM_STARBOARD)})
    return {
        POSITION_SOURCE_ATTR: POSITION_UNKNOWN,
        POSITION_SOURCE_REASON_ATTR: (
            "Humminbird SON headers hold the GPS antenna position; the file does not say where "
            "the transducer is relative to it"
        ),
        "humminbird_recording": "; ".join(r.dat.name or r.path.stem for r in recs),
        "humminbird_sonar_id": "; ".join(sorted({str(r.dat.sonar_id) for r in recs})),
        "humminbird_water_type": "; ".join(sorted({r.dat.water_type for r in recs})),
        "humminbird_dat_size": "; ".join(sorted({str(r.dat.size) for r in recs})),
        "humminbird_header_sizes": ",".join(str(s) for s in sizes),
        "humminbird_beams_skipped": ",".join(str(b) for b in skipped),
        "humminbird_sample_rate_source": rate_source,
        "humminbird_frames_truncated": sum(b.n_truncated for r in recs for b in r.beams.values()),
        "humminbird_resyncs": sum(b.n_resyncs for r in recs for b in r.beams.values()),
    }


def _merged_aux(parts: list[xr.Dataset]) -> xr.Dataset:
    """Per-ping time, position, heading and speed of every frequency, one row per distinct time."""
    t = np.concatenate([p["time"].values for p in parts])
    _, first = np.unique(t, return_index=True)

    def cat(name):
        return np.concatenate([p[name].values for p in parts])[first]

    return xr.Dataset(
        {
            "latitude": (("obs",), cat("latitude").astype(np.float64)),
            "longitude": (("obs",), cat("longitude").astype(np.float64)),
            "heading_deg": (("obs",), cat("heading_deg").astype(np.float32)),
            "speed_knots": (("obs",), (cat("speed_ms") * _MS_TO_KNOTS).astype(np.float32)),
        },
        coords={"time": (("obs",), t[first])},
    )


def _build_navigation_record(parts: list[xr.Dataset]) -> xr.Dataset:
    nav = _merged_aux(parts)
    nav.attrs["source"] = _NAV_SOURCE
    return nav
