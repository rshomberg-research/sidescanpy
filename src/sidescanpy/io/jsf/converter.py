"""
Convert JSFMessage objects into a SideScanSurvey.

The converter takes a flat list of mixed JSFMessages, dispatches by type
internally, groups sonar messages by frequency, and produces a fully
populated SideScanSurvey with one frequency channel per (start, end) pair
plus aux records (navigation, attitude, dvl, sensors).

This module performs no I/O. The reader is responsible for parsing JSF
bytes into JSFMessage instances and for any cross-file message ordering.

2026 rshomberg@uri.edu
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.jsf.messages import (
    JSFMessage,
    SonarDataMessage,
    PitchRollDataMessage,
    DopplerVelocityLogDataMessage,
    PressureSensorReadingMessage,
    NMEAStringMessage,
    NMEARMCMessage,
    NMEAGGAMessage,
    NMEAGLLMessage,
)

from sidescanpy.core.schema import (
    make_empty_navigation_record,
    make_empty_attitude_record,
    make_empty_dvl_record,
    make_empty_sensor_record,
)

logger = logging.getLogger(__name__)

# NOTE: repair is needed when pings do not align for a single channel for a
# single JSF file
RepairMode = Literal["drop", "nan_pad", "abort"]


# ============================================================
# PUBLIC API
# ============================================================

def convert(
    messages: list[JSFMessage],
    *,
    source_files: list[Path],
    survey_name: str | None = None,
    repair_mode: RepairMode = "nan_pad",
) -> SideScanSurvey:
    """
    Convert a list of JSFMessages into a SideScanSurvey.

    Parameters
    ----------
    messages : list of JSFMessage
        Mixed list of all message types from one or more JSF files.
    source_files : list of Path
        File(s) the messages were parsed from. Stored on the survey.
    survey_name : str, optional
        Survey name. Defaults to ``source_files[0].stem``.
    repair_mode : {"drop", "nan_pad", "abort"}, default "nan_pad"
        How to handle unmatched port/starboard pings within a frequency:
        - "drop"    : silently drop orphans
        - "nan_pad" : keep ping, fill missing channel with NaN
        - "abort"   : raise ValueError on first orphan

    Returns
    -------
    SideScanSurvey
        Populated survey with one frequency channel per distinct
        (transmit_freq_start, transmit_freq_end) pair, plus aux records.

    Raises
    ------
    ValueError
        If ``messages`` is empty.
        If file-level attrs (transmit_freq_start/end, sweep_length,
        sample_freq) are not constant within a frequency group.
        If port and starboard sample counts disagree for a paired ping.
        If ``repair_mode == "abort"`` and an orphan ping is found.
    """
    if not messages:
        raise ValueError("Cannot convert empty message list")

    if survey_name is None:
        survey_name = source_files[0].stem

    grouped = _group_by_type(messages)

    survey = SideScanSurvey(
        source_files=source_files,
        source_format="JSF",
        survey_name=survey_name,
    )

    # Frequency channels (one per distinct freq pair)
    by_freq = _group_sonar_by_frequency(grouped["sonar"])
    for freq_key, sonar_msgs in by_freq.items():
        ds = _build_frequency_channel(sonar_msgs, repair_mode=repair_mode)
        survey.add_frequency(freq_key, ds)

    # Aux records (always set, even if empty — caller can check has_*)
    survey.navigation = _build_navigation_record(grouped["nmea"])
    survey.attitude   = _build_attitude_record(grouped["pitch_roll"])
    survey.dvl        = _build_dvl_record(grouped["dvl"])

    if grouped["pressure"]:
        survey.add_sensor("pressure", _build_sensor_record("pressure", grouped["pressure"]))

    return survey


# ============================================================
# DISPATCH BY TYPE
# ============================================================

def _group_by_type(messages: list[JSFMessage]) -> dict[str, list]:
    """Split the mixed message list into typed buckets."""
    grouped: dict[str, list] = {
        "sonar":      [],
        "nmea":       [],
        "pitch_roll": [],
        "dvl":        [],
        "pressure":   [],
        "other":      [],
    }
    for m in messages:
        if isinstance(m, SonarDataMessage):
            grouped["sonar"].append(m)
        elif isinstance(m, NMEAStringMessage):
            grouped["nmea"].append(m)
        elif isinstance(m, PitchRollDataMessage):
            grouped["pitch_roll"].append(m)
        elif isinstance(m, DopplerVelocityLogDataMessage):
            grouped["dvl"].append(m)
        elif isinstance(m, PressureSensorReadingMessage):
            grouped["pressure"].append(m)
        else:
            grouped["other"].append(m)

    if grouped["other"]:
        logger.debug(f"Ignoring {len(grouped['other'])} messages of unhandled types")

    return grouped


def _group_sonar_by_frequency(
    sonar_msgs: list[SonarDataMessage],
) -> dict[str, list[SonarDataMessage]]:
    """Group sonar messages by (transmit_freq_start, transmit_freq_end)."""
    groups: dict[str, list[SonarDataMessage]] = defaultdict(list)
    for m in sonar_msgs:
        # freq_hz_start labels the channel even when start != end (chirp)
        freq_key = f"freq_{int(m.transmit_freq_start)}_hz"
        groups[freq_key].append(m)
    return dict(groups)


# ============================================================
# FREQUENCY CHANNEL BUILDER
# ============================================================

def _build_frequency_channel(
    sonar_msgs: list[SonarDataMessage],
    *,
    repair_mode: RepairMode,
) -> xr.Dataset:
    """Build a schema-valid sidescan dataset from a list of single-frequency
    sonar messages (mixed port + starboard, possibly out of order)."""
    if not sonar_msgs:
        raise ValueError("_build_frequency_channel called with no messages")

    _validate_file_level_attrs(sonar_msgs)

    port_by_ping = {m.ping_number: m for m in sonar_msgs if m.channel == 0}
    stbd_by_ping = {m.ping_number: m for m in sonar_msgs if m.channel == 1}

    paired_pings: list[tuple[SonarDataMessage | None, SonarDataMessage | None]] = _pair_pings_by_number(
        port_by_ping, stbd_by_ping, repair_mode=repair_mode
    )

    # Determine n_samples — must be constant; verified per pair below
    n_samples = _determine_n_samples(paired_pings)

    n_pings = len(paired_pings)
    backscatter = np.full((n_pings, 2, n_samples), np.nan, dtype=np.float32)

    ping_numbers = np.zeros(n_pings, dtype=np.int32)
    timestamps   = np.zeros(n_pings, dtype="int64")  # ns since epoch

    # Per-ping scalars (NaN by default; populated from whichever side is present)
    lat        = np.full(n_pings, np.nan, dtype=np.float64)
    lon        = np.full(n_pings, np.nan, dtype=np.float64)
    heading    = np.full(n_pings, np.nan, dtype=np.float32)
    depth      = np.full(n_pings, np.nan, dtype=np.float32)
    roll       = np.full(n_pings, np.nan, dtype=np.float32)
    pitch      = np.full(n_pings, np.nan, dtype=np.float32)
    altitude   = np.full(n_pings, np.nan, dtype=np.float32)
    sound_vel  = np.full(n_pings, np.nan, dtype=np.float32)
    # heave is not in SonarDataMessage; stays NaN. Real values come later
    # from interpolation against the attitude aux stream.
    heave      = np.full(n_pings, np.nan, dtype=np.float32)

    for i, (port, stbd) in enumerate(paired_pings):
        anchor: SonarDataMessage = port if port is not None else stbd 
        # Both port and stbd carry the same ping_number/timestamp by JSF
        # convention, so taking from either is equivalent.
        ping_numbers[i] = anchor.ping_number
        timestamps[i]   = anchor.timestamp_ns

        if port is not None:
            assert port.N_samples == n_samples, "n_samples mismatch (port)"
            backscatter[i, 0, :] = port.backscatter_intensity
        if stbd is not None:
            assert stbd.N_samples == n_samples, "n_samples mismatch (stbd)"
            backscatter[i, 1, :] = stbd.backscatter_intensity

        # Per-ping scalars — prefer port, fall back to starboard.
        # NOTE: when nan_pad'd (only one side present), the missing side's nav
        # would be near-identical to the present side's nav for the same ping.
        # Currently we just use whichever side exists. If only port exists with
        # NaN nav, the value stays NaN (no fallback-to-stbd nav swap). Revisit
        # if real data shows this loses information.
        scalar_src : SonarDataMessage = port if port is not None else stbd
        lat[i]       = _coerce(scalar_src.latitude)
        lon[i]       = _coerce(scalar_src.longitude)
        heading[i]   = _coerce(scalar_src.heading_deg)
        depth[i]     = _coerce(scalar_src.depth_m)
        roll[i]      = _coerce(scalar_src.roll_deg)
        pitch[i]     = _coerce(scalar_src.pitch_deg)
        altitude[i]  = _coerce(scalar_src.altitude_m)
        sound_vel[i] = _coerce(scalar_src.sound_speed_ms)

    # File-level attrs taken from the first message (already verified constant)
    first = sonar_msgs[0]
    attrs = {
        "freq_hz_start":      int(first.transmit_freq_start),
        "freq_hz_end":        int(first.transmit_freq_end),
        "pulse_length_us":    float(first.sweep_length) * 1e6,
        "sample_rate_hz":     float(first.sample_freq),
        "processing_level":   "raw",
        "processing_history": json.dumps([]),
    }

    ds = xr.Dataset(
        data_vars={
            "backscatter_raw":       (["ping", "channel", "sample"], backscatter),
            "backscatter_intensity": (["ping", "channel", "sample"], backscatter.copy()),
            "latitude":     (["ping"], lat),
            "longitude":    (["ping"], lon),
            "heading_deg":  (["ping"], heading),
            "depth_m":      (["ping"], depth),
            "roll_deg":     (["ping"], roll),
            "pitch_deg":    (["ping"], pitch),
            "heave_m":      (["ping"], heave),
            "altitude_m":   (["ping"], altitude),   # FIXME: calculate altitude from DVL or fathometer+pitch+roll
            "sound_vel_ms": (["ping"], sound_vel),
            "fathometer_reading_m": (["ping"], altitude)
        },
        coords={
            "ping_number": (["ping"], ping_numbers),
            "time":        (["ping"], timestamps.astype("datetime64[ns]")),
            "channel_id":  (["channel"], np.array(["port", "starboard"])),
        },
        attrs=attrs,
    )
    return ds


# ============================================================
# FREQUENCY CHANNEL HELPERS
# ============================================================

_FILE_LEVEL_FIELDS = (
    "transmit_freq_start",
    "transmit_freq_end",
    "sweep_length",
    "sample_freq",
)


def _validate_file_level_attrs(sonar_msgs: list[SonarDataMessage]) -> None:
    """Each file-level attr must be constant across all messages in the group."""
    for field in _FILE_LEVEL_FIELDS:
        values = {getattr(m, field) for m in sonar_msgs}
        if len(values) > 1:
            raise ValueError(
                f"Inconsistent {field} across messages in same frequency group: "
                f"{sorted(values)}. All messages in one frequency channel must "
                f"share file-level acoustic settings."
            )


def _pair_pings_by_number(
    port_by_ping: dict[int, SonarDataMessage],
    stbd_by_ping: dict[int, SonarDataMessage],
    *,
    repair_mode: RepairMode,
) -> list[tuple[SonarDataMessage | None, SonarDataMessage | None]]:
    """
    Match port and starboard messages by ping_number.

    Returns a sorted list of (port_msg|None, stbd_msg|None) tuples. Behaviour
    on orphans depends on repair_mode.
    """
    all_pings = sorted(set(port_by_ping) | set(stbd_by_ping))

    n_orphan = sum(
        1 for p in all_pings
        if (p in port_by_ping) ^ (p in stbd_by_ping)
    )

    if n_orphan > 0:
        if repair_mode == "abort":
            raise ValueError(
                f"Found {n_orphan} incomplete (orphan) ping(s) and repair_mode "
                f"is 'abort'. Use 'drop' or 'nan_pad' to tolerate."
            )
        logger.warning(
            f"{n_orphan} incomplete ping(s) detected; repair_mode='{repair_mode}'"
        )

    paired: list[tuple[SonarDataMessage | None, SonarDataMessage | None]] = []
    for p in all_pings:
        port = port_by_ping.get(p)
        stbd = stbd_by_ping.get(p)
        if port is None or stbd is None:
            if repair_mode == "drop":
                continue
            # nan_pad: keep the pair with one side None
        paired.append((port, stbd))

    return paired


def _determine_n_samples(
    paired_pings: list[tuple[SonarDataMessage | None, SonarDataMessage | None]],
) -> int:
    """Verify n_samples is constant across all messages and return it."""
    sample_lengths: set[int] = set()
    for port, stbd in paired_pings:
        if port is not None:
            sample_lengths.add(int(port.N_samples))
        if stbd is not None:
            sample_lengths.add(int(stbd.N_samples))

    if not sample_lengths:
        raise ValueError("No sonar messages survived pairing")
    if len(sample_lengths) > 1:
        raise ValueError(
            f"Inconsistent sample counts across messages: {sorted(sample_lengths)}. "
            f"Port and starboard must produce the same number of samples per ping."
        )
    return sample_lengths.pop()


def _coerce(value) -> float:    # TODO: remove need for this functions
    """Coerce None to NaN; pass through floats. Once messages.py emits
    NaN directly for invalid fields, this helper becomes a no-op."""
    if value is None:
        return np.nan
    return float(value)


# ============================================================
# AUXILIARY RECORD BUILDERS
# ============================================================


def _build_navigation_record(nmea_msgs: list[NMEAStringMessage]) -> xr.Dataset:
    """Build a navigation record from positional NMEA messages.

    Reads RMC, GGA, GLL — sentence types that carry latitude/longitude.
    Other NMEA types are ignored. Output observations are sorted by time.
    """
    positional: list[tuple[int, float, float]] = []
    for m in nmea_msgs:
        if not isinstance(m, (NMEARMCMessage, NMEAGGAMessage, NMEAGLLMessage)):
            continue
        lat = _coerce(getattr(m, "latitude", None))
        lon = _coerce(getattr(m, "longitude", None))
        ts  = int(m.timestamp) if hasattr(m, "timestamp") else \
              int(np.datetime64(int(m.timestamp * 1e9), "ns").astype("int64"))
        positional.append((ts, lat, lon))

    if not positional:
        return make_empty_navigation_record(source="extracted_from_nmea_messages")
    
    positional.sort(key=lambda t: t[0])
    times = np.array([t[0] for t in positional], dtype="int64").astype("datetime64[ns]")
    lats  = np.array([t[1] for t in positional], dtype=np.float64)
    lons  = np.array([t[2] for t in positional], dtype=np.float64)

    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), lats),
            "longitude": (("obs",), lons),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "extracted_from_nmea_messages"},
    )


def _build_attitude_record(att_msgs: list[PitchRollDataMessage]) -> xr.Dataset:
    """Build an attitude_record from PitchRollDataMessage instances.

    SOURCE SELECTION RATIONALE:
    Attitude is the only direct source for roll/pitch/heave. Sonar embedded
    roll/pitch fields are mirrors of attitude data (1-message lag), so they
    contain no independent information.

    NOTE: validity flags on attitude messages may reject all data even when
    the underlying values are valid (see messages.py TODO). When that
    happens, this record is empty-but-valid, and downstream interpolation
    should fall back to sonar_embedded values for roll/pitch.
    """
    if not att_msgs:
        return make_empty_attitude_record(source="pitch_roll_data_message")
    
    sorted_msgs = sorted(att_msgs, key=lambda m: int(m.timestamp))
    n = len(sorted_msgs)
    times = np.fromiter(
        (int(m.timestamp) for m in sorted_msgs), dtype="int64", count=n
    ).astype("datetime64[ns]")

    return xr.Dataset(
        data_vars={
            "roll_deg":  (("obs",), np.array([_coerce(m.roll_deg)  for m in sorted_msgs], dtype=np.float32)),
            "pitch_deg": (("obs",), np.array([_coerce(m.pitch_deg) for m in sorted_msgs], dtype=np.float32)),
            "heave_m":   (("obs",), np.array([_coerce(m.heave_m)   for m in sorted_msgs], dtype=np.float32)),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "pitch_roll_data_message"},
    )




def _build_dvl_record(dvl_msgs: list[DopplerVelocityLogDataMessage]) -> xr.Dataset:

    if not dvl_msgs:
        return make_empty_dvl_record(source="doppler_velocity_log_data_message")
    
    sorted_msgs = sorted(dvl_msgs, key=lambda m: int(m.timestamp))
    n = len(sorted_msgs)
    times = np.fromiter(
        (int(m.timestamp) for m in sorted_msgs), dtype="int64", count=n
    ).astype("datetime64[ns]")

    return xr.Dataset(
        data_vars={
            "altitude_m": (("obs",), np.array([_coerce(m.altitude_m) for m in sorted_msgs], dtype=np.float32)),
        },
        coords={"time": (("obs",), times)},
        attrs={"source": "doppler_velocity_log_data_message"},
    )


def _build_sensor_record(
    sensor_type: str,
    sensor_msgs: list,
) -> xr.Dataset:
    """Build a generic sensor record. Currently specialised for pressure
    sensor messages (depth_m). Extend with type dispatch as more sensor
    types come online."""
    if not sensor_msgs:
        return make_empty_sensor_record(sensor_type, source="empty")
    
    if sensor_type == "pressure":
        sorted_msgs = sorted(sensor_msgs, key=lambda m: int(m.timestamp))
        n = len(sorted_msgs)
        times = np.fromiter(
            (int(m.timestamp) for m in sorted_msgs), dtype="int64", count=n
        ).astype("datetime64[ns]")

        depth = np.array(
            [_coerce(getattr(m, "depth_m", None)) for m in sorted_msgs],
            dtype=np.float32,
        )

        return xr.Dataset(
            data_vars={
                "depth_m": (("obs",), depth),
            },
            coords={"time": (("obs",), times)},
            attrs={
                "source": "pressure_sensor_reading_message",
                "sensor_type": "pressure",
            },
        )
    
    raise NotImplementedError(
        f"Sensor type '{sensor_type}' is not yet supported by the converter"
    )