"""
Schema validation and xr.Accessor for SideScanPy standard HDF5 format

This module defines the `.ss.h5` (standardized SideScan HDF5) contract that all
datasets must conform to. It provides validation via xr.Accessor and utilities
for tracking processing state.

2026 rshomberg@uri.edu
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from sidescanpy.core.history import PROCESSING_LEVELS
from sidescanpy.core.history import processing_level as _derived_processing_level

logger = logging.getLogger(__name__)

# CONSTANTS: required schema elements
REQUIRED_DIMS   : set = {"ping", "channel", "sample"}
REQUIRED_COORDS : set = {"ping_number", "time", "channel_id"}
# NOTE: Per-ping fields. Notes on closely-related fields:
#
#   fathometer_reading_m
#       Raw altimeter measurement. Distance from transducer to bottom along
#       the altimeter's beam axis. Affected by platform attitude (pitch, roll)
#       — when the platform tilts, the beam tilts with it.
#
#   altitude_m
#       Vertical distance from transducer to bottom. Should be attitude-
#       corrected. For now (until DVL is properly supported and attitude
#       correction is implemented), altitude_m == fathometer_reading_m as a
#       conservative simplification. Functions that need accurate vertical
#       altitude should use altitude_m and we'll fix the source later.
#
#   depth_m
#       Vertical distance from sea surface to transducer. Independent of
#       altitude_m and fathometer_reading_m.
REQUIRED_VARS   : set = {
    "backscatter_intensity",
    "latitude",
    "longitude",
    "heading_deg",
    "depth_m",
    "roll_deg",
    "pitch_deg",
    "heave_m",
    "altitude_m",
    "sound_vel_ms",
    "fathometer_reading_m",
}
# Optional per-ping variables, written by a reader when the format provides
# them (see each reader's docs for the source fields):
#
#   layback_m
#       Horizontal distance from the tow point back to the towfish, metres,
#       as recorded by the acquisition software. 0 can mean "not set".
#   cable_out_m
#       Tow cable paid out, metres.
#   latitude_before_layback, longitude_before_layback
#       Positions as recorded, kept by the ``layback`` step when it moves
#       ``latitude`` / ``longitude`` so the step can be undone.
OPTIONAL_PER_PING_VARS : dict[str, str] = {
    "layback_m": "horizontal distance from tow point to towfish (m), as recorded",
    "cable_out_m": "tow cable out (m), as recorded",
    "latitude_before_layback": "latitude as recorded, before the layback step",
    "longitude_before_layback": "longitude as recorded, before the layback step",
}

# What the per-ping latitude/longitude are the position of. Dataset attribute
# POSITION_SOURCE_ATTR holds one of POSITION_SOURCES, and
# POSITION_SOURCE_REASON_ATTR a short sentence on how the reader decided.
# A missing attribute means unknown.
#   ship     the navigation antenna (or the last nav fix) -- layback not applied
#   towfish  the sonar, already laid back by the acquisition software
#   mixed    differs from ping to ping
#   unknown  the file does not say
POSITION_SOURCE_ATTR        : str = "position_source"
POSITION_SOURCE_REASON_ATTR : str = "position_source_reason"
POSITION_SHIP    : str = "ship"
POSITION_TOWFISH : str = "towfish"
POSITION_MIXED   : str = "mixed"
POSITION_UNKNOWN : str = "unknown"
POSITION_SOURCES : tuple = (POSITION_SHIP, POSITION_TOWFISH, POSITION_MIXED, POSITION_UNKNOWN)


def combine_position_sources(per_ping: np.ndarray | list[str]) -> str:
    """One :data:`POSITION_SOURCES` value for a set of per-ping sources.

    All pings the same → that value; empty → unknown; otherwise mixed.
    """
    values = set(np.asarray(per_ping, dtype=object).tolist())
    if not values:
        return POSITION_UNKNOWN
    if len(values) == 1:
        return str(values.pop())
    return POSITION_MIXED


REQUIRED_ATTRS  : set = {
    "freq_hz_start",
    "freq_hz_end",
    "pulse_length_us",
    "sample_rate_hz",
}
KNOWN_PROCESSING_LEVELS : tuple = PROCESSING_LEVELS
"""Values :func:`sidescanpy.core.history.processing_level` can return."""
LEGACY_PROCESSING_LEVEL_VALUES : set = {"raw", "radiometric"}
"""Values the legacy ``processing_level`` attribute may hold on older files.
The attribute is no longer written; the level is derived from history."""


# CONSTANTS: HDF5 group structure
HDF5_GROUP_AUX_STREAMS  : str = "aux_streams"
HDF5_GROUP_NAVIGATION   : str = f"{HDF5_GROUP_AUX_STREAMS}/navigation"
HDF5_GROUP_ATTITUDE     : str = f"{HDF5_GROUP_AUX_STREAMS}/attitude"
HDF5_GROUP_DVL          : str = f"{HDF5_GROUP_AUX_STREAMS}/dvl"
HDF5_GROUP_SENSORS      : str = f"{HDF5_GROUP_AUX_STREAMS}/sensors"
HDF5_GROUP_FREQUENCIES  : str = "frequencies"
HDF5_FREQ_PREFIX        : str = "hz_"

# .ss.h5 format identity (root attributes). The format version is
# separate from the package version and changes only when the on-disk
# layout does; 0.x until the layout is frozen as 1.0.
SS_FORMAT_NAME          : str = "sidescanpy-ss"
SS_FORMAT_VERSION       : str = "0.1"

# CONSTANTS: I/O configuration
HDF5_CHUNK_PING         : int = 256
HDF5_COMPRESSION        : str = 'gzip'
HDF5_COMPRESSION_LEVEL  : int = 4
# ???: currently arbitrary choices. Consider changing

@xr.register_dataset_accessor("sidescan")
class SideScanAccessor:
    """
    xarray.Dataset accessor for sidescan specific validation and utilities

    Provides methods to validate datasets against the SideScanPy schema, track
    processing state, and compute derived quantities like slant range.

    Accessed via `ds.sidescan` on any xarray.Dataset

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.Dataset(...)
    >>> ds.sidescan.validate()  # Raises ValueError if invalid
    >>> ds.sidescan.is_valid()  # Returns bool
    >>> ds.sidescan.processing_level  # Returns str
    """

    _ds : xr.Dataset

    def __init__(self, ds: xr.Dataset) -> None:
        """Initialize accessor"""
        self._ds = ds

    def validate(self) -> None:
        """
        Validate dataset against SideScanPy schema.

        Checks (in order):

        1. All required dimensions present and channel size == 2
        2. All required coordinates present
        3. All required variables present
        4. All required attributes present
        5. A legacy ``processing_level`` attribute, if present, holds a
           legacy value (``raw``/``radiometric``)

        Raises
        ------
        ValueError
            On first schema violation with specific, actionable message.

        Examples
        --------
        >>> ds.sidescan.validate()  # Raises if invalid, returns None if valid
        """
        # Check dimensions
        missing_dims = REQUIRED_DIMS - set(self._ds.dims)
        if missing_dims:
            raise ValueError(f"Missing dimensions: {missing_dims}")

        if self._ds.sizes.get("channel") != 2:
            raise ValueError(
                f"'channel' dimension must be size 2, got {self._ds.sizes.get('channel', 'missing')}"
            )

        # Check coordinates
        missing_coords = REQUIRED_COORDS - set(self._ds.coords)
        if missing_coords:
            raise ValueError(f"Missing coordinates: {missing_coords}")

        # Check variables
        missing_vars = REQUIRED_VARS - set(self._ds.data_vars)
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")

        # Check attributes
        missing_attrs = REQUIRED_ATTRS - set(self._ds.attrs)
        if missing_attrs:
            raise ValueError(f"Missing attributes: {missing_attrs}")

        # Legacy processing_level attribute (optional; derived from history now)
        level = self._ds.attrs.get("processing_level")
        if level is not None and level not in LEGACY_PROCESSING_LEVEL_VALUES:
            raise ValueError(
                f"legacy 'processing_level' attribute must be one of "
                f"{sorted(LEGACY_PROCESSING_LEVEL_VALUES)}, got '{level}'"
            )

    def is_valid(self) -> bool:
        """
        Check if dataset is valid without raising.

        Returns
        -------
        bool
            True if dataset passes all validations, False otherwise.

        Examples
        --------
        >>> if ds.sidescan.is_valid():
        ...     process(ds)
        """
        try:
            self.validate()
            return True
        except ValueError:
            return False

    @property
    def processing_level(self) -> str:
        """
        Current processing level, derived from ``processing_history``.

        See :func:`sidescanpy.core.history.processing_level`.
        """
        return _derived_processing_level(self._ds)

    @property
    def processing_history(self) -> list[dict[str, Any]]:
        """
        Parse and return processing history.

        Parses JSON from `ds.attrs['processing_history']` into a list of
        dictionaries. Each entry has keys:
        - 'step': str, name of processing step
        - 'timestamp_utc': str, ISO 8601 timestamp
        - 'params': dict, parameters used in step
        - 'function': str, fully qualified function name
        - 'version': str, semantic version of function

        Returns
        -------
        list of dict
            Processing history entries, or empty list if missing/malformed.

        Examples
        --------
        >>> history = ds.sidescan.processing_history
        >>> for entry in history:
        ...     print(f"{entry['step']} at {entry['timestamp_utc']}")
        """
        raw = self._ds.attrs.get("processing_history", "[]")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse processing_history; returning empty list")
            return []

    def append_to_history(
        self,
        step_name: str,
        function_name: str,
        params: dict[str, Any],
        version: str,
    ) -> None:
        """
        Append an entry to processing history.

        Creates a new history entry with current timestamp and updates
        `ds.attrs['processing_history']` as JSON string. Preserves all
        prior entries (append-only).

        Parameters
        ----------
        step_name : str
            Name of processing step (e.g., 'tvg_correction').
        function_name : str
            Fully qualified function name (e.g., 'sidescanpy.processing.radiometric.apply_tvg').
        params : dict
            Parameters used in this step. Can be any JSON-serializable dict.
        version : str
            Semantic version of the function (e.g., '0.2.0').

        Examples
        --------
        >>> ds.sidescan.append_to_history(
        ...     step_name='tvg_correction',
        ...     function_name='sidescanpy.processing.radiometric.apply_tvg',
        ...     params={'slope_db_per_m': 30.0, 'offset_db': 20.0},
        ...     version='0.2.0',
        ... )
        """
        history = self.processing_history
        history.append({
            "step": step_name,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "params": params,
            "function": function_name,
            "version": version,
        })
        self._ds.attrs["processing_history"] = json.dumps(history)


# ============================================================
# AUXILIARY STREAM SCHEMAS
# ============================================================

# TODO: write tests for aux stream schemas

# --- Navigation Record ---

REQUIRED_NAV_DIMS   : set = {"obs"}
REQUIRED_NAV_COORDS : set = {"time"}
REQUIRED_NAV_VARS   : set = {"latitude", "longitude"}
REQUIRED_NAV_ATTRS  : set = {"source"}


@xr.register_dataset_accessor("navigation_record")
class NavigationRecordAccessor:
    """
    xarray.Dataset accessor for navigation auxiliary streams.

    A navigation record is a time series of position fixes, typically extracted
    from NMEA messages (RMC, GLL, GGA) or fallback-extracted from per-ping
    embedded fields. Stored as separate aux stream so it can be re-interpolated
    onto ping times later.

    Required dims: {obs}
    Required coords: {time} (datetime64[ns])
    Required vars: {latitude, longitude}
    Optional vars: heading_deg, speed_knots, course_deg
    Required attrs: {source}

    Examples
    --------
    >>> nav.navigation_record.validate()
    >>> nav.navigation_record.is_empty()
    """

    _ds: xr.Dataset

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def validate(self) -> None:
        missing_dims = REQUIRED_NAV_DIMS - set(self._ds.dims)
        if missing_dims:
            raise ValueError(f"Navigation: missing dimensions: {missing_dims}")

        missing_coords = REQUIRED_NAV_COORDS - set(self._ds.coords)
        if missing_coords:
            raise ValueError(f"Navigation: missing coordinates: {missing_coords}")

        missing_vars = REQUIRED_NAV_VARS - set(self._ds.data_vars)
        if missing_vars:
            raise ValueError(f"Navigation: missing variables: {missing_vars}")

        missing_attrs = REQUIRED_NAV_ATTRS - set(self._ds.attrs)
        if missing_attrs:
            raise ValueError(f"Navigation: missing attributes: {missing_attrs}")

        if self._ds["time"].dtype != np.dtype("datetime64[ns]"):
            raise ValueError(
                f"Navigation: 'time' must be datetime64[ns], got {self._ds['time'].dtype}"
            )

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        """Return True if the dataset has zero observations."""
        return self._ds.sizes.get("obs", 0) == 0


# --- Attitude Record ---

REQUIRED_ATT_DIMS   : set = {"obs"}
REQUIRED_ATT_COORDS : set = {"time"}
REQUIRED_ATT_VARS   : set = {"roll_deg", "pitch_deg", "heave_m"}
OPTIONAL_ATT_VARS   : set = {"heading_deg", "yaw_deg", "temperature_c"}
REQUIRED_ATT_ATTRS  : set = {"source"}


@xr.register_dataset_accessor("attitude_record")
class AttitudeRecordAccessor:
    """
    Auxiliary stream for attitude data (roll, pitch, heave).

    Typically sourced from PitchRollDataMessage (JSF type 2020) or XTF ping
    headers.

    Required dims: {obs}
    Required coords: {time} (datetime64[ns], same timebase as ping times)
    Required vars: {roll_deg, pitch_deg, heave_m}
    Optional vars: heading_deg (compass bearing, degrees clockwise from
    north), yaw_deg, temperature_c
    Required attrs: {source}

    A NaN value marks a reading the source flagged invalid or did not carry.
    """

    _ds: xr.Dataset

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def validate(self) -> None:
        missing_dims = REQUIRED_ATT_DIMS - set(self._ds.dims)
        if missing_dims:
            raise ValueError(f"Attitude: missing dimensions: {missing_dims}")

        missing_coords = REQUIRED_ATT_COORDS - set(self._ds.coords)
        if missing_coords:
            raise ValueError(f"Attitude: missing coordinates: {missing_coords}")

        missing_vars = REQUIRED_ATT_VARS - set(self._ds.data_vars)
        if missing_vars:
            raise ValueError(f"Attitude: missing variables: {missing_vars}")

        missing_attrs = REQUIRED_ATT_ATTRS - set(self._ds.attrs)
        if missing_attrs:
            raise ValueError(f"Attitude: missing attributes: {missing_attrs}")

        if self._ds["time"].dtype != np.dtype("datetime64[ns]"):
            raise ValueError(
                f"Attitude: 'time' must be datetime64[ns], got {self._ds['time'].dtype}"
            )

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        return self._ds.sizes.get("obs", 0) == 0


# --- DVL Record ---

REQUIRED_DVL_DIMS   : set = {"obs"}
REQUIRED_DVL_COORDS : set = {"time"}
REQUIRED_DVL_VARS   : set = {"altitude_m"}
REQUIRED_DVL_ATTRS  : set = {"source"}


@xr.register_dataset_accessor("dvl_record")
class DVLRecordAccessor:
    """
    Auxiliary stream for DVL altimeter data.

    Sourced from DopplerVelocityLogDataMessage (JSF type 2080). Optional vars
    include bottom_vel_x/y/z, water_vel_x/y/z, sound_speed_ms, temperature_c.
    """

    _ds: xr.Dataset

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def validate(self) -> None:
        missing_dims = REQUIRED_DVL_DIMS - set(self._ds.dims)
        if missing_dims:
            raise ValueError(f"DVL: missing dimensions: {missing_dims}")

        missing_coords = REQUIRED_DVL_COORDS - set(self._ds.coords)
        if missing_coords:
            raise ValueError(f"DVL: missing coordinates: {missing_coords}")

        missing_vars = REQUIRED_DVL_VARS - set(self._ds.data_vars)
        if missing_vars:
            raise ValueError(f"DVL: missing variables: {missing_vars}")

        missing_attrs = REQUIRED_DVL_ATTRS - set(self._ds.attrs)
        if missing_attrs:
            raise ValueError(f"DVL: missing attributes: {missing_attrs}")

        if self._ds["time"].dtype != np.dtype("datetime64[ns]"):
            raise ValueError(
                f"DVL: 'time' must be datetime64[ns], got {self._ds['time'].dtype}"
            )

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        return self._ds.sizes.get("obs", 0) == 0


# --- Generic Sensor Record ---
# Catch-all for additional time-series streams. Pressure/depth from the
# pressure sensor message is a candidate to test against DVL altitude.

REQUIRED_GEN_DIMS   : set = {"obs"}
REQUIRED_GEN_COORDS : set = {"time"}
REQUIRED_GEN_VARS   : set = set()  # no required vars — depends on sensor
REQUIRED_GEN_ATTRS  : set = {"source", "sensor_type"}


@xr.register_dataset_accessor("sensor_record")
class SensorRecordAccessor:
    """
    Generic auxiliary stream for arbitrary time-series sensor data.

    Use for streams that don't fit the navigation/attitude/dvl categories,
    e.g. pressure-derived depth, water temperature, conductivity.

    Required dims: {obs}
    Required coords: {time}
    Required vars: (none — depends on sensor)
    Required attrs: {source, sensor_type}
    """

    _ds: xr.Dataset

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def validate(self) -> None:
        missing_dims = REQUIRED_GEN_DIMS - set(self._ds.dims)
        if missing_dims:
            raise ValueError(f"Sensor: missing dimensions: {missing_dims}")

        missing_coords = REQUIRED_GEN_COORDS - set(self._ds.coords)
        if missing_coords:
            raise ValueError(f"Sensor: missing coordinates: {missing_coords}")

        missing_attrs = REQUIRED_GEN_ATTRS - set(self._ds.attrs)
        if missing_attrs:
            raise ValueError(f"Sensor: missing attributes: {missing_attrs}")

        if self._ds["time"].dtype != np.dtype("datetime64[ns]"):
            raise ValueError(
                f"Sensor: 'time' must be datetime64[ns], got {self._ds['time'].dtype}"
            )

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        return self._ds.sizes.get("obs", 0) == 0

    # ============================================================
# EMPTY RECORD BUILDERS
# ============================================================
# These produce zero-observation, schema-valid datasets. Used by the
# converter (when no source messages exist) and by the HDF5 writer
# (which always writes nav/attitude/dvl groups, even when unset).

def make_empty_navigation_record(source: str = "empty") -> xr.Dataset:
    """Zero-obs valid navigation_record."""
    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), np.array([], dtype=np.float64)),
            "longitude": (("obs",), np.array([], dtype=np.float64)),
        },
        coords={"time": (("obs",), np.array([], dtype="datetime64[ns]"))},
        attrs={"source": source},
    )


def make_empty_attitude_record(source: str = "empty") -> xr.Dataset:
    """Zero-obs valid attitude_record."""
    return xr.Dataset(
        data_vars={
            "roll_deg":  (("obs",), np.array([], dtype=np.float32)),
            "pitch_deg": (("obs",), np.array([], dtype=np.float32)),
            "heave_m":   (("obs",), np.array([], dtype=np.float32)),
        },
        coords={"time": (("obs",), np.array([], dtype="datetime64[ns]"))},
        attrs={"source": source},
    )


def make_empty_dvl_record(source: str = "empty") -> xr.Dataset:
    """Zero-obs valid dvl_record."""
    return xr.Dataset(
        data_vars={
            "altitude_m": (("obs",), np.array([], dtype=np.float32)),
        },
        coords={"time": (("obs",), np.array([], dtype="datetime64[ns]"))},
        attrs={"source": source},
    )


def make_empty_sensor_record(
    sensor_type: str,
    source: str = "empty",
    *,
    var_name: str = "depth_m",
) -> xr.Dataset:
    """Zero-obs valid sensor_record."""
    return xr.Dataset(
        data_vars={
            var_name: (("obs",), np.array([], dtype=np.float32)),
        },
        coords={"time": (("obs",), np.array([], dtype="datetime64[ns]"))},
        attrs={"source": source, "sensor_type": sensor_type},
    )
