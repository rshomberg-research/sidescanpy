"""
Schema validation and xr.Accessor for SideScanPy standard HDF5 format

This module defines the `.ss.h5` (standardized SideScan HDF5) contract that all
datasets must conform to. It provides validation via xr.Accessor and utilities
for tracking processing state.

2026 rshomberg@uri.edu
"""

import logging
import json 
from datetime import datetime, timezone
from typing import Optional, Any 

import xarray as xr 
import numpy as np 

logger = logging.getLogger(__name__)

# CONSTANTS: required schema elements
REQUIRED_DIMS   : set = {"ping", "channel", "sample"}
REQUIRED_COORDS : set = {"ping_number", "time", "channel_id"}
REQUIRED_VARS   : set = {
    "backscatter_raw",
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
}
REQUIRED_ATTRS  : set = {
    "freq_hz_start",
    "freq_hz_end",
    "pulse_length_us",
    "sample_rate_hz",
    "range_m",
    "processing_level",
    "processing_history",
}
KNOWN_PROCESSING_LEVELS : set = {
    "raw", 
    "radiometric", 
    "geometric", 
    "georeferenced",
}

# CONSTANTS: HDF5 group structure
HDF5_GROUP_AUX_STREAMS      : str = "aux_streams"
HDF5_GROUP_NAVIGATION       : str = f"{HDF5_GROUP_AUX_STREAMS}/navigation"
HDF5_GROUP_ATTITUDE         : str = f"{HDF5_GROUP_AUX_STREAMS}/attitude"
HDF5_GROUP_ENVIRONMENTAL    : str = f"{HDF5_GROUP_AUX_STREAMS}/environmental"
HDF5_GROUP_FREQUENCIES      : str = "frequencies"
HDF5_FREQ_PREFIX            : str = "hz_"

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
        5. processing_level is one of KNOWN_PROCESSING_LEVELS

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

        # Check processing level
        level = self._ds.attrs.get("processing_level")
        if level not in KNOWN_PROCESSING_LEVELS:
            raise ValueError(
                f"'processing_level' must be one of {KNOWN_PROCESSING_LEVELS}, got '{level}'"
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
        Get current processing level.

        Returns
        -------
        str
            Value of `ds.attrs['processing_level']`, or 'unknown' if missing.
            One of: 'raw', 'radiometric', 'geometric', 'georeferenced'.

        Examples
        --------
        >>> level = ds.sidescan.processing_level
        >>> if level == 'raw':
        ...     apply_corrections(ds)
        """
        return self._ds.attrs.get("processing_level", "unknown")

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
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "function": function_name,
            "version": version,
        })
        self._ds.attrs["processing_history"] = json.dumps(history)

