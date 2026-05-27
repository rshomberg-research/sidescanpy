"""
Convert native JSF HDF5 archives to standard sonar schema HDF5.

Reads `.jsf.h5` archives (from Phase 4a), converts to standard `.ss.h5` format,
and stores auxiliary sensor streams in the same file.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional
import json

import xarray as xr
import numpy as np
import h5py
from scipy.interpolate import interp1d

from sidescanpy.io.standard.writer import write_ss_hdf5

logger = logging.getLogger(__name__)

# BUG: sonar channels are not handled separately - fixed?

def native_hdf5_to_standard(
    native_hdf5_path: Path,
    output_path: Path,
    freq_hz: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert native JSF HDF5 to standard sonar schema HDF5.

    Reads all frequencies from `.jsf.h5`, converts each to standard schema,
    writes to `.ss.h5`, and stores auxiliary streams in the same file.

    Parameters
    ----------
    native_hdf5_path : Path
        Input native `.jsf.h5` file path.
    output_path : Path
        Output standard `.ss.h5` file path.
    freq_hz : int, optional
        If specified, convert only this frequency. If None, convert all.
    overwrite : bool
        If False (default), raise if output exists. If True, overwrite.

    Raises
    ------
    FileNotFoundError
        If input file does not exist.
    FileExistsError
        If output file exists and overwrite=False.
    ValueError
        If specified frequency not found or validation fails.

    Examples
    --------
    >>> native_hdf5_to_standard("survey.jsf.h5", "survey.ss.h5")
    >>> native_hdf5_to_standard("survey.jsf.h5", "survey.ss.h5", freq_hz=1495000)
    """
    native_hdf5_path = Path(native_hdf5_path)
    output_path = Path(output_path)

    if not native_hdf5_path.exists():
        raise FileNotFoundError(f"Native HDF5 file not found: {native_hdf5_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file exists: {output_path}. Use overwrite=True to replace."
        )

    logger.info(f"Converting {native_hdf5_path} to {output_path}")

    # Get list of frequencies
    available_freqs = list_frequencies_in_native_hdf5(native_hdf5_path)
    logger.info(f"Found frequencies: {available_freqs}")

    # Determine which frequencies to convert
    if freq_hz is not None:
        if freq_hz not in available_freqs:
            raise ValueError(
                f"Frequency {freq_hz} not found. Available: {available_freqs}"
            )
        freqs_to_convert = [freq_hz]
    else:
        freqs_to_convert = available_freqs

    logger.info(f"Converting {len(freqs_to_convert)} frequency(ies)")

    datasets_by_freq = []

    # Convert each frequency
    ping_times = None
    for idx, freq in enumerate(freqs_to_convert, 1):
        logger.info(f"Converting frequency {freq} Hz ({idx}/{len(freqs_to_convert)})")
        

        # Read directly from native HDF5 to xarray
        ds = _read_frequency_to_dataset(native_hdf5_path, freq)

        # ===== DIAGNOSTIC INFO =====
        n_pings = ds.sizes['ping']
        n_channels = ds.sizes.get('channel', 0)
        n_samples = ds.sizes.get('sample', 0)
        
        logger.info(f"  ✓ Loaded: {n_pings} pings × {n_channels} channels × {n_samples} samples")
        
        # Check for duplicates
        bs_shape = ds['backscatter_intensity'].shape
        logger.debug(f"    Backscatter shape: {bs_shape}")
        
        # Verify channels are different
        if n_channels == 2:
            port = ds['backscatter_intensity'].values[:, 0, :]
            stbd = ds['backscatter_intensity'].values[:, 1, :]
            are_same = np.allclose(port, stbd, equal_nan=True)
            logger.info(f"    Port == Starboard? {are_same}")
        
        datasets_by_freq.append((freq_hz, ds))
        

        # Save ping times from first frequency for auxiliary interpolation
        if ping_times is None:
            ping_times = ds.coords["time"].values

        # Add encoding to avoid chunking issues with small datasets
        encoding = {}
        for var_name in ds.data_vars:
            encoding[var_name] = {"zlib": True, "complevel": 4}
        
        for coord_name in ds.coords:
            encoding[coord_name] = {"zlib": True, "complevel": 4}

        # Check consistency BEFORE writing
        logger.info("\nValidating dataset consistency...")
        first_n_pings = datasets_by_freq[0][1].sizes['ping']
        
        for freq_hz, ds in datasets_by_freq:
            n_pings = ds.sizes['ping']
            if n_pings != first_n_pings:
                logger.error(f"MISMATCH: {freq_hz} Hz has {n_pings} pings, but first was {first_n_pings}")
                logger.error(f"This suggests inconsistent channel separation between frequencies!")
                raise ValueError(
                    f"Frequency {freq_hz} has {n_pings} pings but expected {first_n_pings}. "
                    "Channel separation may be inconsistent."
                )
            logger.info(f"  ✓ {freq_hz} Hz: {n_pings} pings")
            
        # Validate and write to standard HDF5
        ds.sidescan.validate()
        write_ss_hdf5(
            ds,
            output_path,
            freq_hz=freq,
            overwrite=(idx > 0 or overwrite),
        )

    # Extract and store auxiliary streams in the same file
    logger.info("Extracting auxiliary streams...")
    if ping_times is not None:
        _extract_and_store_auxiliaries_infile(native_hdf5_path, output_path, ping_times)

    logger.info(f"Successfully converted to {output_path}")


def list_frequencies_in_native_hdf5(native_hdf5_path: Path) -> list[int]:
    """
    List all frequency groups in native HDF5 archive.

    Returns
    -------
    list[int]
        Frequency values in Hz, sorted descending.
    """
    native_hdf5_path = Path(native_hdf5_path)

    with h5py.File(native_hdf5_path, "r") as f:
        if "message_types/SonarDataMessage" not in f:  # type: ignore
            return []

        sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
        freq_groups = list(sonar_group.keys())  # type: ignore

        freqs = []
        for group_name in freq_groups:
            if group_name.startswith("hz_"):
                freq_str = group_name[3:]
                try:
                    freq = int(freq_str)
                    freqs.append(freq)
                except ValueError:
                    logger.warning(f"Could not parse frequency: {group_name}")

    return sorted(freqs, reverse=True)

def _read_frequency_to_dataset(
    native_hdf5_path: Path,
    freq_hz: int,
) -> xr.Dataset:
    """
    Read and convert single frequency from native HDF5 to standard schema Dataset.
    """
    native_hdf5_path = Path(native_hdf5_path)
    freq_key = f"hz_{freq_hz}"
    group_path = f"message_types/SonarDataMessage/{freq_key}"

    logger.debug(f"Reading {group_path} from {native_hdf5_path}")

    with h5py.File(native_hdf5_path, "r") as f:
        sonar_group = f[group_path]  # type: ignore

        # Read all fields from HDF5
        archive_data = {}
        for field_name in sonar_group.keys():  # type: ignore
            ds = sonar_group[field_name]  # type: ignore
            archive_data[field_name] = ds[:]

        # ========== CRITICAL FIX ==========
        # Data is interleaved: [ping0_port, ping0_stbd, ping1_port, ping1_stbd, ...]
        channels = archive_data.get("channel")
        if channels is not None:
            # Get actual number of unique pings
            unique_pings = np.unique(archive_data["ping_number"])
            n_pings = len(unique_pings)
            
            logger.debug(f"Found {len(archive_data['ping_number'])} records ({n_pings} pings × 2 channels)")
            
            # Separate port and starboard
            port_mask = channels == 0
            stbd_mask = channels == 1
            
            port_indices = np.where(port_mask)[0]
            stbd_indices = np.where(stbd_mask)[0]
        else:
            raise ValueError("No 'channel' field in HDF5 data - cannot separate port/starboard")
    
        # Process backscatter data (stored as JSON strings)
        backscatter_raw = archive_data.get("backscatter_intensity")
        if backscatter_raw is not None:
            # Convert JSON strings to arrays
            try:
                import json
                
                # Process ALL backscatter records first
                bs_arrays = []
                sample_lengths = []
                
                for bs_json in backscatter_raw:
                    if isinstance(bs_json, bytes):
                        bs_json = bs_json.decode('utf-8')
                    
                    if isinstance(bs_json, str):
                        # Parse JSON string to list
                        bs_list = json.loads(bs_json)
                        bs_array = np.array(bs_list, dtype=np.float32)
                    else:
                        bs_array = np.asarray(bs_json, dtype=np.float32)
                    
                    bs_arrays.append(bs_array)
                    sample_lengths.append(len(bs_array))
                
                n_samples = int(np.max(sample_lengths))
                
                # Pad ragged arrays to consistent size
                bs_array_padded = np.full((len(bs_arrays), n_samples), np.nan, dtype=np.float32)
                for i, bs in enumerate(bs_arrays):
                    bs_array_padded[i, :len(bs)] = bs
                
                logger.debug(f"  Backscatter: {n_pings} pings × 2 channels × {n_samples} samples")
                print(f"  Backscatter: {n_pings} pings × 2 channels × {n_samples} samples")
                
                # NOW separate into port and starboard
                backscatter_3d = np.full((n_pings, 2, n_samples), np.nan, dtype=np.float32)
                backscatter_3d[:, 0, :] = bs_array_padded[port_indices, :]  # Port
                backscatter_3d[:, 1, :] = bs_array_padded[stbd_indices, :]  # Starboard

            except Exception as e:
                logger.warning(f"Failed to process backscatter: {e}")
                n_samples = int(archive_data.get("N_samples", [2000])[0])
                backscatter_3d = np.full((n_pings, 2, n_samples), np.nan, dtype=np.float32)
        else:
            n_samples = int(archive_data.get("N_samples", [2000])[0])
            backscatter_3d = np.full((n_pings, 2, n_samples), np.nan, dtype=np.float32)

        # Build data variables with field name mapping
        data_vars = {}
        coords = {}

        # For scalar fields, we need to handle interleaved data
        # Take only the port channel data (first occurrence of each ping)
        port_data_indices = port_indices
        
        # Direct mappings (name stays same) - use port channel only
        for field_name in ["latitude", "longitude"]:
            if field_name in archive_data:
                data_vars[field_name] = (["ping"], archive_data[field_name][port_data_indices].astype(np.float32))

        # Mapped field names - use port channel only
        field_mapping = {
            "heading": "heading_deg",
            "pitch": "pitch_deg",
            "roll": "roll_deg",
            "depth": "depth_m",
            "altitude": "altitude_m",
        }

        for archive_name, schema_name in field_mapping.items():
            if archive_name in archive_data:
                data_vars[schema_name] = (["ping"], archive_data[archive_name][port_data_indices].astype(np.float32))

        # Backscatter data
        data_vars["backscatter_intensity"] = (["ping", "channel", "sample"], backscatter_3d)
        data_vars["backscatter_raw"] = (["ping", "channel", "sample"], backscatter_3d.copy())

        # Convert timestamp to datetime64 - use port channel only
        if "timestamp" in archive_data:
            timestamps = archive_data["timestamp"][port_data_indices]
            try:
                time_ns = np.asarray(timestamps, dtype=np.int64)
                time_dt64 = time_ns.astype("datetime64[ns]")
                coords["time"] = (["ping"], time_dt64)
            except Exception as e:
                logger.warning(f"Failed to convert timestamps: {e}")
                coords["time"] = (["ping"], np.arange(n_pings, dtype=np.int64))
        else:
            coords["time"] = (["ping"], np.arange(n_pings, dtype=np.int64))

        # Ping number as coordinate - use port channel only
        if "ping_number" in archive_data:
            coords["ping_number"] = (["ping"], archive_data["ping_number"][port_data_indices].astype(np.int32))

        # Dimension coordinates
        coords["ping"] = np.arange(n_pings)
        coords["channel_id"] = (["channel"], np.array(["port", "starboard"]))
        coords["sample"] = np.arange(n_samples)

        # Add all required variables from schema
        required_vars = [
            "latitude", "longitude", "heading_deg", "pitch_deg", "roll_deg",
            "depth_m", "altitude_m", "heave_m", "sound_vel_ms"
        ]

        for var_name in required_vars:
            if var_name not in data_vars:
                logger.debug(f"Filling missing variable {var_name} with NaN")
                data_vars[var_name] = (["ping"], np.full(n_pings, np.nan, dtype=np.float32))

        # Calculate required attributes from archive data (use port data)
        attrs = {
            "freq_hz_start": freq_hz,
            "freq_hz_end": freq_hz,
            "processing_level": "raw",
            "processing_history": json.dumps([]),
        }

        # Sample rate in Hz
        if "sample_freq" in archive_data:
            sample_rate = float(archive_data["sample_freq"][port_data_indices[0]])
            attrs["sample_rate_hz"] = sample_rate
        else:
            attrs["sample_rate_hz"] = 1000.0  # Default

        # Pulse length in microseconds
        if "transmit_freq_start" in archive_data:
            sweep_len = float(archive_data.get("sweep_length", [1.0])[port_data_indices[0]])
            if sweep_len > 0:
                pulse_length_us = (sweep_len * 1e6)
                attrs["pulse_length_us"] = pulse_length_us
            else:
                attrs["pulse_length_us"] = 100.0
        else:
            attrs["pulse_length_us"] = 100.0

        # Range in meters
        if "sound_speed" in archive_data:
            sound_speed = float(archive_data["sound_speed"][port_data_indices[0]])
        else:
            sound_speed = 1500.0
        
        if "sample_interval" in archive_data:
            sample_interval_us = float(archive_data["sample_interval"][port_data_indices[0]])
            max_range_m = (sound_speed * sample_interval_us * n_samples / 2) / 1e6
            attrs["range_m"] = max_range_m
        else:
            attrs["range_m"] = 100.0

        # Create dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )

    logger.debug(f"Loaded dataset: {dict(ds.sizes)}")
    return ds

def _extract_and_store_auxiliaries_infile(
    native_hdf5_path: Path,
    output_path: Path,
    ping_times: np.ndarray,
) -> None:
    """
    Extract auxiliary sensor data from native HDF5 and store in output file.

    Writes to auxiliary stream groups within the same .ss.h5 file.

    Parameters
    ----------
    native_hdf5_path : Path
        Input native archive file.
    output_path : Path
        Output standard file (appends to this file).
    ping_times : np.ndarray
        Ping times for interpolation reference.
    """
    native_hdf5_path = Path(native_hdf5_path)
    output_path = Path(output_path)

    with h5py.File(native_hdf5_path, "r") as f_in:
        if "message_types" not in f_in:  # type: ignore
            logger.debug("No auxiliary data in native HDF5")
            return

        # Extract each auxiliary message type
        _extract_attitude(f_in, output_path, ping_times)
        _extract_environmental(f_in, output_path, ping_times)
        _extract_dvl(f_in, output_path, ping_times)


def _extract_attitude(
    f_in: h5py.File,
    output_path: Path,
    ping_times: np.ndarray,
) -> None:
    """Extract pitch/roll/heading data"""
    if "message_types/PitchRollDataMessage" not in f_in:  # type: ignore
        logger.debug("No PitchRollDataMessage found")
        return

    logger.debug("Extracting pitch/roll data...")
    pr_group = f_in["message_types/PitchRollDataMessage"]  # type: ignore

    # Read data
    fields = {}
    for field_name in ["pitch", "roll", "heading", "timestamp"]:
        if field_name in pr_group:  # type: ignore
            fields[field_name] = pr_group[field_name][:]  # type: ignore

    if not fields or "timestamp" not in fields:
        logger.debug("Insufficient attitude data to interpolate")
        return

    # Interpolate to ping times
    with h5py.File(output_path, "a") as f_out:
        if "auxiliary" not in f_out:  # type: ignore
            f_out.create_group("auxiliary")  # type: ignore

        aux = f_out["auxiliary"]  # type: ignore

        for field_name in ["pitch", "roll", "heading"]:
            if field_name in fields:
                try:
                    interp_data = _interpolate_to_ping_times(
                        fields["timestamp"],
                        fields[field_name],
                        ping_times,
                    )
                    aux.create_dataset(f"attitude/{field_name}", data=interp_data)  # type: ignore
                except Exception as e:
                    logger.warning(f"Could not interpolate {field_name}: {e}")


def _extract_environmental(
    f_in: h5py.File,
    output_path: Path,
    ping_times: np.ndarray,
) -> None:
    """Extract pressure/temperature/salinity data"""
    if "message_types/PressureSensorReadingMessage" not in f_in:  # type: ignore
        logger.debug("No PressureSensorReadingMessage found")
        return

    logger.debug("Extracting environmental data...")
    psr_group = f_in["message_types/PressureSensorReadingMessage"]  # type: ignore

    fields = {}
    for field_name in ["pressure", "temperature", "salinity", "sound_speed", "depth", "timestamp"]:
        if field_name in psr_group:  # type: ignore
            fields[field_name] = psr_group[field_name][:]  # type: ignore

    if not fields or "timestamp" not in fields:
        logger.debug("Insufficient environmental data")
        return

    with h5py.File(output_path, "a") as f_out:
        if "auxiliary" not in f_out:  # type: ignore
            f_out.create_group("auxiliary")  # type: ignore

        aux = f_out["auxiliary"]  # type: ignore

        for field_name in ["pressure", "temperature", "salinity", "sound_speed", "depth"]:
            if field_name in fields:
                try:
                    interp_data = _interpolate_to_ping_times(
                        fields["timestamp"],
                        fields[field_name],
                        ping_times,
                    )
                    aux.create_dataset(f"environmental/{field_name}", data=interp_data)  # type: ignore
                except Exception as e:
                    logger.warning(f"Could not interpolate {field_name}: {e}")


def _extract_dvl(
    f_in: h5py.File,
    output_path: Path,
    ping_times: np.ndarray,
) -> None:
    """Extract DVL velocity and altitude data"""
    if "message_types/DopplerVelocityLogDataMessage" not in f_in:  # type: ignore
        logger.debug("No DopplerVelocityLogDataMessage found")
        return

    logger.debug("Extracting DVL data...")
    dvl_group = f_in["message_types/DopplerVelocityLogDataMessage"]  # type: ignore

    fields = {}
    for field_name in [
        "bottom_dist", "bottom_vel_x", "bottom_vel_y", "bottom_vel_z",
        "water_vel_x", "water_vel_y", "water_vel_z",
        "pitch", "roll", "heading", "timestamp"
    ]:
        if field_name in dvl_group:  # type: ignore
            fields[field_name] = dvl_group[field_name][:]  # type: ignore

    if not fields or "timestamp" not in fields:
        logger.debug("Insufficient DVL data")
        return

    with h5py.File(output_path, "a") as f_out:
        if "auxiliary" not in f_out:  # type: ignore
            f_out.create_group("auxiliary")  # type: ignore

        aux = f_out["auxiliary"]  # type: ignore

        for field_name in ["bottom_dist", "bottom_vel_x", "bottom_vel_y", "bottom_vel_z",
                          "water_vel_x", "water_vel_y", "water_vel_z"]:
            if field_name in fields:
                try:
                    interp_data = _interpolate_to_ping_times(
                        fields["timestamp"],
                        fields[field_name],
                        ping_times,
                    )
                    aux.create_dataset(f"dvl/{field_name}", data=interp_data)  # type: ignore
                except Exception as e:
                    logger.warning(f"Could not interpolate {field_name}: {e}")


def _interpolate_to_ping_times(
    aux_times: np.ndarray,
    aux_values: np.ndarray,
    ping_times: np.ndarray,
) -> np.ndarray:
    """
    Interpolate auxiliary data to ping times.

    Parameters
    ----------
    aux_times : np.ndarray
        Timestamps of auxiliary measurements.
    aux_values : np.ndarray
        Values of auxiliary measurements.
    ping_times : np.ndarray
        Target ping times for interpolation.

    Returns
    -------
    np.ndarray
        Interpolated values at ping times.
    """
    # Remove NaN values
    valid = ~(np.isnan(aux_times) | np.isnan(aux_values))
    if np.sum(valid) < 2:
        logger.warning("Insufficient valid data for interpolation")
        return np.full_like(ping_times, np.nan, dtype=np.float64)

    aux_times_clean = aux_times[valid]
    aux_values_clean = aux_values[valid]

    # Sort by time
    sort_idx = np.argsort(aux_times_clean)
    aux_times_clean = aux_times_clean[sort_idx]
    aux_values_clean = aux_values_clean[sort_idx]

    # Interpolate
    try:
        f = interp1d(
            aux_times_clean,
            aux_values_clean,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
        )
        interpolated = f(ping_times)
        return interpolated
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}")
        return np.full_like(ping_times, np.nan, dtype=np.float64)