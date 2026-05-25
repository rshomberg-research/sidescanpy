"""
Convert JSF binary files to lossless native HDF5 format (.jsf.h5)

Stores raw JSF messages organized by type in HDF5 groups. No schema validation,
preserves all vendor fields. Primary purpose is format conversion for archival
and to provide a bridge to standard conversion.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

import xarray as xr
import numpy as np
import h5py

from sidescanpy.io.jsf.reader import read_jsf
from sidescanpy.io.jsf.messages import (
    JSFMessageBody,
    SonarDataMessage,
    PitchRollDataMessage,
    PressureSensorReadingMessage,
    DopplerVelocityLogDataMessage,
    NMEAStringMessage,
    FileTimeStampMessage,
    SystemInformationMessage,
    NavigationOffsetsMessage,    
)

logger = logging.getLogger(__name__)


def jsf_to_native_hdf5(
    jsf_file_path   : Path,
    output_path     : Path,
    overwrite       : bool = False,
) -> None:
    """
    Convert JSF binary file to native HDF5 format (.jsf.h5).

    Reads all messages from JSF file and stores them losslessly in HDF5,
    organized by message type. No schema validation or normalization.

    Parameters
    ----------
    jsf_file_path : Path
        Input JSF binary file path.
    output_path : Path
        Output .jsf.h5 HDF5 file path.
    overwrite : bool
        If False (default), raise if file exists. If True, overwrite.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.
    FileExistsError
        If output file exists and overwrite=False.

    Examples
    --------
    >>> jsf_to_native_hdf5("survey.jsf", "survey.jsf.h5")
    """
    
    if not jsf_file_path.exists():
        raise FileNotFoundError(f"JSF file not found: {jsf_file_path}")
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use overwrite=True to replace")
    
    logger.info(f"Reading JSF file: {jsf_file_path}")
    messages = read_jsf(jsf_file_path)

    logger.debug(f"Read {len(messages)} messages")
    
    logger.debug(f"Writing to {output_path}")
    _write_messages_to_hdf5(messages, output_path)
    
    logger.info(f"Successfully converted {jsf_file_path} to {output_path}")


def _write_messages_to_hdf5(messages: list[JSFMessageBody], output_path: Path) -> None:
    """
    Write JSF messages to HDF5 organized by type and frequency
    
    Structure:
    - frequencies/hz_XXXXXX/ (sonar data grouped by transmit frequency)
    - auxiliary/    (navigation, attitude, environmental, NMEA strings, etc.)
    - metadata/     (file-level, timestamps, system info)
    """
     
    # Organize messages by type
    sonar_msgs  = defaultdict(list)
    aux_msgs    = defaultdict(list)
    meta_msgs   = []

    for msg in messages:
        if isinstance(msg, SonarDataMessage):
            freq_hz = int(msg.transmit_freq_start)
            sonar_msgs[freq_hz].append(msg)
        elif isinstance(
            msg, (
                PitchRollDataMessage,
                PressureSensorReadingMessage,
                DopplerVelocityLogDataMessage,
                NMEAStringMessage,
            )
        ):
            msg_type = type(msg).__name__
            aux_msgs[msg_type].append(msg)
        elif isinstance(
            msg, (
                FileTimeStampMessage,
                SystemInformationMessage,
                NavigationOffsetsMessage,
            )
        ):
            meta_msgs.append(msg)
        else:
            logger.warning(f"Skipping unhandled message type: {type(msg).__name__}")

    with h5py.File(output_path, "w") as f:

        # Write sonar data
        if sonar_msgs:
            freq_group = f.create_group("frequencies")
            for freq_hz, msgs in sorted(sonar_msgs.items()):    # ???: how are we sorting these?
                _write_sonar_frequency(freq_group, freq_hz, msgs)

        # Write auxiliary data
        if aux_msgs:
            aux_group = f.create_group("auxiliary")
            for msg_type, msgs in aux_msgs.items():
                _write_auxiliary_messages(aux_group, msg_type, msgs)

        # Write metadata
        if meta_msgs:
            meta_group = f.create_group("metadata")
            _write_metadata_messages(meta_group, meta_msgs)


def _write_sonar_frequency(
        freq_group: h5py.Group,
        freq_hz: int,
        messages: list[SonarDataMessage]
        ) -> None:
    """Write sonar messages for a single frequency to HDF5"""
    freq_key = f"hz_{freq_hz}"
    ds_group = freq_group.create_group(freq_key)

    logger.debug(f"Writing {len(messages)} sonar messages to {freq_key} group")

    # Organize by ping to handle port/starboard 
    ping_dict = defaultdict(lambda: {0: None, 1: None})
    for msg in messages:
        ping_dict[msg.ping_number][msg.channel] = msg

    # Determine dimensions
    n_pings     = len(ping_dict)
    n_channels  = 2 # ???: is this always true?
    n_samples   = max(msg.N_samples for msg in messages)    # ???: should be same for all messages

    # Create datasets for backscatter data
    backscatter_shape = (n_pings, n_channels, n_samples)
    backscatter_ds = ds_group.create_dataset(
        "backscatter",
        shape=backscatter_shape,
        dtype=np.float32,
        fillvalue=np.nan,
    )

    # Create datasets for navigation data
    nav_datasets = {
        "timestamp": ds_group.create_dataset("timestamp", shape=(n_pings, n_channels), dtype='float64'),
        "latitude": ds_group.create_dataset("latitude", shape=(n_pings, n_channels), dtype='float32', fillvalue=np.nan),
        "longitude": ds_group.create_dataset("longitude", shape=(n_pings, n_channels), dtype='float32', fillvalue=np.nan),
        "heading_deg": ds_group.create_dataset("heading_deg", shape=(n_pings, n_channels), dtype='float32', fillvalue=np.nan),
        "depth_m": ds_group.create_dataset("depth_m", shape=(n_pings, n_channels), dtype='float32', fillvalue=np.nan),
        "ping_number": ds_group.create_dataset("ping_number", shape=(n_pings, n_channels), dtype='int32'),
    }
    
    # Fill datasets
    for ping_idx, (ping_num, channels) in enumerate(sorted(ping_dict.items())):
        # ???: how is this sorted?
        for ch_num in [0, 1]:
            msg = channels[ch_num]
            if msg is not None:
                n_actual = len(msg.samples) # should always be the same
                backscatter_ds[ping_idx, ch_num, :n_actual] = msg.samples

                # Navigation data
                nav_datasets["timestamp"][ping_idx, ch_num] = msg.timestamp
                nav_datasets["latitude"][ping_idx, ch_num] = msg.latitude
                nav_datasets["longitude"][ping_idx, ch_num] = msg.longitude
                nav_datasets["heading_deg"][ping_idx, ch_num] = msg.heading
                nav_datasets["depth_m"][ping_idx, ch_num] = msg.depth
                nav_datasets["ping_number"][ping_idx, ch_num] = msg.ping_number

    # Store metadata as attributes
    first_msg = messages[0]
    ds_group.attrs["freq_hz_start"] = first_msg.transmit_freq_start
    ds_group.attrs["freq_hz_end"] = first_msg.transmit_freq_end
    ds_group.attrs["sample_rate_hz"] = first_msg.sample_freq
    ds_group.attrs["pulse_length_us"] = first_msg.sweep_length * 1e6
    ds_group.attrs["n_pings"] = n_pings
    ds_group.attrs["n_samples"] = n_samples

    logger.debug(f"Wrote frequency {freq_key}: {n_pings} pings, {n_samples} sampels")

def _write_auxiliary_messages(
        aux_group: h5py.Group,
        msg_type: str,
        messages: list[JSFMessageBody]
) -> None:
    """Write auxiliary sensor messages to HDF5."""
    msg_group = aux_group.create_group(msg_type)
    
    logger.debug(f"Writing {len(messages)} {msg_type} messages")
    
    # For now, store as simple string representation
    # This preserves the data for debugging/archival
    msg_group.attrs["count"] = len(messages)
    msg_group.attrs["message_type"] = msg_type

    # raise NotImplementedError
    

def _write_metadata_messages(
        meta_group: h5py.Group,
        messages: list[JSFMessageBody]
) -> None:
    """Write metadata messages to HDF5."""
    logger.debug(f"Writing {len(messages)} metadata messages")
    
    meta_group.attrs["count"] = len(messages)
    # raise NotImplementedError
