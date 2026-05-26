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
import json

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
    jsf_file_path: Path,
    output_path: Path,
    overwrite: bool = False,
    skip_private: bool = True,
) -> None:
    """
    Archive JSF messages to native HDF5 with full fidelity.

    Reads JSF binary file and stores all messages organized by type,
    preserving all vendor fields as-is. SonarDataMessages split by frequency.

    Parameters
    ----------
    jsf_file_path : Path
        Input JSF binary file path.
    output_path : Path
        Output .jsf.h5 HDF5 file path.
    overwrite : bool
        If False (default), raise if file exists. If True, overwrite.
    skip_private : bool
        If True (default), skip fields starting with '_' (internal implementation fields).

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.
    FileExistsError
        If output file exists and overwrite=False.

    Examples
    --------
    >>> jsf_to_native_hdf5("survey.jsf", "survey.jsf.h5")
    >>> jsf_to_native_hdf5("survey.jsf", "survey.jsf.h5", skip_private=False)  # Include all fields
    """
    jsf_file_path = Path(jsf_file_path)
    output_path = Path(output_path)

    if not jsf_file_path.exists():
        raise FileNotFoundError(f"JSF file not found: {jsf_file_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file exists: {output_path}. Use overwrite=True to replace."
        )

    logger.info(f"Reading JSF file: {jsf_file_path}")
    messages = read_jsf(jsf_file_path)

    logger.info(f"Read {len(messages)} messages")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Archiving to {output_path}")
    _archive_messages_to_hdf5(messages, output_path, skip_private=skip_private)

    logger.info(f"Successfully archived {jsf_file_path} to {output_path}")

def _archive_messages_to_hdf5(
    messages: list,
    output_path: Path,
    skip_private: bool = True,
) -> None:
    """
    Archive JSF messages to HDF5 organized by type and frequency.
    
    Each message is stored as a collection of datasets representing its fields.
    
    Parameters
    ----------
    messages : list
        List of JSF message objects.
    output_path : Path
        Output HDF5 file path.
    skip_private : bool
        If True (default), skip fields starting with '_'.
    """
    with h5py.File(output_path, "w") as f:
        # Organize messages by type
        messages_by_type = defaultdict(list)
        sonar_by_freq = defaultdict(list)

        for msg in messages:
            if isinstance(msg, SonarDataMessage):
                freq_hz = int(msg.transmit_freq_start)
                sonar_by_freq[freq_hz].append(msg)
            else:
                msg_type = type(msg).__name__
                messages_by_type[msg_type].append(msg)

        # Create root group
        root = f.create_group("message_types")

        # Store sonar messages by frequency
        if sonar_by_freq:
            sonar_group = root.create_group("SonarDataMessage")
            for freq_hz, msgs in sorted(sonar_by_freq.items()):
                freq_key = f"hz_{freq_hz}"
                freq_group = sonar_group.create_group(freq_key)
                _store_messages(freq_group, msgs, skip_private=skip_private)
                logger.debug(f"Stored {len(msgs)} SonarDataMessages at {freq_hz} Hz")

        # Store other messages by type
        for msg_type, msgs in messages_by_type.items():
            if msgs:
                type_group = root.create_group(msg_type)
                _store_messages(type_group, msgs, skip_private=skip_private)
                logger.debug(f"Stored {len(msgs)} {msg_type} messages")

def _store_messages(
    group: h5py.Group,
    messages: list,
    skip_private: bool = True,
) -> None:
    """
    Store a list of dataclass messages as indexed HDF5 datasets.

    Each field of each message becomes a dataset indexed by message number.
    
    Parameters
    ----------
    group : h5py.Group
        HDF5 group to store datasets in.
    messages : list
        List of message dataclass instances.
    skip_private : bool
        If True (default), skip fields starting with '_'.
    """
    if not messages:
        return

    msg_type = type(messages[0]).__name__
    n_messages = len(messages)

    logger.debug(f"Storing {n_messages} {msg_type} messages")

    # Get all field names from first message
    import dataclasses
    fields = {f.name for f in dataclasses.fields(messages[0])}

    # Filter out private fields if requested
    if skip_private:
        fields = {f for f in fields if not f.startswith("_")}

    # Separate fields by type for efficient storage
    for field_name in sorted(fields):
        field_values = [getattr(msg, field_name) for msg in messages]

        try:
            _store_field_as_dataset(group, field_name, field_values)
        except Exception as e:
            logger.warning(f"Could not store field {field_name}: {e}")


def _store_field_as_dataset(group: h5py.Group, field_name: str, values: list) -> None:
    """
    Store a message field as an HDF5 dataset.

    Handles various data types: scalars, arrays, strings, None values.
    """
    if not values:
        return

    first_val = values[0]

    # Handle None values
    if all(v is None for v in values):
        logger.debug(f"Skipping all-None field: {field_name}")
        return

    # Scalar numeric types
    if isinstance(first_val, (int, np.integer)):
        data = np.array(values, dtype=np.int64)
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, (float, np.floating)):
        data = np.array(values, dtype=np.float64)
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, bool):
        data = np.array(values, dtype=bool)
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, str):
        # String arrays need special handling
        data = np.array(values, dtype=h5py.string_dtype(encoding='utf-8'))
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, bytes):
        data = np.array(values, dtype=object)
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, np.ndarray):
        # Variable-length arrays - store as ragged
        try:
            data = np.stack(values)
            group.create_dataset(field_name, data=data)
        except ValueError:
            # Variable sizes - use object array
            data = np.array(values, dtype=object)
            group.create_dataset(field_name, data=data)

    elif isinstance(first_val, list):
        # Variable-length lists - store as JSON strings
        data = np.array([json.dumps(v) for v in values], dtype=h5py.string_dtype(encoding='utf-8'))
        group.create_dataset(field_name, data=data)

    elif isinstance(first_val, np.datetime64):
        # NumPy datetime64 - convert to int64 (nanoseconds since epoch)
        data = np.array(values, dtype='datetime64[ns]').astype('int64')
        group.create_dataset(field_name, data=data)
        group[field_name].attrs['dtype'] = 'datetime64[ns]'  # Mark for reconstruction

    elif isinstance(first_val, np.timedelta64):
        # NumPy timedelta64 - store as int64
        data = np.array(values, dtype='timedelta64[ns]').astype('int64')
        group.create_dataset(field_name, data=data)
        group[field_name].attrs['dtype'] = 'timedelta64[ns]'

    else:
        # Fallback: serialize as JSON strings
        logger.debug(f"Serializing {field_name} ({type(first_val)}) as JSON")
        data = np.array([json.dumps(v, default=str) for v in values], dtype=h5py.string_dtype(encoding='utf-8'))
        group.create_dataset(field_name, data=data)