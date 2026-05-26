"""
JSF file reading, parsing, and archiving to native HDF5 format.

This module provides utilities to:
1. Parse JSF binary files into message objects
2. Archive messages to native HDF5 format (preserving all fields)
3. Cache/pickle message objects for testing

The native HDF5 format is an intermediate format that stores all JSF messages
organized by type and frequency, with full fidelity to the original binary data.
For analysis, use the converter module to transform to standard schema.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
import json

import numpy as np
import h5py

from sidescanpy.io.jsf.messages import JSFMessageHeader, JSFMessageBody

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: JSF Binary File Reading
# =============================================================================

def read_jsf(fpath: Path, N: Optional[int] = None) -> list:
    """
    Parse all messages from a JSF binary file.

    Reads JSF binary format and deserializes messages sequentially until
    end-of-file or limit reached. Each message is validated and context
    is set from the message header.

    Parameters
    ----------
    fpath : Path
        Path to JSF binary file.
    N : int, optional
        Maximum number of messages to read. If None (default), read all.

    Returns
    -------
    list[JSFMessageBody]
        List of parsed message objects in order.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.
    ValueError
        If message deserialization fails (logged but continues).

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"))
    >>> print(f"Read {len(messages)} messages")
    
    >>> # Read first 1000 messages only
    >>> messages = read_jsf(Path("survey.jsf"), N=1000)
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")

    logger.info(f"Opening JSF file: {fpath}")
    all_messages = []
    cnt = 0

    with open(fpath, 'rb') as f:
        while (N is None or cnt < N):
            # Read message header
            header_bytes = f.read(JSFMessageHeader.STRUCT.size)
            if len(header_bytes) < JSFMessageHeader.STRUCT.size:
                logger.debug("End of file reached")
                break  # EOF or truncated message

            header = JSFMessageHeader.from_bytes(header_bytes)
            logger.debug(f"Message {len(all_messages)+1}: {header}")

            # Read message body
            message_bytes = f.read(header.message_size)
            try:
                message = JSFMessageBody.from_bytes(header.message_type, message_bytes)
                message.set_context(header)
                all_messages.append(message)
                logger.debug(f"Parsed: {type(message).__name__}")
            except ValueError as e:
                logger.warning(f"Failed to parse message: {e}")
                break

            cnt += 1

    logger.info(f"Successfully parsed {len(all_messages)} messages from {fpath}")
    return all_messages


def read_jsf_iter(fpath: Path):
    """
    Parse messages from JSF file as generator (memory efficient).

    Yields messages one at a time instead of loading entire file into memory.
    Useful for processing large JSF files.

    Parameters
    ----------
    fpath : Path
        Path to JSF binary file.

    Yields
    ------
    JSFMessageBody
        Next parsed message object.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.

    Examples
    --------
    >>> for message in read_jsf_iter(Path("survey.jsf")):
    ...     print(type(message).__name__)
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")

    logger.info(f"Opening JSF file (streaming): {fpath}")

    with open(fpath, 'rb') as f:
        while True:
            header_bytes = f.read(JSFMessageHeader.STRUCT.size)
            if len(header_bytes) < JSFMessageHeader.STRUCT.size:
                break

            header = JSFMessageHeader.from_bytes(header_bytes)
            message_bytes = f.read(header.message_size)

            try:
                message = JSFMessageBody.from_bytes(header.message_type, message_bytes)
                message.set_context(header)
                yield message
            except ValueError as e:
                logger.warning(f"Failed to parse message: {e}")
                continue


# =============================================================================
# SECTION 2: Native HDF5 Archiving (Phase 4a)
# =============================================================================

def jsf_to_native_hdf5(
    jsf_file_path: Path,
    output_path: Path,
    overwrite: bool = False,
    skip_private: bool = True,
) -> None:
    """
    Archive JSF messages to native HDF5 with full fidelity.

    Reads JSF binary file and stores all messages organized by type,
    preserving all vendor fields as-is. SonarDataMessages are split by
    frequency for efficient access.

    The native HDF5 format is an intermediate archive format. It maintains
    the complete structure of JSF messages without interpretation, making
    it suitable for long-term storage and later conversion to analysis formats.

    Parameters
    ----------
    jsf_file_path : Path
        Input JSF binary file path.
    output_path : Path
        Output .jsf.h5 HDF5 file path.
    overwrite : bool
        If False (default), raise if file exists. If True, overwrite.
    skip_private : bool
        If True (default), skip fields starting with '_' (internal implementation).
        If False, include all fields including private ones.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.
    FileExistsError
        If output file exists and overwrite=False.

    Examples
    --------
    >>> jsf_to_native_hdf5(
    ...     Path("survey.jsf"),
    ...     Path("survey.jsf.h5")
    ... )

    >>> # Include private fields
    >>> jsf_to_native_hdf5(
    ...     Path("survey.jsf"),
    ...     Path("survey.jsf.h5"),
    ...     skip_private=False
    ... )
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

    logger.info(f"Archiving to HDF5: {output_path}")
    _archive_messages_to_hdf5(messages, output_path, skip_private=skip_private)

    logger.info(f"Successfully archived {jsf_file_path} to {output_path}")


def _archive_messages_to_hdf5(
    messages: list,
    output_path: Path,
    skip_private: bool = True,
) -> None:
    """
    Archive JSF messages to HDF5 organized by type and frequency.

    Internal function that performs the actual HDF5 writing. Messages are
    organized hierarchically:
    - Root: /message_types
      - By type: /message_types/SonarDataMessage, /PitchRollDataMessage, etc.
      - For sonar: /message_types/SonarDataMessage/hz_FREQ for each frequency

    Each message field becomes a dataset indexed by message number.

    Parameters
    ----------
    messages : list
        List of JSF message objects.
    output_path : Path
        Output HDF5 file path.
    skip_private : bool
        If True, skip fields starting with '_'.
    """
    with h5py.File(output_path, "w") as f:
        # Organize messages by type
        messages_by_type = defaultdict(list)
        sonar_by_freq = defaultdict(list)

        for msg in messages:
            # Import here to avoid circular imports
            from sidescanpy.io.jsf.messages import SonarDataMessage

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
    This preserves the complete message structure in HDF5 format.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to store datasets in.
    messages : list
        List of message dataclass instances.
    skip_private : bool
        If True (default), skip fields starting with '_'.

    Notes
    -----
    - Fields are sorted alphabetically for consistent organization
    - Ragged arrays (variable length) stored as object dtype
    - Datetime fields marked with metadata for reconstruction
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

    # Store each field as a dataset
    for field_name in sorted(fields):
        field_values = [getattr(msg, field_name) for msg in messages]

        try:
            _store_field_as_dataset(group, field_name, field_values)
        except Exception as e:
            logger.warning(f"Could not store field {field_name}: {e}")

def _store_field_as_dataset(
    group: h5py.Group,
    field_name: str,
    values: list,
) -> None:
    """
    Store a field's values as an HDF5 dataset.

    Intelligently handles different data types:
    - Scalar types: Direct numpy array
    - Strings: UTF-8 encoded HDF5 string dtype
    - Datetime: Stored as int64 nanoseconds with dtype metadata
    - Ragged arrays: Object dtype HDF5 dataset
    - Lists: Converted to numpy arrays or stored as JSON

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to store in.
    field_name : str
        Name of field (becomes dataset name).
    values : list
        Values from all messages.
    """
    if not values:
        return

    first_val = values[0]

    # Handle strings
    if isinstance(first_val, str):
        str_array = np.array(values, dtype=object)
        ds = group.create_dataset(
            field_name,
            data=str_array,
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        return

    # Handle datetime
    if hasattr(first_val, '__class__') and 'datetime64' in str(type(first_val)):
        values_array = np.array(values, dtype='datetime64[ns]').astype(np.int64)
        ds = group.create_dataset(field_name, data=values_array)
        ds.attrs['dtype'] = 'datetime64[ns]'
        return

    # Handle numpy arrays (including ragged)
    if isinstance(first_val, np.ndarray):
        try:
            # Try to create regular array
            values_array = np.array(values)
            group.create_dataset(field_name, data=values_array)
        except (ValueError, TypeError):
            # Fall back to object array for ragged
            values_array = np.array(values, dtype=object)
            ds = group.create_dataset(field_name, data=values_array)
            logger.debug(f"Stored {field_name} as ragged object array")
        return

    # Handle lists (convert to JSON strings for HDF5 compatibility)
    if isinstance(first_val, list):
        json_values = [json.dumps(v) for v in values]
        ds = group.create_dataset(
            field_name,
            data=json_values,
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        ds.attrs['format'] = 'json'
        return

    # Handle scalars (int, float, bool, etc.)
    try:
        values_array = np.array(values)
        group.create_dataset(field_name, data=values_array)
    except Exception as e:
        logger.warning(f"Could not store {field_name} (type: {type(first_val)}): {e}")

# =============================================================================
# SECTION 3: Pickle Utilities (for testing)
# =============================================================================

def save_temp_file(
    data: list,
    filepath: Path = Path("temp_data.pkl"),
) -> None:
    """
    Save parsed messages to pickle file for testing.

    Allows caching parsed JSF messages to avoid re-parsing large files
    during repeated test runs. Useful for testing and development.

    Parameters
    ----------
    data : list
        List of message objects to save.
    filepath : Path
        Output pickle file path. Default: temp_data.pkl

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"))
    >>> save_temp_file(messages, Path("cached_messages.pkl"))
    """
    from pickle import dump

    with open(filepath, 'wb') as f:
        logger.debug(f"Saving {len(data)} messages to {filepath}")
        dump(data, f)

    logger.info(f"Saved {len(data)} messages to {filepath}")


def load_temp_file(filepath: Path = Path("temp_data.pkl")) -> list:
    """
    Load parsed messages from pickle file.

    Retrieves previously cached messages saved with save_temp_file().

    Parameters
    ----------
    filepath : Path
        Pickle file path to load. Default: temp_data.pkl

    Returns
    -------
    list
        List of message objects.

    Raises
    ------
    FileNotFoundError
        If pickle file does not exist.

    Examples
    --------
    >>> messages = load_temp_file(Path("cached_messages.pkl"))
    >>> print(f"Loaded {len(messages)} messages")
    """
    from pickle import load

    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        logger.debug(f"Loading messages from {filepath}")
        data = load(f)

    logger.info(f"Loaded {len(data)} messages from {filepath}")
    return data


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    jsf_file = Path('data/survey-a/jsf/20251015170335.jsf')
    head = 1000  # Limit for speed during tests

    # Parse file
    messages = read_jsf(jsf_file, N=head)
    logging.info(f"Parsed {len(messages)} total messages")

    # Cache to pickle
    fpath = Path("temp_jsf_data.pkl")
    save_temp_file(messages, filepath=fpath)
    logging.info(f"Saved to: {fpath}")

    # Load from pickle
    messages = load_temp_file(fpath)
    logging.info(f"Loaded from: {fpath}")