"""
JSF file reading, parsing, and inspection utilities.

This module provides:
1. Parse JSF binary files into message objects (streaming or bulk)
2. Inspect and debug JSF messages (slice, filter, export)
3. Export messages to JSON or other formats for inspection

The module focuses on raw message access without intermediate formats.
For analysis/conversion, use the converter module to transform to standard schema.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any
import json
from dataclasses import asdict, fields as dataclass_fields
from collections import defaultdict

import numpy as np

from sidescanpy.io.jsf.messages import (
    JSFMessageHeader,
    JSFMessage,
    SonarDataMessage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: Core JSF Reading
# =============================================================================

def read_jsf(fpath: Path, N: Optional[int] = None) -> list:
    """
    Parse all messages from a JSF binary file.

    Reads JSF binary format and deserializes messages sequentially until
    end-of-file or limit reached. Each message is validated and deserialized.

    Parameters
    ----------
    fpath : Path
        Path to JSF binary file.
    N : int, optional
        Maximum number of messages to read. If None (default), read all.

    Returns
    -------
    list[JSFMessage]
        List of parsed message objects in order.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.

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
                break

            try:
                header = JSFMessageHeader.from_bytes(header_bytes)
            except Exception as e:
                logger.warning(f"Failed to parse header at offset {f.tell()}: {e}")
                break

            # Read message body
            message_bytes = f.read(header.message_size)
            if len(message_bytes) < header.message_size:
                logger.warning(f"Truncated message at record {cnt}")
                break

            try:
                if header.message_type == SonarDataMessage.MESSAGE_TYPE:
                    message = SonarDataMessage.from_bytes(header, message_bytes)
                else:
                    message = JSFMessage.from_bytes(header, message_bytes)
                
                all_messages.append(message)
                logger.debug(f"Message {cnt}: {type(message).__name__}")
            except Exception as e:
                logger.warning(f"Failed to parse message at record {cnt}: {e}")

            cnt += 1

    logger.info(f"Successfully parsed {len(all_messages)} messages from {fpath}")
    return all_messages


def read_jsf_iter(fpath: Path) -> Generator[JSFMessage, None, None]:
    """
    Parse messages from JSF file as generator (memory efficient).

    Yields messages one at a time instead of loading entire file into memory.
    Useful for processing large JSF files or streaming to HDF5.

    Parameters
    ----------
    fpath : Path
        Path to JSF binary file.

    Yields
    ------
    JSFMessage
        Next parsed message object.

    Raises
    ------
    FileNotFoundError
        If JSF file does not exist.

    Examples
    --------
    >>> for message in read_jsf_iter(Path("survey.jsf")):
    ...     print(type(message).__name__)
    ...     if isinstance(message, SonarDataMessage):
    ...         process_sonar(message)
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")

    logger.info(f"Opening JSF file (streaming): {fpath}")
    record_count = 0

    with open(fpath, 'rb') as f:
        while True:
            header_bytes = f.read(JSFMessageHeader.STRUCT.size)
            if len(header_bytes) < JSFMessageHeader.STRUCT.size:
                break

            try:
                header = JSFMessageHeader.from_bytes(header_bytes)
            except Exception as e:
                logger.warning(f"Failed to parse header at record {record_count}: {e}")
                break

            message_bytes = f.read(header.message_size)
            if len(message_bytes) < header.message_size:
                logger.warning(f"Truncated message at record {record_count}")
                break

            try:
                message = JSFMessage.from_bytes(header, message_bytes)
                record_count += 1
                yield message
            except Exception as e:
                logger.warning(f"Failed to parse message at record {record_count}: {e}")
                continue


# =============================================================================
# SECTION 2: Message Inspection & Slicing
# =============================================================================

def get_messages_by_index(
    fpath: Path,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> List[JSFMessage]:
    """
    Retrieve JSF messages by index range (slicing).

    Efficiently reads a specific range of messages from JSF file without
    loading entire file. Useful for debugging or inspecting specific regions.

    Parameters
    ----------
    fpath : Path
        Path to JSF binary file.
    start_idx : int
        Starting message index (0-indexed). Default: 0
    end_idx : int, optional
        Ending message index (exclusive). If None, read to end of file.

    Returns
    -------
    list[JSFMessage]
        Messages in the specified range.

    Examples
    --------
    >>> # Get messages 100-150
    >>> messages = get_messages_by_index(Path("survey.jsf"), 100, 150)
    
    >>> # Get first 50 messages
    >>> messages = get_messages_by_index(Path("survey.jsf"), 0, 50)
    """
    messages = []
    
    for idx, message in enumerate(read_jsf_iter(fpath)):
        if idx < start_idx:
            continue
        if end_idx is not None and idx >= end_idx:
            break
        messages.append(message)
    
    logger.info(f"Retrieved messages [{start_idx}:{end_idx}] = {len(messages)} messages")
    return messages


def filter_messages_by_type(
    messages: List[JSFMessage],
    message_type: str,
) -> List[JSFMessage]:
    """
    Filter messages by type name.

    Parameters
    ----------
    messages : list[JSFMessage]
        Input messages.
    message_type : str
        Message type name (e.g., 'SonarDataMessage', 'PitchRollDataMessage').

    Returns
    -------
    list[JSFMessage]
        Filtered messages matching the type.

    Examples
    --------
    >>> sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
    >>> pitch_msgs = filter_messages_by_type(messages, 'PitchRollDataMessage')
    """
    filtered = [m for m in messages if type(m).__name__ == message_type]
    logger.debug(f"Filtered {len(filtered)} {message_type} messages from {len(messages)} total")
    return filtered


def filter_messages_by_frequency(
    messages: List[SonarDataMessage],
    freq_hz: int,
) -> List[SonarDataMessage]:
    """
    Filter SonarDataMessages by transmit frequency.

    Parameters
    ----------
    messages : list[SonarDataMessage]
        Input sonar messages.
    freq_hz : int
        Frequency in Hz to filter by.

    Returns
    -------
    list[SonarDataMessage]
        Messages matching the frequency.

    Examples
    --------
    >>> sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
    >>> freq_1495 = filter_messages_by_frequency(sonar_msgs, 1495000)
    """
    filtered = [
        m for m in messages
        if isinstance(m, SonarDataMessage)
        and int(m.transmit_freq_start) == freq_hz
    ]
    logger.debug(f"Filtered {len(filtered)} messages at {freq_hz} Hz")
    return filtered


def get_message_statistics(messages: List[JSFMessage]) -> Dict[str, Any]:
    """
    Get statistics about a collection of messages.

    Parameters
    ----------
    messages : list[JSFMessage]
        Input messages.

    Returns
    -------
    dict
        Statistics including type breakdown, frequency distribution, etc.

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"))
    >>> stats = get_message_statistics(messages)
    >>> print(f"Total messages: {stats['total']}")
    >>> print(f"By type: {stats['by_type']}")
    """
    stats = {
        'total': len(messages),
        'by_type': defaultdict(int),
        'frequencies': defaultdict(int),
    }
    
    for msg in messages:
        msg_type = type(msg).__name__
        stats['by_type'][msg_type] += 1
        
        if isinstance(msg, SonarDataMessage):
            freq = int(msg.transmit_freq_start)
            stats['frequencies'][freq] += 1
    
    return dict(stats)


# =============================================================================
# SECTION 3: JSON Export (for inspection/debugging)
# =============================================================================

def _serialize_message(message: JSFMessage) -> Dict[str, Any]:
    """
    Serialize a single JSFMessage to JSON-compatible dict.

    Handles special types (numpy, datetime, etc.) that aren't JSON serializable.

    Parameters
    ----------
    message : JSFMessage
        Message to serialize.

    Returns
    -------
    dict
        JSON-compatible dictionary.
    """
    msg_dict = {}
    
    for field in dataclass_fields(message):
        field_name = field.name
        field_value = getattr(message, field_name)
        
        # Skip private fields
        if field_name.startswith('_'):
            continue
        
        # Handle special types
        if field_value is None:
            msg_dict[field_name] = None
        elif isinstance(field_value, np.ndarray):
            # Convert numpy arrays to lists, but truncate large ones
            if field_value.size > 1000:
                msg_dict[field_name] = {
                    "_type": "numpy.ndarray",
                    "shape": list(field_value.shape),
                    "dtype": str(field_value.dtype),
                    "size": field_value.size,
                    "first_10": field_value.flat[:10].tolist(),
                    "last_10": field_value.flat[-10:].tolist(),
                }
            else:
                msg_dict[field_name] = field_value.tolist()
        elif isinstance(field_value, (np.integer, np.floating)):
            msg_dict[field_name] = field_value.item()
        elif isinstance(field_value, np.datetime64):
            msg_dict[field_name] = str(field_value)
        elif isinstance(field_value, (list, tuple)):
            msg_dict[field_name] = [
                x.item() if isinstance(x, (np.integer, np.floating)) else x
                for x in field_value
            ]
        else:
            msg_dict[field_name] = field_value
    
    msg_dict['_message_type'] = type(message).__name__
    return msg_dict


def export_messages_to_json(
    messages: List[JSFMessage],
    output_path: Path,
    pretty: bool = True,
    max_messages: Optional[int] = None,
) -> None:
    """
    Export messages to JSON for inspection/debugging.

    Creates a JSON file with all messages. Large numpy arrays are summarized
    to keep file size reasonable. Useful for inspecting message structure
    and debugging conversion issues.

    Parameters
    ----------
    messages : list[JSFMessage]
        Messages to export.
    output_path : Path
        Output JSON file path.
    pretty : bool
        If True (default), use indentation for readability.
    max_messages : int, optional
        Maximum messages to export. If None, export all.

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"), N=100)
    >>> export_messages_to_json(messages, Path("debug.json"))
    
    >>> # Export only sonar messages
    >>> sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
    >>> export_messages_to_json(sonar_msgs, Path("sonar_debug.json"), max_messages=50)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if max_messages is not None:
        messages = messages[:max_messages]
    
    # Serialize all messages
    serialized = []
    for i, msg in enumerate(messages):
        try:
            msg_dict = _serialize_message(msg)
            msg_dict['_index'] = i
            serialized.append(msg_dict)
        except Exception as e:
            logger.warning(f"Failed to serialize message {i}: {e}")
    
    # Write to JSON
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(serialized, f, indent=2, default=str)
        else:
            json.dump(serialized, f, default=str)
    
    logger.info(f"Exported {len(serialized)} messages to {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_messages_by_type_to_json(
    messages: List[JSFMessage],
    output_dir: Path,
) -> None:
    """
    Export messages to separate JSON files organized by type.

    Creates one JSON file per message type for easier inspection.

    Parameters
    ----------
    messages : list[JSFMessage]
        Messages to export.
    output_dir : Path
        Output directory for JSON files.

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"), N=1000)
    >>> export_messages_by_type_to_json(messages, Path("debug_output"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by type
    by_type = defaultdict(list)
    for msg in messages:
        msg_type = type(msg).__name__
        by_type[msg_type].append(msg)
    
    # Write each type to its own file
    for msg_type, msgs in sorted(by_type.items()):
        output_file = output_dir / f"{msg_type}.json"
        
        serialized = []
        for i, msg in enumerate(msgs):
            try:
                msg_dict = _serialize_message(msg)
                msg_dict['_index'] = i
                serialized.append(msg_dict)
            except Exception as e:
                logger.warning(f"Failed to serialize {msg_type} message {i}: {e}")
        
        with open(output_file, 'w') as f:
            json.dump(serialized, f, indent=2, default=str)
        
        logger.info(f"  {msg_type}: {len(serialized)} messages → {output_file}")


# =============================================================================
# SECTION 4: Pickle Utilities (for testing)
# =============================================================================

def save_messages_to_pickle(
    messages: List[JSFMessage],
    filepath: Path,
) -> None:
    """
    Save parsed messages to pickle file for caching during testing.

    Allows caching parsed JSF messages to avoid re-parsing large files
    during repeated test runs.

    Parameters
    ----------
    messages : list[JSFMessage]
        Messages to save.
    filepath : Path
        Output pickle file path.

    Examples
    --------
    >>> messages = read_jsf(Path("survey.jsf"))
    >>> save_messages_to_pickle(messages, Path("cached_messages.pkl"))
    """
    from pickle import dump

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        dump(messages, f)

    logger.info(f"Saved {len(messages)} messages to {filepath}")


def load_messages_from_pickle(filepath: Path) -> List[JSFMessage]:
    """
    Load parsed messages from pickle file.

    Retrieves previously cached messages saved with save_messages_to_pickle().

    Parameters
    ----------
    filepath : Path
        Pickle file path to load.

    Returns
    -------
    list[JSFMessage]
        List of message objects.

    Raises
    ------
    FileNotFoundError
        If pickle file does not exist.

    Examples
    --------
    >>> messages = load_messages_from_pickle(Path("cached_messages.pkl"))
    """
    from pickle import load

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        messages = load(f)

    logger.info(f"Loaded {len(messages)} messages from {filepath}")
    return messages


# =============================================================================
# SECTION 5: Example Usage
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    jsf_file = Path('data/survey-a/jsf/20251015170335.jsf')
    
    # Example 1: Read first 1000 messages
    print("\n1️⃣  Reading first 1000 messages...")
    messages = read_jsf(jsf_file, N=1000)
    stats = get_message_statistics(messages)
    print(f"   Total: {stats['total']}")
    print(f"   By type: {dict(stats['by_type'])}")
    print(f"   Frequencies: {dict(stats['frequencies'])}")
    
    # Example 2: Export to JSON for inspection
    print("\n2️⃣  Exporting to JSON...")
    export_messages_to_json(messages, Path("debug_messages.json"), max_messages=100)
    export_messages_by_type_to_json(messages, Path("debug_by_type"))
    
    # Example 3: Filter and inspect specific types
    print("\n3️⃣  Filtering messages...")
    sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
    print(f"   Found {len(sonar_msgs)} sonar messages")
    
    if sonar_msgs:
        freq_1495 = filter_messages_by_frequency(sonar_msgs, 1495000)
        print(f"   {len(freq_1495)} messages at 1495000 Hz")
    
    # Example 4: Cache to pickle
    print("\n4️⃣  Caching to pickle...")
    save_messages_to_pickle(messages, Path("temp_jsf_messages.pkl"))
    loaded = load_messages_from_pickle(Path("temp_jsf_messages.pkl"))
    print(f"   Loaded {len(loaded)} messages from cache")