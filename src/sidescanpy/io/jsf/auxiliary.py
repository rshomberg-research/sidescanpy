"""Extract and interpolate auxiliary JSF message data.

Auxiliary messages (pressure, attitude, navigation, etc.) are less frequent
than sonar pings but contain more accurate environmental and position data.
This module provides tools to extract auxiliary data and interpolate it into
sonar ping timestamps for data enrichment.

2026 rshomberg@uri.edu
"""

# TODO: fix this file. Currently this is just some old broken code

import logging
import numpy as np
from typing import List, Dict
from scipy.interpolate import interp1d


def extract_auxiliary_data(
    messages: List,
    message_type: str,
    fields: List[str]
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract auxiliary message data by type and fields.
    
    Args:
        messages: List of all JSF messages
        message_type: Name of auxiliary message class (e.g., 'PressureSensorReadingMessage')
        fields: List of field names to extract (e.g., ['depth', 'temperature', 'sound_speed'])
    
    Returns:
        (timestamps, {field: values}) sorted by timestamp
    """
    timestamps = []
    data = {field: [] for field in fields}
    
    for msg in messages:
        if type(msg).__name__ != message_type:
            continue
        
        timestamps.append(msg.timestamp)
        for field in fields:
            data[field].append(getattr(msg, field))
    
    if not timestamps:
        logging.debug(f"No {message_type} found")
        return np.array([]), {f: np.array([]) for f in fields}
    
    # Convert to numpy arrays and sort by time
    timestamps = np.array(timestamps)
    for field in fields:
        data[field] = np.array(data[field])
    
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    for field in fields:
        data[field] = data[field][sort_idx]
    
    logging.debug(f"Extracted {len(timestamps)} {message_type} readings")
    return timestamps, data


def interpolate_auxiliary(
    sonar_times: np.ndarray,
    aux_times: np.ndarray,
    aux_values: np.ndarray,
    kind: str = 'linear',
    fill_value: str = 'extrapolate'
) -> np.ndarray:
    """
    Interpolate auxiliary data to sonar ping times.
    
    Args:
        sonar_times: Timestamps of sonar pings (1D array)
        aux_times: Timestamps of auxiliary measurements (sorted)
        aux_values: Values of auxiliary measurements
        kind: Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.)
              See scipy.interpolate.interp1d for full list
        fill_value: How to handle extrapolation ('extrapolate', 'nan', or numeric value)
    
    Returns:
        Array of interpolated values matching sonar_times
    """
    if len(aux_times) == 0:
        return np.full_like(sonar_times, np.nan)
    
    if len(aux_times) == 1:
        return np.full_like(sonar_times, aux_values[0])
    
    try:
        f = interp1d(
            aux_times,
            aux_values,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value
        )
        return f(sonar_times)
    
    except Exception as e:
        logging.warning(f"Interpolation failed ({kind}): {e}, returning NaN")
        return np.full_like(sonar_times, np.nan)

if __name__=='__main__':

    logging.basicConfig(
        level=logging.DEBUG
    )

    logging.info("Testing io/jsf/auxiliary.py")

    # Load pickled messages
    import pickle

    pkl_file = Path("temp_jsf_data.pkl")
    
    if not pkl_file.exists():
        logging.error(f"File not found: {pkl_file}")
        exit(1)
    
    logging.info(f"Loading messages from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        all_messages = pickle.load(f)
    
    logging.info(f"Loaded {len(all_messages)} total messages\n")
    
    # Group messages by type
    # TODO: use defaultdict for cleaner
    grouped_by_type = {}
    for msg in all_messages:
        cls_name = type(msg).__name__ 
        if cls_name not in grouped_by_type:
            grouped_by_type[cls_name] = []
        grouped_by_type[cls_name].append(msg)

    for msg_type in sorted(grouped_by_type.keys()):
        count = len(grouped_by_type[msg_type])
        logging.info(f"  {msg_type}: {count}")

    # Inspect the different types  
    # The goal now is to pull relevant data from each of the message types. This
    # will require some hardcoding, and I will want to manually compare the data
    # to see what is most precise. Lets see what relevant variables there are,
    # see what overlaps, and compare that to the data in the SonarDataMessages

    for msg_type in sorted(grouped_by_type.keys()):
        print(grouped_by_type[msg_type][-1])
        print()

    
    # Inspect: NMEARMCMessage
    # timestamp
    # source
    # latitude
    # longitude
    # speed_knots 
    # course_deg 
    
    # msgs = grouped_by_type['NMEARMCMessage']
    # print(msgs[0])

    # Inspect: NMEAStringMessage:
    # timestamp
    # source

    # Inspect: NavigationOffsetsMessage
    # Fixed values (one per file?)
    # TODO: all zero in my cases, but may be relevant later (ignore for now)

    # Inspect: PitchRollDataMessage
    # timestamp
    # pitch
    # roll
    # heading
    # Several empty fields as well

    # Inspect: PressureSensorReadingMessage
    # timestamp
    # pressure
    # temperature
    # salinity
    # conductivity
    # soundspeed
    # depth
    # ???: many are zero which is definitely not correct

    # Inspect: SonarMessageStatusMessage
    # Not useful

    # Inspect: SystemInformationMessage
    # system_type
    # sonar_software_version
    # num_subsystems = 2
    # num_serial_devices = 2
    # tow_vehicle_serial = 66836

    # Inspect: UnknownMessage2040:
    # ???

    # Compare to: SonarDataMessage
    # timestamp (simultaneous for channel and freq (x4 messages))
    # ping_number (different counter per freq)
    # channel
    # sound_speed = None
    # course
    # speed
    # latitude
    # longitude
    # pressure_pse
    # depth
    # altitude
    # heading: different from course?
    # pitch
    # roll