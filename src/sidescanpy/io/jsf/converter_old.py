"""Convert JSF SonarDataMessage objects to xarray Datasets.

JSF files consist of SonarDataMessages and auxiliary messages. The
SonarDataMessages can be filtered by frequency channel, generally high-freq and
low-freq. Additionally, there is a channel, generally 0=port 1=starboard. The
frequency channels must be separated because the samples per-ping are not the
same for different frequencies preventing frequency channel from being used as a
xarray dimension.

2026 rshomberg@uri.edu
"""

import logging
import numpy as np
import xarray as xr
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

from sidescanpy.core.survey import MultiFrequencySideScanSurvey


def to_survey(messages: List, source_file: str) -> MultiFrequencySideScanSurvey:
    """
    Convert list of JSF SonarDataMessage objects to MultiFrequencySideScanSurvey.
    
    Args:
        messages: List of SonarDataMessage objects
        source_file: Path to JSF file
    
    Returns:
        MultiFrequencySideScanSurvey with all frequencies
    """
    logging.info(f"Converting {len(messages)} messages to survey")
    
    sonar_messages = [m for m in messages if type(m).__name__ == 'SonarDataMessage']
    logging.info(f"Found {len(sonar_messages)} sonar messages")
    
    # Group messages by frequency
    freq_messages = _group_by_frequency(sonar_messages)
    logging.info(f"Found {len(freq_messages)} frequencies")
    
    # Create survey
    survey = MultiFrequencySideScanSurvey(
        source_file=source_file,
        source_format='JSF',
        survey_name=Path(source_file).stem
    )
    
    # Convert each frequency
    for freq_hz, freq_msgs in freq_messages.items():
        freq_key = _frequency_to_key(freq_hz)
        logging.info(f"Converting {freq_key}: {len(freq_msgs)} messages")
        
        ds = _messages_to_dataset(freq_msgs, freq_hz)
        survey.add_frequency(freq_key, ds)
    
    return survey


def _group_by_frequency(messages: List) -> Dict[float, List]:
    """Group sonar messages by transmit frequency."""
    freq_messages = defaultdict(list)
    
    for msg in messages:
        freq_hz = msg.transmit_freq_start
        freq_messages[freq_hz].append(msg)
    
    return dict(sorted(freq_messages.items(), reverse=True))


def _frequency_to_key(freq_hz: float) -> str:
    """Convert frequency Hz to standardized key string."""
    return f"freq_{int(freq_hz)}_hz"


def _messages_to_dataset(messages: List, freq_hz: float) -> xr.Dataset:
    """
    Convert sonar messages for single frequency to xarray.Dataset.
    
    Organizes messages by ping and channel into 3D array:
    dims: [ping, channel, sample]
    
    Args:
        messages: Sonar messages for this frequency
        freq_hz: Frequency in Hz
    
    Returns:
        xarray.Dataset with sonar data
    """
    # Group by ping
    ping_groups = defaultdict(lambda: {0: None, 1: None})
    
    for msg in messages:
        ping_groups[msg.ping_number][msg.channel] = msg
    
    # Extract dimensions
    n_pings = len(ping_groups)
    n_channels = 2
    
    # Determine max samples
    sample_counts = set()
    for channels in ping_groups.values():
        for msg in channels.values():
            if msg is not None:
                sample_counts.add(msg.N_samples)
    
    if len(sample_counts) > 1:
        logging.warning(f"Variable sample counts: {sample_counts}")
    
    n_samples = max(sample_counts) if sample_counts else 0
    
    logging.debug(f"Dataset: {n_pings} pings × {n_channels} channels × {n_samples} samples")
    
    # Preallocate arrays
    samples_array = np.full((n_pings, n_channels, n_samples), np.nan)
    ping_numbers = np.zeros((n_pings, n_channels), dtype=int)
    timestamps = np.zeros((n_pings, n_channels), dtype='datetime64[ns]')
    latitudes = np.full((n_pings, n_channels), np.nan)
    longitudes = np.full((n_pings, n_channels), np.nan)
    headings = np.full((n_pings, n_channels), np.nan)
    depths = np.full((n_pings, n_channels), np.nan)
    
    # Fill arrays
    for ping_idx, (ping_num, channels) in enumerate(sorted(ping_groups.items())):
        for ch_num in [0, 1]:
            msg = channels[ch_num]
            
            if msg is not None:
                n_actual = len(msg.samples)
                samples_array[ping_idx, ch_num, :n_actual] = msg.samples
                ping_numbers[ping_idx, ch_num] = msg.ping_number
                timestamps[ping_idx, ch_num] = msg.timestamp
                latitudes[ping_idx, ch_num] = msg.latitude or np.nan
                longitudes[ping_idx, ch_num] = msg.longitude or np.nan
                headings[ping_idx, ch_num] = msg.heading or np.nan
                depths[ping_idx, ch_num] = msg.depth or np.nan
    
    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'samples': (['ping', 'channel', 'sample'], samples_array),
            'ping_number': (['ping', 'channel'], ping_numbers),
            'timestamp': (['ping', 'channel'], timestamps),
            'latitude': (['ping', 'channel'], latitudes),
            'longitude': (['ping', 'channel'], longitudes),
            'heading': (['ping', 'channel'], headings),
            'depth': (['ping', 'channel'], depths),
        },
        coords={
            'ping': np.arange(n_pings),
            'channel': np.array([0, 1]),
            'sample': np.arange(n_samples),
        },
        attrs={
            'transmit_freq_start': int(freq_hz),
            'transmit_freq_end': int(freq_hz),
        }
    )
    
    return ds


if __name__ == '__main__':
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(message)s'
    )
    
    logging.info("Testing io/jsf/converter.py\n")
    
    # Load pickled messages
    import pickle
    pkl_file = Path("./tests/data/temp_jsf_data.pkl")
    
    if not pkl_file.exists():
        logging.error(f"File not found: {pkl_file}")
        exit(1)
    
    logging.info(f"Loading messages from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        all_messages = pickle.load(f)
    
    logging.info(f"Loaded {len(all_messages)} messages\n")
    
    # Test: Convert to survey
    logging.info("=" * 70)
    logging.info("Test: Convert SonarDataMessages to MultiFrequencySurvey")
    logging.info("=" * 70 + "\n")
    
    survey = to_survey(all_messages, source_file=str(pkl_file))
    
    logging.info(f"Created survey with {survey.n_frequencies} frequency(ies)\n")
    
    for freq_key in survey.frequencies:
        freq_survey = survey.get_frequency(freq_key)
        ds = freq_survey.dataset
        
        logging.info(f"{freq_key}:")
        logging.info(f"  Dimensions: {dict(ds.sizes)}")
        logging.info(f"  Data variables: {ds.data_vars}")
        logging.info(f"  Attributes: {dict(ds.attrs)}")
        
        # Validate data integrity
        n_pings, n_channels = ds['samples'].shape[:2]
        logging.info(f"  Data shape: {n_pings} pings × {n_channels} channels")
        
        # Check for missing values
        n_valid = np.sum(~np.isnan(ds['latitude'].values))
        logging.info(f"  Valid lat/lon: {n_valid}/{n_pings * n_channels}")
    
    logging.info("=" * 70)
    logging.info("Test completed successfully!")
    logging.info("=" * 70)