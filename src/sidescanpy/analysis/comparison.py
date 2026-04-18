"""Simple data comparison plots"""

# TODO: implement logging

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import matplotlib.dates as mdates

from sidescanpy.io.hdf5_manager import HDF5Loader

def _convert_timestamps(timestamps):
    """Convert timestamps to datetime objects.
    
    Handles both seconds (1e9 scale) and nanoseconds (1e18 scale)
    """
    # Detect scale: if max > 1e15, it's probably nanoseconds
    ts_array = np.array(timestamps)
    max_ts = np.max(ts_array)
    
    if max_ts > 1e15:
        # Nanoseconds - convert to seconds
        timestamps_sec = ts_array / 1e9
    else:
        # Seconds - use as-is
        timestamps_sec = ts_array
    
    # Convert to datetime
    return [datetime.utcfromtimestamp(ts) for ts in timestamps_sec]

def plot_sonar_vs_metadata(hdf5_file: str, metadata_table: str, 
                          field_name: str, sonar_field: str = None,
                          output_file: str = None):
    """
    Plot sonar data field vs metadata field.
    If metadata not found, just plot sonar data.
    
    Args:
        hdf5_file: Path to HDF5 file
        metadata_table: Name of metadata table (e.g., 'PressureSensorReadingMessages')
        field_name: Field name in metadata table
        sonar_field: Field name in sonar data (default: same as field_name)
        output_file: Path to save PNG (if None, show on screen)
    """
    if sonar_field is None:
        sonar_field = field_name
    
    with HDF5Loader(hdf5_file) as loader:
        # Get low frequency (first available)
        freqs = loader.get_sonar_frequencies()
        if not freqs:
            print("No sonar data found")
            return
        
        low_freq, _ = min(freqs, key=lambda x: x[0])
        
        # Load sonar data (channel 0)
        sonar_data = loader.get_sonar_pings(low_freq, channel=0)
        sonar_times = _convert_timestamps(sonar_data['timestamps'])
        sonar_values = sonar_data.get(sonar_field)
        
        if sonar_values is None:
            print(f"Field '{sonar_field}' not found in sonar data")
            print(f"Available sonar fields: {list(sonar_data.keys())}")
            return
        
        # Try to load metadata
        metadata = loader.get_metadata_table(metadata_table)
        meta_times = None
        meta_values = None
        
        if metadata:
            meta_times = _convert_timestamps(metadata.get('timestamp'))
            meta_values = metadata.get(field_name)
            
            if meta_values is None:
                print(f"Field '{field_name}' not found in {metadata_table}")
                print(f"Available fields: {list(metadata.keys())}")
        else:
            print(f"Metadata table '{metadata_table}' not found")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(sonar_times, sonar_values, 'b-', label=f'Sonar {sonar_field}', linewidth=2)
    
    if meta_times is not None and meta_values is not None:
        ax.plot(meta_times, meta_values, 'r.', label=f'Metadata {field_name}', markersize=6)
    
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(field_name)
    ax.set_title(f'{metadata_table}: {field_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # fig.autofmt_xdate(rotation=45, ha='right')
    # FIXME: readable timestamps

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=100)
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_stacked_timeseries(hdf5_file: str, 
                           fields: List[Tuple[str, str, Optional[str]]] = None,
                           title: str = "Stacked Timeseries",
                           output_file: str = None):
    """
    Plot multiple fields as stacked timeseries subplots.
    
    Args:
        hdf5_file: Path to HDF5 file
        fields: List of (metadata_table, field_name, sonar_field) tuples
                If sonar_field is None, uses field_name
        title: Title for the figure
        output_file: Path to save PNG (if None, show on screen)
    
    Example:
        plot_stacked_timeseries(
            'file.h5',
            fields=[
                ('PitchRollDataMessages', 'roll_deg', 'roll'),
                ('PitchRollDataMessages', 'pitch_deg', 'pitch'),
                ('PitchRollDataMessages', 'heading', 'heading'),
            ],
            title='Attitude (Roll, Pitch, Heading)'
        )
    """
    if fields is None:
        fields = []
    
    with HDF5Loader(hdf5_file) as loader:
        # Get low frequency
        freqs = loader.get_sonar_frequencies()
        if not freqs:
            print("No sonar data found")
            return
        
        low_freq, _ = min(freqs, key=lambda x: x[0])
        
        # Load sonar data once
        sonar_data = loader.get_sonar_pings(low_freq, channel=0)
        sonar_times = _convert_timestamps(sonar_data['timestamps'])
        
        # Prepare data for all fields
        plot_data = []
        
        for metadata_table, field_name, sonar_field in fields:
            if sonar_field is None:
                sonar_field = field_name
            
            sonar_values = sonar_data.get(sonar_field)
            
            # Try to load metadata
            metadata = loader.get_metadata_table(metadata_table)
            meta_times = None
            meta_values = None
            
            if metadata:
                meta_times = _convert_timestamps(metadata.get('timestamp'))
                meta_values = metadata.get(field_name)
            
            plot_data.append({
                'label': field_name,
                'sonar_times': sonar_times,
                'sonar_values': sonar_values,
                'meta_times': meta_times,
                'meta_values': meta_values,
            })
    
    # Create subplots
    fig, axes = plt.subplots(len(plot_data), 1, 
                             figsize=(12, 3 * len(plot_data)), 
                             sharex=True)
    
    # Handle single subplot case
    if len(plot_data) == 1:
        axes = [axes]
    
    for ax, data in zip(axes, plot_data):
        if data['sonar_values'] is not None:
            ax.plot(data['sonar_times'], data['sonar_values'], '.-', 
                   label='Sonar', linewidth=2)
        
        if data['meta_values'] is not None and data['meta_times'] is not None:
            ax.plot(data['meta_times'], data['meta_values'], '.', 
                   label='Metadata', markersize=6)
        
        ax.set_ylabel(data['label'])
        ax.grid(True, alpha=0.3)
    
    # Set x-axis label on bottom plot
    axes[-1].set_xlabel('Timestamp')
    axes[0].legend(loc='best')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # # Format x-axis on bottom plot
    # axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    # axes[-1].set_xlabel('Time')    
    # fig.autofmt_xdate(rotation=45, ha='right')
    # FIXME: readable timestamps
    
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=100)
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()