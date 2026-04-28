"""
HDF5 file I/O for sonar data

2026 rshomberg@uri.edu
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# TODO: modify to match existing data
class JSFtoHDF5Manager:
    """Convert JSF messages to HDF5 format"""
    # TODO: convert to functions
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_messages(self, messages: List) -> None:
        """Save all messages to HDF5"""
        logging.info(f"Saving {len(messages)} messages to {self.output_file}")
        
        with h5py.File(self.output_file, 'w') as f:
            # Separate message types
            sonar_msgs = [m for m in messages 
                         if type(m).__name__ == 'SonarDataMessage']
            legacy_msgs = [m for m in messages 
                          if type(m).__name__ == 'LegacySideScanDataMessage']
            
            # Save primary data
            if sonar_msgs:
                self._save_sonar_data(f, sonar_msgs)
                logging.info(f"  Saved {len(sonar_msgs)} sonar pings")
            
            if legacy_msgs:
                self._save_legacy_sonar(f, legacy_msgs)
                logging.info(f"  Saved {len(legacy_msgs)} legacy sonar pings")
            
            # Save metadata (all other messages)
            metadata_msgs = [m for m in messages 
                            if type(m).__name__ not in ['SonarDataMessage', 'LegacySideScanDataMessage']]
            if metadata_msgs:
                self._save_metadata(f, metadata_msgs)
                logging.info(f"  Saved {len(metadata_msgs)} metadata messages")
    
    def _save_sonar_data(self, hdf5_file: h5py.File, messages: List) -> None:
        """Save SonarDataMessages with ragged arrays by frequency/channel"""
        sonar_grp = hdf5_file.create_group('sonar')
        
        # Group by (frequency, channel)
        freq_channel_groups = {}
        for msg in messages:
            key = (msg.transmit_freq_start, msg.channel)
            if key not in freq_channel_groups:
                freq_channel_groups[key] = []
            freq_channel_groups[key].append(msg)
        
        for (freq, channel), msgs in freq_channel_groups.items():
            group_name = f'freq_{int(freq)}_ch_{channel}'
            grp = sonar_grp.create_group(group_name)
            
            logging.debug(f"  Saving {len(msgs)} pings: {group_name}")
            
            # Core arrays
            ping_numbers = np.array([m.ping_number for m in msgs], dtype=np.int32)
            timestamps = np.array([int(m.timestamp.astype('datetime64[ns]').astype('int64')) 
                                  for m in msgs], dtype=np.int64)
            n_samples = np.array([m.N_samples for m in msgs], dtype=np.int32)
            
            grp.create_dataset('ping_number', data=ping_numbers)
            grp.create_dataset('timestamp', data=timestamps)
            grp.create_dataset('sample_count', data=n_samples)
            
            # Ragged array: flatten all samples, track indices
            all_samples = np.concatenate([m.samples for m in msgs if hasattr(m, 'samples')])
            sample_indices = np.concatenate([[0], np.cumsum(n_samples)[:-1]])
            
            grp.create_dataset('samples', data=all_samples, compression='gzip', compression_opts=4)
            grp.create_dataset('sample_indices', data=sample_indices)
            
            # Metadata arrays (one per ping)
            grp.create_dataset('latitude', data=np.array([m.latitude for m in msgs], dtype=np.float32))
            grp.create_dataset('longitude', data=np.array([m.longitude for m in msgs], dtype=np.float32))
            grp.create_dataset('heading', data=np.array([m.heading for m in msgs], dtype=np.float32))
            grp.create_dataset('pitch', data=np.array([m.pitch for m in msgs], dtype=np.float32))
            grp.create_dataset('roll', data=np.array([m.roll for m in msgs], dtype=np.float32))
            grp.create_dataset('depth', data=np.array([m.depth for m in msgs], dtype=np.float32))
            grp.create_dataset('altitude', data=np.array([m.altitude for m in msgs], dtype=np.float32))
            grp.create_dataset('course', data=np.array([m.course for m in msgs], dtype=np.float32))
            grp.create_dataset('speed', data=np.array([m.speed for m in msgs], dtype=np.float32))
            # TODO: add sound speed here and other useful datatypes
            
            # Attributes (one per group)
            grp.attrs['transmit_freq_start'] = msgs[0].transmit_freq_start
            grp.attrs['transmit_freq_end'] = msgs[0].transmit_freq_end
            grp.attrs['sample_freq'] = msgs[0].sample_freq
            grp.attrs['sample_interval'] = msgs[0].sample_interval
            grp.attrs['sweep_length'] = msgs[0].sweep_length
            grp.attrs['channel'] = channel
            grp.attrs['num_pings'] = len(msgs)
    
    def _save_legacy_sonar(self, hdf5_file: h5py.File, messages: List) -> None:
        """Save LegacySideScanDataMessages (similar structure to sonar data)"""
        legacy_grp = hdf5_file.create_group('legacy_sonar')
        
        # Group by frequency/channel if available
        freq_channel_groups = {}
        for msg in messages:
            freq = getattr(msg, 'frequency', 0)
            channel = getattr(msg, 'channel', 0)
            key = (freq, channel)
            if key not in freq_channel_groups:
                freq_channel_groups[key] = []
            freq_channel_groups[key].append(msg)
        
        for (freq, channel), msgs in freq_channel_groups.items():
            group_name = f'freq_{int(freq)}_ch_{channel}' if freq else 'default'
            grp = legacy_grp.create_group(group_name)
            
            # Similar structure as sonar data
            n_samples = np.array([getattr(m, 'N_samples', len(getattr(m, 'samples', []))) 
                                 for m in msgs], dtype=np.int32)
            
            grp.create_dataset('sample_count', data=n_samples)
            
            if all(hasattr(m, 'samples') for m in msgs):
                all_samples = np.concatenate([m.samples for m in msgs])
                sample_indices = np.concatenate([[0], np.cumsum(n_samples)[:-1]])
                grp.create_dataset('samples', data=all_samples, compression='gzip', compression_opts=4)
                grp.create_dataset('sample_indices', data=sample_indices)
    
    def _save_metadata(self, hdf5_file: h5py.File, messages: List) -> None:
        """Save non-sonar messages as metadata tables"""
        metadata_grp = hdf5_file.create_group('metadata')
        
        # Group by message type
        msg_by_type = {}
        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type not in msg_by_type:
                msg_by_type[msg_type] = []
            msg_by_type[msg_type].append(msg)
        
        # Save each message type as a table
        for msg_type, msgs in msg_by_type.items():
            self._save_message_table(metadata_grp, msg_type, msgs)
    
    def _save_message_table(self, parent_grp: h5py.Group, table_name: str, messages: List) -> None:
        """Save messages of same type as HDF5 table"""
        grp = parent_grp.create_group(table_name)
        
        # Extract fields from first message
        sample_msg = messages[0]
        
        # Get all attributes that are numeric or datetime
        for attr_name in dir(sample_msg):
            if attr_name.startswith('_') or callable(getattr(sample_msg, attr_name)):
                continue
            
            try:
                values = []
                for msg in messages:
                    val = getattr(msg, attr_name)
                    values.append(val)
                
                # Try to convert to numpy array
                if all(isinstance(v, (int, float, bool, np.integer, np.floating)) for v in values):
                    arr = np.array(values)
                    grp.create_dataset(attr_name, data=arr, compression='gzip', compression_opts=4)
                elif all(isinstance(v, np.datetime64) for v in values):
                    arr = np.array([int(v.astype('datetime64[ns]').astype('int64')) for v in values])
                    grp.create_dataset(attr_name, data=arr)
            except (TypeError, ValueError):
                # Skip fields that can't be easily converted
                continue
        
        grp.attrs['num_records'] = len(messages)


class HDF5Loader:
    """Load data from HDF5 file"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, 'r')
    
    def get_sonar_frequencies(self) -> List[Tuple[float, int]]:
        """Get list of (frequency, channel) tuples available"""
        if 'sonar' not in self.file:
            return []
        
        freq_channels = []
        for group_name in self.file['sonar'].keys():
            # Parse: freq_580000_ch_0
            parts = group_name.split('_')
            freq = float(parts[1])
            channel = int(parts[3])
            freq_channels.append((freq, channel))
        
        return sorted(freq_channels)
    
    def get_sonar_pings(self, frequency: float, channel: int = 0) -> Dict:
        """Load sonar data for specific frequency/channel
        
        Returns dict with keys: ping_numbers, timestamps, latitude, longitude, 
        heading, pitch, roll, depth, altitude, samples
        """
        group_name = f'freq_{int(frequency)}_ch_{channel}'
        grp = self.file[f'sonar/{group_name}']
        
        # Load metadata
        data = {
            'ping_numbers': grp['ping_number'][:],
            'timestamps': grp['timestamp'][:],
            'latitude': grp['latitude'][:],
            'longitude': grp['longitude'][:],
            'heading': grp['heading'][:],
            'pitch': grp['pitch'][:],
            'roll': grp['roll'][:],
            'depth': grp['depth'][:],
            'altitude': grp['altitude'][:],
            'course': grp['course'][:],
            'speed': grp['speed'][:],
        }
        
        # Reconstruct ragged samples array
        sample_indices = grp['sample_indices'][:]
        all_samples = grp['samples'][:]
        sample_counts = grp['sample_count'][:]
        
        samples = []
        for i in range(len(sample_indices)):
            start_idx = sample_indices[i]
            count = sample_counts[i]
            end_idx = start_idx + count
            samples.append(all_samples[start_idx:end_idx])
        
        data['samples'] = samples
        data['frequency'] = frequency
        data['channel'] = channel
        
        return data
    
    def get_metadata_table(self, table_name: str) -> Dict[str, np.ndarray]:
        """Load metadata table as dict of arrays"""
        if 'metadata' not in self.file or table_name not in self.file['metadata']:
            return {}
        
        grp = self.file[f'metadata/{table_name}']
        return {name: grp[name][:] for name in grp.keys()}
    
    def list_metadata_tables(self) -> List[str]:
        """Get available metadata table names"""
        if 'metadata' not in self.file:
            return []
        return list(self.file['metadata'].keys())
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Inspect HDF5 sonar file')
    parser.add_argument('hdf5_file', type=Path, help='Path to HDF5 file')
    args = parser.parse_args()
    
    with HDF5Loader(str(args.hdf5_file)) as loader:
        # Show sonar data
        freq_channels = loader.get_sonar_frequencies()
        print(f"Sonar data: {len(freq_channels)} frequency/channel combinations")
        for freq, ch in freq_channels:
            data = loader.get_sonar_pings(freq, ch)
            print(f"  {int(freq)} Hz, Channel {ch}: {len(data['ping_numbers'])} pings, "
                  f"{sum(len(s) for s in data['samples'])} total samples")
        
        # Show metadata
        metadata_tables = loader.list_metadata_tables()
        print(f"\nMetadata tables: {len(metadata_tables)}")
        for table_name in metadata_tables:
            table = loader.get_metadata_table(table_name)
            if table:
                n_records = len(next(iter(table.values())))
                fields = list(table.keys())
                print(f"  {table_name}: {n_records} records, {len(fields)} fields")