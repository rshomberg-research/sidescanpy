"""Generate visualization plots from sonar data"""

import h5py
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# FIXME: is this used? It should probably get moved to the vis module


class SonarPlotter:
    """Generate various sonar data visualizations"""
    
    def __init__(self, hdf5_file: str):
        self.hdf5_file = Path(hdf5_file)
    
    def plot_waterfall(self, 
                       frequency: float, 
                       channel: int = 0,
                       ping_start: Optional[int] = None,
                       ping_end: Optional[int] = None,
                       output_file: Optional[str] = None) -> np.ndarray:
        """
        Generate time-series waterfall plot (pings vs slant range)
        
        Args:
            frequency: Transmit frequency to plot
            channel: Channel to plot (0 or 1)
            ping_start: Start ping number (None = first)
            ping_end: End ping number (None = last)
            output_file: Path to save PNG (if None, just return array)
            
        Returns:
            2D numpy array (pings x samples)
        """
        logger.info(f"Generating waterfall: {int(frequency)} Hz, Ch {channel}")
        
        with h5py.File(self.hdf5_file, 'r') as f:
            data = self._load_sonar_section(
                f, frequency, channel, ping_start, ping_end
            )
        
        if data is None:
            logger.warning(f"No data found for {frequency} Hz, Ch {channel}")
            return None
        
        # Create waterfall array (uniform sample length for visualization)
        waterfall = self._create_waterfall_array(data['samples'])
        
        # Plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(waterfall.T, aspect='auto', origin='upper', cmap='gray')

# FIXME: everything below here looks broken?
from typing import ClassVar, Type, Dict
        ax.set_xlabel('Ping Number')
        ax.set_ylabel('Slant Range Sample')
        ax.set_title(f'Waterfall: {int(frequency)} Hz, Channel {channel}')
        
        plt.colorbar(im, ax=ax, label='Intensity')
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved waterfall to {output_file}")
        
        plt.close()
        return waterfall
    
    def plot_corrected_mosaic(self,
                             frequency: float,
                             channel: int = 0,
                             ping_start: Optional[int] = None,
                             ping_end: Optional[int] = None,
                             output_file: Optional[str] = None,
                             resolution: float = 0.01) -> np.ndarray:
        """
        Generate corrected mosaic (distance along track vs distance across track)
        Accounts for vehicle altitude and attitude
        
        Args:
            frequency: Transmit frequency to plot
            channel: Channel to plot (0 or 1)
            ping_start: Start ping number
            ping_end: End ping number
            output_file: Path to save PNG
            resolution: Grid resolution in meters
            
        Returns:
            2D mosaic array
        """
        logger.info(f"Generating corrected mosaic: {int(frequency)} Hz, Ch {channel}")
        
        with h5py.File(self.hdf5_file, 'r') as f:
            data = self._load_sonar_section(
                f, frequency, channel, ping_start, ping_end
            )
        
        if data is None:
            logger.warning(f"No data found for {frequency} Hz, Ch {channel}")
            return None
        
        # Get sonar parameters
        sample_interval = self._get_sonar_params(frequency, channel)['sample_interval']
        
        # Build georeferenced mosaic
        mosaic = self._build_mosaic(
            data['samples'],
            data['latitude'],
            data['longitude'],
            data['heading'],
            data['altitude'],
            sample_interval,
            resolution,
            channel
        )
        
        # Plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(mosaic, aspect='equal', origin='upper', cmap='gray')
        
        ax.set_xlabel('Distance Across Track (m)')
        ax.set_ylabel('Distance Along Track (m)')
        ax.set_title(f'Corrected Mosaic: {int(frequency)} Hz, Channel {channel}')
        
        plt.colorbar(im, ax=ax, label='Intensity')
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved mosaic to {output_file}")
        
        plt.close()
        return mosaic
    
    def plot_georectified_mosaic(self,
                                frequency: float,
                                channel: int = 0,
                                ping_start: Optional[int] = None,
                                ping_end: Optional[int] = None,
                                output_file: Optional[str] = None,
                                resolution: float = 0.01) -> np.ndarray:
        """
        Generate georectified mosaic (latitude vs longitude grid)
        
        Args:
            frequency: Transmit frequency to plot
            channel: Channel to plot (0 or 1)
            ping_start: Start ping number
            ping_end: End ping number
            output_file: Path to save PNG
            resolution: Grid resolution in degrees
            
        Returns:
            2D georectified mosaic array
        """
        logger.info(f"Generating georectified mosaic: {int(frequency)} Hz, Ch {channel}")
        
        with h5py.File(self.hdf5_file, 'r') as f:
            data = self._load_sonar_section(
                f, frequency, channel, ping_start, ping_end
            )
        
        if data is None:
            logger.warning(f"No data found for {frequency} Hz, Ch {channel}")
            return None
        
        # Get sonar parameters
        params = self._get_sonar_params(frequency, channel)
        sample_interval = params['sample_interval']
        
        # Build georectified mosaic
        mosaic, extent = self._build_georectified_mosaic(
            data['samples'],
            data['latitude'],
            data['longitude'],
            data['heading'],
            data['altitude'],
            sample_interval,
            resolution,
            channel
        )
        
        # Plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(mosaic, aspect='equal', origin='upper', cmap='gray', extent=extent)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Georectified Mosaic: {int(frequency)} Hz, Channel {channel}')
        
        plt.colorbar(im, ax=ax, label='Intensity')
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved georectified mosaic to {output_file}")
        
        plt.close()
        return mosaic
    
    # ---- Helper Methods ----
    
    def _load_sonar_section(self, hdf5_file: h5py.File, frequency: float, channel: int,
                           ping_start: Optional[int] = None, 
                           ping_end: Optional[int] = None) -> Optional[Dict]:
        """Load subset of sonar data"""
        group_name = f'freq_{int(frequency)}_ch_{channel}'
        
        if 'sonar' not in hdf5_file or group_name not in hdf5_file['sonar']:
            return None
        
        grp = hdf5_file[f'sonar/{group_name}']
        
        # Get ping indices
        ping_numbers = grp['ping_number'][:]
        
        if ping_start is None:
            idx_start = 0
        else:
            idx_start = np.searchsorted(ping_numbers, ping_start)
        
        if ping_end is None:
            idx_end = len(ping_numbers)
        else:
            idx_end = np.searchsorted(ping_numbers, ping_end) + 1
        
        logger.debug(f"Loading pings {idx_start}:{idx_end} (ping numbers {ping_numbers[idx_start]}-{ping_numbers[idx_end-1]})")
        
        # Load data
        sample_indices = grp['sample_indices'][idx_start:idx_end]
        sample_counts = grp['sample_count'][idx_start:idx_end]
        all_samples = grp['samples'][:]
        
        # Reconstruct ragged samples
        samples = []
        for i, (idx, count) in enumerate(zip(sample_indices, sample_counts)):
            samples.append(all_samples[idx:idx+count])
        
        return {
            'samples': samples,
            'latitude': grp['latitude'][idx_start:idx_end],
            'longitude': grp['longitude'][idx_start:idx_end],
            'heading': grp['heading'][idx_start:idx_end],
            'pitch': grp['pitch'][idx_start:idx_end],
            'roll': grp['roll'][idx_start:idx_end],
            'altitude': grp['altitude'][idx_start:idx_end],
            'ping_numbers': ping_numbers[idx_start:idx_end],
        }
    
    def _get_sonar_params(self, frequency: float, channel: int) -> Dict:
        """Get sonar acquisition parameters"""
        with h5py.File(self.hdf5_file, 'r') as f:
            group_name = f'freq_{int(frequency)}_ch_{channel}'
            grp = f[f'sonar/{group_name}']
            
            return {
                'transmit_freq_start': grp.attrs['transmit_freq_start'],
                'transmit_freq_end': grp.attrs['transmit_freq_end'],
                'sample_freq': grp.attrs['sample_freq'],
                'sample_interval': grp.attrs['sample_interval'],
                'sweep_length': grp.attrs['sweep_length'],
            }
    
    def _create_waterfall_array(self, samples: List[np.ndarray], 
                               max_samples: Optional[int] = None) -> np.ndarray:
        """
        Create uniform 2D array from ragged samples (pads with zeros)
        
        Args:
            samples: List of 1D sample arrays (ragged)
            max_samples: Maximum samples to use (None = auto)
            
        Returns:
            2D array (num_pings x max_samples)
        """
        if max_samples is None:
            max_samples = max(len(s) for s in samples)
        
        waterfall = np.zeros((len(samples), max_samples), dtype=samples[0].dtype)
        
        for i, s in enumerate(samples):
            n = min(len(s), max_samples)
            waterfall[i, :n] = s[:n]
        
        return waterfall
    
    def _build_mosaic(self, samples: List[np.ndarray],
                     latitude: np.ndarray, longitude: np.ndarray,
                     heading: np.ndarray, altitude: np.ndarray,
                     sample_interval: float, resolution: float,
                     channel: int) -> np.ndarray:
        """
        Build corrected mosaic (along-track vs across-track).
        Very simplified - projects samples to ground based on altitude and attitude.
        """
        # Calculate cross-track distance for each sample
        max_samples = max(len(s) for s in samples)
        cross_track = np.arange(max_samples) * sample_interval
        
        # Calculate along-track distance from first ping
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)
        
        # Haversine distance
        dlat = np.diff(lat_rad, prepend=lat_rad[0])
        dlon = np.diff(lon_rad, prepend=lon_rad[0])
        along_track = np.cumsum(6371000 * np.sqrt(dlat**2 + (np.cos(lat_rad) * dlon)**2))
        
        # Create grid
        along_grid = np.arange(0, along_track[-1], resolution)
        across_grid = cross_track
        
        # Create mosaic
        mosaic = np.zeros((len(along_grid), len(across_grid)))
        
        for i, ping_samples in enumerate(samples):
            # Find grid position for this ping
            j = np.searchsorted(along_grid, along_track[i])
            if j < len(along_grid):
                # Map samples to across-track grid
                n = min(len(ping_samples), len(across_grid))
                mosaic[j, :n] = ping_samples[:n]
        
        return mosaic
    
    def _build_georectified_mosaic(self, samples: List[np.ndarray],
                                  latitude: np.ndarray, longitude: np.ndarray,
                                  heading: np.ndarray, altitude: np.ndarray,
                                  sample_interval: float, resolution: float,
                                  channel: int) -> Tuple[np.ndarray, Tuple]:
        """
        Build georectified mosaic (lat/lon grid).
        Projects each sample to lat/lon based on heading, altitude, and cross-track distance.
        """
        # Get lat/lon bounds
        lat_min, lat_max = latitude.min(), latitude.max()
        lon_min, lon_max = longitude.min(), longitude.max()
        
        # Create grid
        lat_grid = np.arange(lat_min, lat_max, resolution)
        lon_grid = np.arange(lon_min, lon_max, resolution)
        
        mosaic = np.zeros((len(lat_grid), len(lon_grid)))
        
        # Simple projection: each ping maps samples perpendicular to heading
        for i, ping_samples in enumerate(samples):
            ping_lat = latitude[i]
            ping_lon = longitude[i]
            ping_heading = heading[i]
            ping_alt = altitude[i]
            
            # Convert heading to radians (0=North, 90=East)
            heading_rad = np.radians(ping_heading)
            
            for j, sample_val in enumerate(ping_samples):
                # Cross-track distance
                cross_dist = j * sample_interval
                
                # Project to lat/lon (simplified)
                delta_lat = (cross_dist * np.cos(heading_rad)) / 111000  # meters to degrees
                delta_lon = (cross_dist * np.sin(heading_rad)) / (111000 * np.cos(np.radians(ping_lat)))
                
                sample_lat = ping_lat + delta_lat
                sample_lon = ping_lon + delta_lon
                
                # Find grid position
                lat_idx = np.searchsorted(lat_grid, sample_lat)
                lon_idx = np.searchsorted(lon_grid, sample_lon)
                
                if 0 <= lat_idx < len(lat_grid) and 0 <= lon_idx < len(lon_grid):
                    mosaic[lat_idx, lon_idx] = max(mosaic[lat_idx, lon_idx], sample_val)
        
        extent = (lon_min, lon_max, lat_max, lat_min)  # imshow extent
        return mosaic, extent


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Generate sonar plots')
    parser.add_argument('hdf5_file', type=Path, help='Input HDF5 file')
    parser.add_argument('--frequency', type=float, required=True, help='Frequency to plot (Hz)')
    parser.add_argument('--channel', type=int, default=0, help='Channel (0 or 1)')
    parser.add_argument('--type', choices=['waterfall', 'mosaic', 'georectified'],
                       default='waterfall', help='Plot type')
    parser.add_argument('--ping-start', type=int, help='Start ping number')
    parser.add_argument('--ping-end', type=int, help='End ping number')
    parser.add_argument('--output', type=Path, help='Output PNG file')
    args = parser.parse_args()
    
    plotter = SonarPlotter(str(args.hdf5_file))
    
    if args.type == 'waterfall':
        plotter.plot_waterfall(args.frequency, args.channel, 
                              args.ping_start, args.ping_end, 
                              str(args.output) if args.output else None)
    elif args.type == 'mosaic':
        plotter.plot_corrected_mosaic(args.frequency, args.channel,
                                     args.ping_start, args.ping_end,
                                     str(args.output) if args.output else None)
    elif args.type == 'georectified':
        plotter.plot_georectified_mosaic(args.frequency, args.channel,
                                        args.ping_start, args.ping_end,
                                        str(args.output) if args.output else None)