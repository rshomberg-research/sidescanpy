"""
Convert HDF5 to CF-compliant NetCDF

2026 rshomberg@uri.edu
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from .hdf5_manager import HDF5Loader    # FIXME: improve modularity

class HDF5toCFNetCDF:
    """Convert HDF5 sonar data to CF-compliant NetCDF"""
    # TODO: convert to functions
    
    def __init__(self, hdf5_file: str):
        self.hdf5_file = Path(hdf5_file)
    
    def export_all_frequencies(self, output_dir: Path | None = None) -> dict:
        """
        Export to separate NetCDF files for each frequency.
        
        Returns:
            Dict mapping frequency to output file path
        """
        if output_dir is None:
            output_dir = self.hdf5_file.parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with HDF5Loader(str(self.hdf5_file)) as loader:
            freq_channels = loader.get_sonar_frequencies()
        
        if not freq_channels:
            logging.warning("No sonar data found")
            return {}
        
        # Get unique frequencies
        frequencies = sorted(set(f for f, ch in freq_channels))
        
        output_files = {}
        
        for freq in frequencies:
            output_file = output_dir / f"{self.hdf5_file.stem}_freq{int(freq)}.nc"
            self._export_frequency(freq, str(output_file))
            output_files[freq] = str(output_file)
            logging.info(f"Exported {int(freq)} Hz to {output_file.name}")
        
        return output_files
    
    def _export_frequency(self, frequency: float, output_file: str) -> None:
        """Export single frequency to NetCDF"""
        
        with HDF5Loader(str(self.hdf5_file)) as loader:
            # Get all channels for this frequency
            freq_channels = [(f, ch) for f, ch in loader.get_sonar_frequencies() if f == frequency]
            
            if not freq_channels:
                logging.warning(f"No data found for {frequency} Hz")
                return
            
            # Create output file
            ds = nc.Dataset(output_file, 'w', format='NETCDF4')
            
            # Global attributes (CF-compliant)
            ds.Conventions = 'CF-1.8'
            ds.title = f'Sidescan Sonar Survey - {int(frequency)} Hz'
            ds.institution = 'University of Rhode Island'
            ds.source = f'JSF file: {self.hdf5_file.name}'
            ds.date_created = datetime.utcnow().isoformat()
            ds.history = f'Converted from HDF5 on {datetime.utcnow().isoformat()}'
            
            # Load data from all channels
            all_pings = []
            for freq, channel in freq_channels:
                data = loader.get_sonar_pings(freq, channel)
                all_pings.append({
                    'channel': channel,
                    'data': data,
                })
            
            if not all_pings:
                ds.close()
                return
            
            # Get common dimensions
            n_pings = len(all_pings[0]['data']['ping_numbers'])
            max_samples = max(len(s) for ping in all_pings 
                            for s in ping['data']['samples'])
            
            # Create dimensions
            ds.createDimension('time', n_pings)
            ds.createDimension('sample', max_samples)
            ds.createDimension('channel', len(all_pings))
            
            # Create coordinate variables (CF-required)
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'seconds since 1970-01-01 00:00:00'
            time_var.calendar = 'standard'
            time_var.long_name = 'Time'
            time_var.standard_name = 'time'
            time_var[:] = all_pings[0]['data']['timestamps'] / 1e9
            
            latitude_var = ds.createVariable('latitude', 'f4', ('time',))
            latitude_var.units = 'degrees_north'
            latitude_var.long_name = 'Latitude'
            latitude_var.standard_name = 'latitude'
            latitude_var[:] = all_pings[0]['data']['latitude']
            
            longitude_var = ds.createVariable('longitude', 'f4', ('time',))
            longitude_var.units = 'degrees_east'
            longitude_var.long_name = 'Longitude'
            longitude_var.standard_name = 'longitude'
            longitude_var[:] = all_pings[0]['data']['longitude']
            
            channel_var = ds.createVariable('channel', 'i1', ('channel',))
            channel_var.long_name = 'Sonar channel'
            channel_var[:] = [p['channel'] for p in all_pings]
            
            # Create sample index variable
            sample_var = ds.createVariable('sample', 'i4', ('sample',))
            sample_var.long_name = 'Sample index'
            sample_var.units = '1'
            sample_var[:] = np.arange(max_samples)
            
            # Create sonar parameters as global attributes
            sonar_params = self._get_sonar_params(frequency)
            if sonar_params:
                # print(sonar_params.keys())
                # ds.transmit_freq_start = sonar_params['transmit_freq_start']
                # ds.transmit_freq_end = sonar_params['transmit_freq_end']
                # ds.sample_rate = sonar_params['sample_freq']
                # ds.sample_interval = sonar_params['sample_interval']
                # ds.sweep_length = sonar_params['sweep_length']
                pass    # FIXME: these are stored separately in the hdf5
            
            # Acoustic data (with padding for ragged arrays)
            acoustic = ds.createVariable('acoustic_intensity', 'f4', 
                                        ('channel', 'time', 'sample'),
                                        fill_value=np.nan, 
                                        # compression='gzip'
                                        )
            acoustic.units = 'raw_counts'
            acoustic.long_name = 'Acoustic backscatter intensity'
            acoustic.coordinates = 'latitude longitude'
            
            for ch_idx, ping_data in enumerate(all_pings):
                for t_idx, sample_array in enumerate(ping_data['data']['samples']):
                    n_samples = len(sample_array)
                    acoustic[ch_idx, t_idx, :n_samples] = sample_array
            
            # Metadata variables
            heading_var = ds.createVariable('heading', 'f4', ('time',), 
                                            # compression='gzip',
                                            )
            heading_var.units = 'degrees'
            heading_var.long_name = 'Vessel heading'
            heading_var.standard_name = 'platform_orientation'
            heading_var[:] = all_pings[0]['data']['heading']
            
            pitch_var = ds.createVariable('pitch', 'f4', ('time',), 
                                        #   compression='gzip',
                                          )
            pitch_var.units = 'degrees'
            pitch_var.long_name = 'Vessel pitch'
            pitch_var[:] = all_pings[0]['data']['pitch']
            
            roll_var = ds.createVariable('roll', 'f4', ('time',), 
                                        #  compression='gzip',
                                         )
            roll_var.units = 'degrees'
            roll_var.long_name = 'Vessel roll'
            roll_var[:] = all_pings[0]['data']['roll']
            
            depth_var = ds.createVariable('depth', 'f4', ('time',), 
                                        #   compression='gzip',
                                          )
            depth_var.units = 'meters'
            depth_var.long_name = 'Water depth'
            depth_var.standard_name = 'sea_water_depth'
            depth_var[:] = all_pings[0]['data']['depth']
            
            altitude_var = ds.createVariable('altitude', 'f4', ('time',),
                                            #   compression='gzip',
                                              )
            altitude_var.units = 'meters'
            altitude_var.long_name = 'Altitude above seafloor'
            altitude_var.standard_name = 'height_above_sea_floor'
            altitude_var[:] = all_pings[0]['data']['altitude']
            
            course_var = ds.createVariable('course', 'f4', ('time',), 
                                        #    compression='gzip',
                                           )
            course_var.units = 'degrees'
            course_var.long_name = 'Vessel course'
            course_var[:] = all_pings[0]['data']['course']
            
            speed_var = ds.createVariable('speed', 'f4', ('time',), 
                                        #   compression='gzip',
                                          )
            speed_var.units = 'meters per second'
            speed_var.long_name = 'Vessel speed'
            speed_var[:] = all_pings[0]['data']['speed']
            
            ds.close()
    
    def _get_sonar_params(self, frequency: float) -> dict:
        """Get sonar parameters"""
        with HDF5Loader(str(self.hdf5_file)) as loader:
            freq_channels = loader.get_sonar_frequencies()
            for f, ch in freq_channels:
                if f == frequency:
                    return loader.get_sonar_pings(f, ch)
        return {}
    
    def _get_sonar_data(self, frequency: float, channel: int):
        """Get raw sonar parameters from HDF5"""
        import h5py
        
        with h5py.File(self.hdf5_file, 'r') as f:
            group_name = f'freq_{int(frequency)}_ch_{channel}'
            grp = f[f'sonar/{group_name}']
            
            return {
                # 'transmit_freq_start': grp.attrs['transmit_freq_start'],
                # 'transmit_freq_end': grp.attrs['transmit_freq_end'],
                # 'sample_freq': grp.attrs['sample_freq'],
                'sample_interval': grp.attrs['sample_interval'],
                'sweep_length': grp.attrs['sweep_length'],
            }