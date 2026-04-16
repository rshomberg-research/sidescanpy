"""Apply auxiliary corrections to sonar data"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AuxiliaryProcessor:
    """Apply auxiliary messages to refine sonar data"""
    
    def __init__(self, hdf5_file: str):
        self.hdf5_file = Path(hdf5_file)
        self.file = None
    
    def process(self, output_file: Optional[str] = None) -> str:
        """
        Apply all available auxiliary corrections to sonar data.
        
        Args:
            output_file: Path to save refined data. If None, overwrites input.
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = str(self.hdf5_file)
        else:
            output_file = str(output_file)
            # Copy file if different from input
            if output_file != str(self.hdf5_file):
                import shutil
                shutil.copy(str(self.hdf5_file), output_file)
        
        logger.info(f"Refining sonar data from {self.hdf5_file}")
        
        with h5py.File(output_file, 'r+') as f:
            if 'sonar' not in f:
                logger.warning("No sonar data found in file")
                return output_file
            
            # Get all sonar groups
            sonar_grp = f['sonar']
            
            for group_name in sonar_grp.keys():
                logger.info(f"Processing: {group_name}")
                sonar_data_grp = sonar_grp[group_name]
                
                # Get sonar timestamps
                sonar_times = sonar_data_grp['timestamp'][:]
                
                # Get available auxiliary data and interpolate
                refinements = self._get_interpolated_auxiliary(f, sonar_times)
                
                # Apply refinements
                for field_name, values in refinements.items():
                    if field_name in sonar_data_grp:
                        logger.debug(f"  Updating {field_name}")
                        del sonar_data_grp[field_name]
                    sonar_data_grp.create_dataset(field_name, data=values, compression='gzip')
                
                if refinements:
                    logger.info(f"  Applied {len(refinements)} refinements")
        
        logger.info(f"Saved refined data to {output_file}")
        return output_file
    
    def _get_interpolated_auxiliary(self, hdf5_file: h5py.File, sonar_times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get interpolated auxiliary data for sonar timestamps.
        
        Returns dict mapping field names to interpolated arrays
        """
        refinements = {}
        
        if 'metadata' not in hdf5_file:
            logger.warning("No metadata found in HDF5 file")
            return refinements
        
        metadata_grp = hdf5_file['metadata']
        
        # Priority order for different message types
        # GPS/Navigation data
        if 'NMEARMCMessages' in metadata_grp:
            logger.debug("Interpolating NMEA RMC (GPS) data")
            gps_interp = self._interpolate_table(
                metadata_grp['NMEARMCMessages'],
                sonar_times,
                fields=['latitude', 'longitude', 'course', 'speed']
            )
            refinements.update(gps_interp)
        
        # DVL data (altitude, attitude, sound speed)
        if 'DopplerVelocityLogDataMessages' in metadata_grp:
            logger.debug("Interpolating DVL data")
            dvl_interp = self._interpolate_table(
                metadata_grp['DopplerVelocityLogDataMessages'],
                sonar_times,
                fields=['altitude', 'pitch', 'roll', 'heading', 'sound_speed']
            )
            refinements.update(dvl_interp)
        
        # Pitch/Roll data (often more accurate than DVL)
        if 'PitchRollDataMessages' in metadata_grp:
            logger.debug("Interpolating Pitch/Roll data")
            pitch_roll_interp = self._interpolate_table(
                metadata_grp['PitchRollDataMessages'],
                sonar_times,
                fields=['pitch', 'roll', 'heading']
            )
            refinements.update(pitch_roll_interp)
        
        # Pressure data (for depth correction)
        if 'PressureSensorReadingMessages' in metadata_grp:
            logger.debug("Interpolating Pressure data")
            pressure_interp = self._interpolate_table(
                metadata_grp['PressureSensorReadingMessages'],
                sonar_times,
                fields=['depth', 'pressure_psi', 'temperature_c', 'sound_speed_m_s']
            )
            refinements.update(pressure_interp)
        
        return refinements
    
    def _interpolate_table(self, table_grp: h5py.Group, sonar_times: np.ndarray, 
                          fields: List[str]) -> Dict[str, np.ndarray]:
        """
        Interpolate auxiliary table data to sonar timestamps.
        
        Args:
            table_grp: HDF5 group containing timestamp and data fields
            sonar_times: Target timestamps (int64, nanoseconds since epoch)
            fields: Field names to interpolate
            
        Returns:
            Dict mapping field names to interpolated arrays
        """
        if 'timestamp' not in table_grp:
            logger.warning(f"No timestamp in {table_grp.name}")
            return {}
        
        aux_times = table_grp['timestamp'][:]
        
        # Handle duplicate timestamps
        if len(aux_times) != len(np.unique(aux_times)):
            logger.warning(f"Duplicate timestamps in {table_grp.name}, using last value")
            _, unique_idx = np.unique(aux_times, return_index=True)
            aux_times = aux_times[sorted(unique_idx)]
        
        refinements = {}
        
        for field_name in fields:
            if field_name not in table_grp:
                logger.debug(f"  Field '{field_name}' not in {table_grp.name}")
                continue
            
            aux_values = table_grp[field_name][:]
            
            # Filter to unique times
            if len(aux_times) != len(np.unique(aux_times)):
                _, unique_idx = np.unique(aux_times, return_index=True)
                aux_values = aux_values[sorted(unique_idx)]
            
            # Interpolate to sonar times
            try:
                interp_values = np.interp(
                    sonar_times.astype(float),
                    aux_times.astype(float),
                    aux_values.astype(float),
                    left=np.nan,
                    right=np.nan
                )
                refinements[field_name] = interp_values.astype(aux_values.dtype)
                logger.debug(f"  Interpolated {field_name}: {np.sum(~np.isnan(interp_values))}/{len(interp_values)} valid")
            except Exception as e:
                logger.warning(f"  Failed to interpolate {field_name}: {e}")
        
        return refinements


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Refine sonar data with auxiliary messages')
    parser.add_argument('hdf5_file', type=Path, help='Input HDF5 file')
    parser.add_argument('--output', type=Path, help='Output file (default: overwrite input)')
    args = parser.parse_args()
    
    processor = AuxiliaryProcessor(str(args.hdf5_file))
    processor.process(str(args.output) if args.output else None)