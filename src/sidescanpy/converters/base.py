"""Abstract base class for sonar format converters."""

from abc import ABC, abstractmethod
from pathlib import Path
import xarray as xr


class SonarConverter(ABC):
    """Abstract base for sonar file format converters."""
    
    @abstractmethod
    def read(self, filepath: Path) -> dict:
        """
        Read raw sonar data from file.
        """
        pass
    
    @abstractmethod
    def to_netcdf(self, data: dict, output_path: Path) -> xr.Dataset:
        """
        Convert to standardized netCDF Dataset.      
        """
        pass
    
    def convert(self, input_path: Path, output_path: Path) -> xr.Dataset:
        """
        Full conversion pipeline.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        raw_data = self.read(input_path)
        dataset = self.to_netcdf(raw_data, output_path)
        dataset.to_netcdf(output_path)
        
        return dataset