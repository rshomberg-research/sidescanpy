"""
Sidescanpy: Process and analyze sonar data.

Main modules:
- converters: Format conversion (JSF → netCDF, etc)
- analysis: Data analysis and processing
- io: I/O utilities for netCDF
- visualization: Plotting and visualization
"""

__version__ = "0.1.0"
__author__ = "Russ Shomberg"

# Import key classes/functions for convenient access
try:
    from sidescanpy.converters import convert_to_netcdf, get_converter
    from sidescanpy.io import load_netcdf, save_netcdf
except ImportError:
    pass  # Optional dependencies

__all__ = [
    'convert_to_netcdf',
    'get_converter',
    'load_netcdf',
    'save_netcdf',
]