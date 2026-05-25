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

from sidescanpy.core.schema import SideScanAccessor  # noqa: F401