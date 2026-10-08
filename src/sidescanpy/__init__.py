"""
Sidescanpy: Process and analyze sonar data.

Main modules:
- converters: Format conversion (JSF → netCDF, etc)
- analysis: Data analysis and processing
- io: I/O utilities for netCDF
- visualization: Plotting and visualization
"""

import logging

from sidescanpy._version import __version__  # noqa: F401

__author__ = "Russ Shomberg"

from sidescanpy.core.schema import SideScanAccessor  # noqa: F401

# Library convention: no output unless the application configures logging
# (see sidescanpy.log.configure_logging).
logging.getLogger("sidescanpy").addHandler(logging.NullHandler())
