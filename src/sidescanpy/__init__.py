"""
Sidescanpy: Process and analyze sonar data.

Main modules:
- converters: Format conversion (JSF → netCDF, etc)
- analysis: Data analysis and processing
- io: I/O utilities for netCDF
- visualization: Plotting and visualization
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Reads the installed package's actual metadata (pyproject.toml's
    # own version, embedded at build time) -- a hardcoded literal here
    # silently went stale for a long time (found stuck at "0.1.0" while
    # pyproject.toml had moved through the whole 0.4.x series), and this
    # value gets stamped into real .ss.h5 processing_history records
    # (processing/{georeferencing,interpolate,geometric,farrange}.py),
    # not just printed for humans -- a wrong version there is a real
    # provenance bug, not a cosmetic one.
    __version__ = version("sidescanpy")
except PackageNotFoundError:
    # Only reachable running from source with the package never
    # installed/built at all (e.g. a bare `sys.path` hack) -- every
    # real caller (editable install, wheel install) has metadata.
    __version__ = "0.0.0+unknown"

__author__ = "Russ Shomberg"

from sidescanpy.core.schema import SideScanAccessor  # noqa: F401