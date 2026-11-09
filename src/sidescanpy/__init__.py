"""
sidescanpy: read, correct, grid and export sidescan sonar data.

Subpackages, in import order (each may import only those before it):

- :mod:`sidescanpy.core` -- the data model: :class:`~sidescanpy.core.SideScanSurvey`,
  frequency channels, schema accessors and processing history.
- :mod:`sidescanpy.io` -- EdgeTech JSF, Klein SDF and XTF readers,
  ``.ss.h5`` and NetCDF files, target catalogs.
- :mod:`sidescanpy.processing` -- correction steps (TVG, geometric
  ranges, georeferencing, navigation interpolation), slicing, bottom and
  noise detection, segments.
- :mod:`sidescanpy.analysis` -- measurements that return result
  dataclasses (units, nadir, usable range, TVG decay fit, ...).
- :mod:`sidescanpy.grids` -- gridding onto UTM or geographic rasters,
  GeoTIFF output.
- :mod:`sidescanpy.targets` -- target catalogs and matching targets to
  pings.
- :mod:`sidescanpy.export` -- image windows, chips and COCO/YOLO
  datasets.
- :mod:`sidescanpy.viz` -- plotting helpers (needs the ``viz`` extra).

Also :mod:`sidescanpy.datasets` (public example recordings) and
:mod:`sidescanpy.log` (logging setup for scripts).
"""

import logging

from sidescanpy._version import __version__  # noqa: F401

__author__ = "Russ Shomberg"

from sidescanpy.core.schema import SideScanAccessor  # noqa: F401

# Library convention: no output unless the application configures logging
# (see sidescanpy.log.configure_logging).
logging.getLogger("sidescanpy").addHandler(logging.NullHandler())
