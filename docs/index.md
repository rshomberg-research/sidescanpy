# sidescanpy

Read, correct, grid and export sidescan sonar data.

sidescanpy converts EdgeTech JSF, Klein SDF and XTF surveys into one
`xarray`-based model stored as `.ss.h5`, then corrects it (TVG, ground
range, georeferencing), slices it, grids it to rasters and GeoTIFF, and
builds COCO/YOLO training datasets from target catalogs. Small analysis
functions measure a dataset's units, nadir, usable range and TVG state.

:::{admonition} Pre-release
:class: warning
Until 1.0, any 0.x release may change the `.ss.h5` file format, the
API and the CLI. Keep your raw sonar files, pin the version you use,
and expect to re-convert `.ss.h5` files after an upgrade. See
[Pre-release status](prerelease.md).
:::

[![A raw and a corrected EdgeTech line](assets/gallery/correct_a_line.png)](examples/plot_correct_a_line.rst)

*A public EdgeTech recording, raw (left) and after the bottom track, TVG
and slant-to-ground-range steps (right). Code and data:
[Correct a line](examples/plot_correct_a_line.rst).*

## Quickstart

```python
from sidescanpy import datasets
from sidescanpy.grids import grid_channel, to_geotiff
from sidescanpy.io import read_and_convert
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.processing import correct_survey

# A public Benthos XTF line; any JSF, SDF or XTF path works the same way.
path = datasets.fetch("jncc_mid_irish_sea_MIS11_0.xtf")
survey = read_and_convert([path], fmt="xtf")

# Navigation onto pings, TVG, slant to ground range, per-sample lat/lon.
survey = correct_survey(survey)
write_survey(survey, "MIS11_0.ss.h5")

# Grid one frequency at 1 m and write a GeoTIFF (needs the geo extra).
raster = grid_channel(survey.get_frequency("hz_400000").dataset, resolution_m=1.0)
to_geotiff(raster, "MIS11_0_400khz.tif")
```

[Install](install.md) covers the install and its extras. [Getting started](getting_started.md)
walks through this code step by step, with the output of each step.

## Where to go next

- **Start here:** [Install](install.md), [Getting started](getting_started.md).
- **Examples:** [real data, code and results](examples/index.rst), all on
  openly licensed recordings you can download.
- **Guide:** [Building a workflow](workflow.md), [Data model](core.md), [Processing](processing.md),
  [Gridding](grids.md), [Targets and datasets](datasets.md),
  [Analysis](analysis.md), [Visualization](viz.md),
  [Command line](cli.md), [Logging](logging.md), [Limitations](limitations.md).
- **Formats:** [overview](formats.md), [JSF](io_jsf.md), [SDF](io_sdf.md), [XTF](io_xtf.md), [HDF5](io_hdf5.md),
  [NetCDF](io_netcdf.md).
- **Reference:** [API reference](reference/index.md), one page per module, generated from the docstrings.

```{toctree}
:hidden:

Install <install>
Getting started <getting_started>
Examples <examples/index>
Guide <guide>
Formats <formats>
API reference <reference/index>
Development <development>
Pre-release status <prerelease>
```
