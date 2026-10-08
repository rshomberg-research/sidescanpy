# sidescanpy

Read, correct, grid and export sidescan sonar data.

sidescanpy converts EdgeTech JSF, Klein SDF and XTF surveys into one
`xarray`-based model stored as `.ss.h5`, then corrects it (TVG, ground
range, georeferencing), slices it, grids it to rasters and GeoTIFF, and
builds COCO/YOLO training datasets from target catalogs. Small analysis
functions measure a dataset's units, nadir, usable range and TVG state.

- **Start:** install and the quickstart are in the
  [README](https://github.com/rshomberg-uri/sidescanpy#readme).
- **Guide:** [Building a workflow](workflow.md), [Data model](core.md), [Processing](processing.md),
  [Gridding](grids.md), [Targets and datasets](datasets.md),
  [Analysis](analysis.md), [Visualization](viz.md),
  [Command line](cli.md), [Logging](logging.md).
- **Formats:** [JSF](io_jsf.md), [SDF](io_sdf.md), [XTF](io_xtf.md), [HDF5](io_hdf5.md),
  [NetCDF](io_netcdf.md).
- **Reference:** [API reference](reference/index.md), one page per module, generated from the docstrings.
