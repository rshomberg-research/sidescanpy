# Data format

sidescanpy works in two steps. First a reader converts each raw
recording (EdgeTech JSF, Klein SDF, XTF) into one common data format.
Then every other part of the library runs on that format and only on
it: corrections, gridding, targets, exports and analysis never see the
raw file. Adding a new source format means writing one reader into this
format; everything after it works unchanged.

```text
  raw recordings              data format                       results
 ┌──────────────┐  reader   ┌────────────────────────┐
 │ EdgeTech JSF ├──────────▶│                        │  corrections   corrected .ss.h5
 ├──────────────┤           │   SideScanSurvey       │  gridding      GeoTIFF
 │ Klein SDF    ├──────────▶│   in memory            ├───────────────▶ image chips, COCO / YOLO
 ├──────────────┤           │                        │  targets       targets.json
 │ XTF          ├──────────▶│   .ss.h5 on disk       │  analysis      metrics, plots
 ├──────────────┤           │   (.nc: one frequency) │
 │ a new source ├──────────▶│                        │
 └──────────────┘           └────────────────────────┘
```

The format is the same thing in memory and on disk. In memory it is a
`SideScanSurvey`: one `xarray.Dataset` per frequency with dimensions
`(ping, channel, sample)`, plus the navigation, attitude and sensor
records on their own time axes ([Data model](../core.md) describes the
structure and its conventions). On disk it is an `.ss.h5` file, an HDF5
file of NetCDF-4 groups that any HDF5 or NetCDF tool and plain xarray
can open.

```python
from sidescanpy.io import read_and_convert
from sidescanpy.io.hdf5 import read_survey, write_survey

survey = read_and_convert(["line_001.jsf"], fmt="jsf")   # raw -> data format
write_survey(survey, "line_001.ss.h5")                    # convert once
survey = read_survey("line_001.ss.h5")                    # work from here on
```

Corrections change the data in place and record each step in the
file's `processing_history`, so a corrected `.ss.h5` is still the same
format, one processing level further on.

:::{admonition} A draft, open for comment
:class: warning
The format is at version `0.1` (the `format_version` attribute in every
file). It is shared now so that people who work with sidescan data can
shape it before it is frozen as 1.0: variable names, units, the
grouping of frequencies and sensor streams, metadata conventions, and
what a source needs that the format cannot yet hold. Comments and
proposals are welcome as
[GitHub issues](https://github.com/rshomberg-research/sidescanpy/issues).

Until 1.0 the layout may change in any release, and a file written by
one version may not read in a later one. Keep the raw recordings and
re-convert after an upgrade; the `sidescanpy_version` attribute records
which version wrote a file.
:::

## Pages

| Page | What it covers |
|---|---|
| [The .ss.h5 file](ss_h5.md) | the file layout: groups, variables, attributes, encoding, reading it without sidescanpy |
| [NetCDF, one frequency](netcdf.md) | one frequency of the same format as a standalone `.nc` file |
| [Data model](../core.md) | the in-memory structure and its conventions (sides, signs, units, processing levels) |
| [Source formats](../sources/index.md) | the readers that convert raw recordings into this format |

## Derived outputs

Everything else sidescanpy writes is made from the data format and is a
product, not a format to work from:

| Output | Function | Use |
|---|---|---|
| GeoTIFF | `grids.to_geotiff` | a gridded channel for a GIS (needs the `geo` extra) |
| PNG images with COCO and YOLO labels | `export.ml.build_target_dataset`, `export.ml.write_swath_tiles` | machine-learning datasets |
| `targets.json`, CSV | `targets.write_targets_json`, `write_targets_csv` | target catalogs |
| JSF or XTF subset | `io.jsf.write_subset`, `io.xtf.write_subset` | a run of raw pings copied byte for byte, for test files and bug reports |

```{toctree}
:hidden:

The .ss.h5 file <ss_h5>
NetCDF, one frequency <netcdf>
```
