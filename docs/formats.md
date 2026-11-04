# Formats

sidescanpy reads three raw sidescan formats into one data model and
writes it as `.ss.h5`. Everything downstream (corrections, gridding,
export) works on that model, so it does not matter which format a survey
came from.

```python
from sidescanpy.io import read_and_convert, read_and_convert_segments

survey = read_and_convert(["line_001.xtf"], fmt="xtf")        # "jsf", "sdf" or "xtf"
for stem, survey in read_and_convert_segments("line_001.xtf"):  # split at range changes
    ...
```

## Raw formats read

| Format | Reader | Tested with | Notes |
|---|---|---|---|
| EdgeTech JSF (`.jsf`) | [`io.jsf`](io_jsf.md) | EdgeTech towfish recorded by Discover, low and high frequency (public file: 219 + 521 kHz) | sidescan sonar data (message 80), NMEA position, attitude (2020), pressure (2060) and DVL (2080) messages |
| Klein SDF (`.sdf`) | [`io.sdf`](io_sdf.md) | Klein 4900 | sonar pages 3501 and 3502 (the UUV-3500 layout); depth is a voltage and comes out NaN; no sound velocity |
| XTF (`.xtf`) | [`io.xtf`](io_xtf.md) | EdgeTech Discover 7.22; Isis 7.0.41 (Benthos SIS-1624); Isis (EdgeTech 4200); KleinXlt 4.0 / SonarPro 12 (Klein 3000/3900) | sonar packets only; other packet types are counted and skipped; latitude/longitude navigation only (`NavUnits` 3) |

"Tested with" means real files from that source have been converted and
inspected. The public EdgeTech line was written by Discover both as JSF
and as XTF, and the two readers return identical samples
([Three formats, one model](examples/three_formats.md)).
Other sonars and writers of the same formats often work; if one does
not, see [Reporting a problem](limitations.md#reporting-a-problem).

A file whose range setting changes partway through has a different
sample interval before and after the change. `read_and_convert_segments`
and `sidescan-convert survey` split XTF and SDF files at those changes
and convert each run on its own. `read_and_convert` keeps the file whole
and uses one sample rate per frequency, so pings recorded at another
range setting get the wrong ranges (the XTF reader warns when this
happens).

## Target catalogs read

| Catalog | Function | Notes |
|---|---|---|
| EdgeTech Discover `*.target` | `io.read_target_catalog` | positions included |
| Klein `Target_*.tgt` | `io.read_target_catalog` | positions resolved afterwards from the ping each target names (`targets.correlate`) |

Both become a list of format-independent `Target` objects, saved as
`targets.json` or CSV. See [Targets and datasets](datasets.md).

## Files written

| Output | Function | Use |
|---|---|---|
| `.ss.h5` | `io.hdf5.write_survey` | a whole survey, every frequency and aux record ([HDF5](io_hdf5.md)) |
| NetCDF (`.nc`) | `io.netcdf.write_frequency_channel` | one frequency on its own ([NetCDF](io_netcdf.md)) |
| GeoTIFF | `grids.to_geotiff` | a gridded channel for a GIS (needs the `geo` extra) |
| PNG images with COCO and YOLO labels | `export.ml.build_target_dataset` | machine-learning datasets |
| `targets.json`, CSV | `targets.write_targets_json`, `write_targets_csv` | target catalogs |
| JSF or XTF subset | `io.jsf.write_subset`, `io.xtf.write_subset` | a run of pings copied byte for byte, for test files and bug reports |

## Not supported

These are not read: Teledyne/Reson `.s7k`, HYPACK `.hsx`/`.hs2`, Marine
Sonic `.mst`/`.sds`, Lowrance `.sl2`/`.sl3`, Humminbird, Imagenex and
interferometric or synthetic-aperture products. Within the supported
formats:

- XTF bathymetry, subbottom and other non-sonar packets are skipped.
- XTF files with projected navigation (`NavUnits` 0) raise
  `NotImplementedError`.
- Klein System 3000 pages (versions 3000 and 3001) are recognized but
  not converted.
- JSF messages other than sonar data, NMEA, attitude, pressure and DVL
  (system information, situation, cable counter and others) are parsed
  or skipped but not converted.

Many vendor tools export XTF, which is often the quickest way in for a
format sidescanpy does not read.
