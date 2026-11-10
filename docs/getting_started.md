# Getting started

This page takes one public sidescan line from the raw file to a GeoTIFF:
download, convert, inspect, check the gain, correct, grid. It needs the
`viz` and `geo` extras ([Install](install.md)) and about 250 MB of disk,
and runs in under a minute once the file is downloaded.

The line is from a 2008 UK JNCC survey in the Irish Sea: a Benthos
SIS-1624 towfish at 100 and 400 kHz, recorded as XTF by Isis. Any JSF,
SDF or XTF file works the same way; only the format string changes.

## Download a recording

```python
from sidescanpy import datasets

path = datasets.fetch("jncc_mid_irish_sea_MIS11_0.xtf")
print(datasets.citation("jncc_mid_irish_sea_MIS11_0.xtf"))
```

`fetch` downloads the 41 MB file once, checks its SHA-256 and caches it
in `~/.cache/sidescanpy` (or `SIDESCANPY_DATA_DIR`). Later calls return
the cached path without going online. `datasets.list_datasets()` names
every recording it can fetch; [Examples](example_data.md#the-example-data)
lists them with their sources and licences.

## Convert to the data model

```python
from sidescanpy.io import read_and_convert
from sidescanpy.io.hdf5 import write_survey

survey = read_and_convert([path], fmt="xtf")
write_survey(survey, "MIS11_0.ss.h5")

for key in sorted(survey.frequencies):
    print(key, dict(survey.get_frequency(key).dataset.sizes))
```

```
hz_100000 {'ping': 1234, 'channel': 2, 'sample': 4096}
hz_400000 {'ping': 1234, 'channel': 2, 'sample': 4096}
```

`read_and_convert` returns a `SideScanSurvey`: one `xarray.Dataset` per
frequency, keyed `hz_<frequency>`, plus the navigation, attitude, DVL and
sensor records the file carried. Backscatter has dimensions
`(ping, channel, sample)`, with channel 0 port and 1 starboard. Per-ping
values (position, heading, attitude, altitude, sound velocity) are
variables along `ping`. This is sidescanpy's [data format](format/index.md):
every reader converts into it and everything else runs on it.
[Data model](core.md) describes the structure and
[The .ss.h5 file](format/ss_h5.md) the file.

The `.ss.h5` file is a derived product. Keep the raw XTF: until 1.0 the
file format may change and you may need to convert again.

## Look at it

```python
import matplotlib.pyplot as plt
from sidescanpy.processing import compute_geometric_ranges
from sidescanpy.viz import plot_waterfall

ds = survey.get_frequency("hz_400000").dataset
print(f"altitude {float(ds.altitude_m.median()):.1f} m")

fig, ax = plt.subplots(figsize=(10, 6))
plot_waterfall(compute_geometric_ranges(ds), x_axis="slant_range", gamma=0.5, ax=ax)
fig.savefig("MIS11_0_raw.png")
```

```
altitude 31.8 m
```

The waterfall shows raw amplitude against slant range, port mirrored to
the left. The dark band in the middle is the water column: nothing comes
back until the pulse reaches the seabed, about 32 m below the towfish.
`compute_geometric_ranges` only adds range variables here; it does not
change the backscatter. [Read a file](examples/plot_read_a_file.rst) shows a
waterfall like this one.

## Check the gain before correcting

Time-varying gain (TVG) adds back the spreading and absorption losses
that make far range darker than near range. Some recorders apply part or
all of it before writing the file, and correcting again washes out the
far range. `analysis.fit_tvg_models` fits the TVG the data still needs
(no TVG, absorption only, or 20, 30 or 40 log R with or without
absorption) to the median level against range:

```python
from sidescanpy.analysis import fit_tvg_models

for key in sorted(survey.frequencies):
    ds = compute_geometric_ranges(survey.get_frequency(key).dataset)
    fit = fit_tvg_models(ds)
    print(key, fit.best, fit.tvg_status)
```

```
hz_100000 flat full
hz_400000 20 log R absorption_only
```

The best model is the TVG to apply (`fit.tvg_params`), and the status
names what the recorder has already removed: `none` (raw losses, apply
full TVG), `spreading_only`, `absorption_only`, or `full` (apply none).
Here the 100 kHz level hardly falls with range (the recorder levelled
it, so apply no TVG) and the 400 kHz level falls like 20 log R. Treat it
as a hypothesis: `analysis.check_tvg` applies the model and measures how
flat the corrected data is. [Processing](processing.md#tvg)
explains the TVG terms and [Limitations](limitations.md) what is not
settled about recorder gain.

## Correct

Each correction is a function that takes the survey and returns a
corrected copy, recording itself, with its parameters, in each
frequency's `processing_history`:

```python
from sidescanpy.core.history import applied_steps
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
)

survey = interpolate_aux_onto_pings(survey)   # navigation and attitude onto the pings
survey = apply_tvg(survey)                    # 30 log R spreading + Francois-Garrison absorption
survey = compute_geometric_ranges(survey)     # slant and ground range
survey = compute_georeferencing(survey)       # latitude and longitude per sample
print(applied_steps(survey.get_frequency("hz_400000").dataset))
write_survey(survey, "MIS11_0.corrected.ss.h5")
```

```
['interpolate_aux', 'tvg', 'geometric', 'georef']
```

Leave a call out to skip that correction. To follow the gain check above
instead, start again from the converted file and turn off the terms the
recorder already applied, per frequency (a term set to 0 is not
applied):

```python
from sidescanpy.io.hdf5 import read_survey

survey = interpolate_aux_onto_pings(read_survey("MIS11_0.ss.h5"))
tvg = {
    "hz_100000": {"spreading_loss_db": 0.0},
    "hz_400000": {"spreading_loss_db": 0.0, "absorption_db_per_m": 0.0},
}
for freq_key, terms in tvg.items():
    ds = survey.get_frequency(freq_key).dataset
    survey.add_frequency(freq_key, apply_tvg(ds, **terms))
survey = compute_georeferencing(compute_geometric_ranges(survey))
```

`apply_tvg` refuses to run twice on the same data, so gains never stack;
`invert` undoes a recorded TVG if you want to try another recipe.
[Correct a line](examples/plot_correct_a_line.rst) adds
`altitude_from_bottom_track` for files that record no altitude, and
[Processing](processing.md#running-steps-by-name) shows the same
corrections run from a list of step names, for config files and large
files.

## Grid and write a GeoTIFF

```python
from sidescanpy.grids import grid_channel, to_geotiff

raster = grid_channel(survey.get_frequency("hz_400000").dataset, resolution_m=1.0)
print(raster)
to_geotiff(raster, "MIS11_0_400khz.tif")
```

```
SideScanGriddedRaster(channel_name='400 kHz', shape=(1048, 544), coverage=55.9%, crs=32630)
```

`grid_channel` picks the UTM zone of the data (zone 30N here, EPSG:32630)
and averages the samples that fall near each 1 m cell. Port and starboard
samples are gridded as separate observations, never averaged first. The
GeoTIFF opens in QGIS or any GIS. [Gridding](grids.md) covers the
methods and search radius, and [Grid to a map](examples/plot_grid_to_a_map.rst)
fills the gaps between pings with inverse-distance weighting.

## The same from the command line

```bash
sidescan-convert survey --help
sidescan-correct --help
sidescan-export grid --help
```

The CLI runs the same library calls on whole folders of files; see
[Command line](cli.md).

## Next

- [Building a workflow](workflow.md): one script per step and one config
  per survey, for real surveys.
- [Data format](format/index.md): the format every reader converts into.
- [Source formats](sources/index.md): what each reader supports.
- [API reference](reference/index.md): every function and its parameters.

Data: JNCC mid-Irish Sea, Open Government Licence v3.0 (see
[the example data](example_data.md#sources-and-licences)).
