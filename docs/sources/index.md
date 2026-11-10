# Source formats

A source format is what a sonar or its acquisition software records.
sidescanpy reads each one only to convert it into the
[data format](../format/index.md); after that, nothing depends on which
source a survey came from. Each reader maps the source's channels and
data streams onto the same structure: one dataset per frequency with
port and starboard, per-ping navigation and attitude, and the aux
records on their own time axes.

```python
from sidescanpy.io import read_and_convert, read_and_convert_segments

survey = read_and_convert(["line_001.xtf"], fmt="xtf")        # "jsf", "sdf" or "xtf"
for stem, survey in read_and_convert_segments("line_001.xtf"):  # split at range changes
    ...
```

## Sources read

| Source | Reader | Tested with | Data streams converted |
|---|---|---|---|
| [EdgeTech JSF](jsf.md) (`.jsf`) | `io.jsf` | EdgeTech towfish recorded by Discover, low and high frequency (public file: 219 + 521 kHz) | sonar (message 80) with its embedded per-ping values, NMEA position, attitude (2020), pressure (2060), DVL (2080) |
| [Klein SDF](sdf.md) (`.sdf`) | `io.sdf` | Klein 4900 | sonar pages 3501 and 3502 (UUV-3500 layout), each carrying its own navigation and attitude |
| [XTF](xtf.md) (`.xtf`) | `io.xtf` | EdgeTech Discover 7.22; Isis 7.0.41 (Benthos SIS-1624); Isis (EdgeTech 4200); KleinXlt 4.0 / SonarPro 12 (Klein 3000/3900); GeoDAS (Klein 3000) | sonar packets, each carrying its own navigation and attitude |

"Tested with" means real files from that source have been converted and
inspected. The public EdgeTech line was written by Discover both as JSF
and as XTF, and the two readers return identical samples
([Comparing results across sonars and file formats](../examples/plot_compare_sonars_and_formats.rst)).
Other sonars and writers of the same formats often work; if one does
not, see [Reporting a problem](../limitations.md#reporting-a-problem).

Every source page has the same sections: the file structure, the
channels and data streams it carries and where each lands in the data
format, the field-by-field mapping, and the known quirks of the devices
seen so far.

## Range changes

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

Both become a list of source-independent `Target` objects, saved as
`targets.json` or CSV. See [Targets and datasets](../datasets.md).

## Not read yet

Teledyne/Reson `.s7k`, HYPACK `.hsx`/`.hs2`, Marine Sonic `.mst`/`.sds`,
Lowrance `.sl2`/`.sl3`, Humminbird, Imagenex and interferometric or
synthetic-aperture products. Within the sources read:

- XTF bathymetry, subbottom and other non-sonar packets are skipped.
- Klein System 3000 SDF pages (versions 3000 and 3001) are recognized
  but not converted.
- JSF messages other than sonar data, NMEA, attitude, pressure and DVL
  (system information, situation, cable counter and others) are parsed
  or skipped but not converted.

Many vendor tools export XTF, which is often the quickest way in for a
source sidescanpy does not read. To add a reader, see
[Adding a source format](new_source.md).

```{toctree}
:hidden:

EdgeTech JSF <jsf>
Klein SDF <sdf>
XTF <xtf>
Adding a source format <new_source>
```
