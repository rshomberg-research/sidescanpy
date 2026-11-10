# Example data

The [examples](examples/index.rst), the [Theory pages](theory/index.rst) and [Getting started](getting_started.md) run on real, openly licensed recordings that anyone can download.

## The example data

`sidescanpy.datasets.fetch(name)` returns a local path to a recording,
downloading it the first time. Each recording is one file inside a large
public agency archive (8 to 57 GB); `fetch` reads just that file out of
the archive with HTTP range requests, checks its SHA-256 and caches it.
A few archives (PANGAEA) serve each file on its own URL; those are
downloaded directly.

```python
from sidescanpy import datasets

datasets.list_datasets()          # the names fetch() accepts
path = datasets.fetch("usgs_grand_bay_150602205800.xtf")
print(datasets.citation("usgs_grand_bay_150602205800.xtf"))
```

Files are cached in `~/.cache/sidescanpy`, or in `SIDESCANPY_DATA_DIR`
when set. A file already there with the right checksum is used without
going online.

| Name | Size | Recording |
|---|---|---|
| `jncc_wight_barfleur_20130321174522.jsf` | 40 MB | EdgeTech JSF (Discover), 219 + 521 kHz, no recorded altitude |
| `jncc_wight_barfleur_20130321174522H.xtf` | 27 MB | The same pings, high frequency, as XTF (Discover 7.22) |
| `jncc_wight_barfleur_20130321174522L.xtf` | 13 MB | The same pings, low frequency, as XTF (Discover 7.22) |
| `jncc_mid_irish_sea_MIS11_0.xtf` | 41 MB | Benthos SIS-1624, XTF (Isis 7.0.41), 100 + 400 kHz |
| `jncc_mid_irish_sea_MIS13_2.xtf` | 4 MB | The same sonar, a short line |
| `usgs_grand_bay_150602205800.xtf` | 5 MB | Klein, XTF (KleinXlt 4.0 / SonarPro 12), shallow estuary |
| `usgs_grand_bay_150531200100.xtf` | 12 MB | Klein, XTF; the recorded altitude is a lost bottom track |
| `bgs_london_gateway_dynamo_78.xtf`, `_80`, `_85`, `_86`, `_88`, `_90` | 9 to 27 MB each | Klein 400 kHz, XTF (GeoDAS, 2007): six passes over the Dynamo wreck |
| `bgs_london_gateway_eastoaze_31.xtf`, `ernaboldt_13`, `storm_04`, `amethyst_25`, `ash_11` | 9 to 26 MB each | the same survey: one line over each of five more wrecks |
| `bgs_london_gateway_aisha_05.xtf` | 28 MB | Klein 400 kHz, XTF (SonarPro, 2008): the Aisha wreck |

### Published products and their recordings

The Theory pages compare sidescanpy with mosaics that other software
made from the same recordings, so they also fetch those mosaics and
whole lines. These files are larger (about 830 MB in all, 640 MB of it
downloaded for the Grand Bay line and mosaic alone, which unpack to
950 MB); the examples don't use them.

| Name | Size | File |
|---|---|---|
| `usgs_grand_bay_150602202800.xtf` | 236 MB (117 MB download) | Klein 100 kHz, XTF (SonarPro): a whole line under the mosaic |
| `usgs_grand_bay_mosaic_30cmgray.tif` | 715 MB (287 MB download) | The release's mosaic (CleanSweep 3.9), 0.30 m, UTM 16N, 8-bit palette |
| `jncc_solan_bank_SB04_10.xtf` | 79 MB | Benthos SIS-1500, 200 kHz, XTF (Isis); positions are the ship's |
| `jncc_solan_bank_SB4_10.tif` | 16 MB | Isis/TritonMap GeoTIFF of that line, 0.5 m, UTM 30N |
| `jncc_east_rockall_ER_C1_SS_001_6.xtf` | 76 MB | EdgeTech 4200, 120 + 410 kHz, XTF (Isis); UTM 29N navigation |
| `jncc_east_rockall_ER_C1_SS_001_6_CMG.tif` | 57 MB | Isis/TritonMap GeoTIFF of that line, 0.4 m, UTM 28N |
| `awi_ofobs_ps124_20210217004957.jsf` and four more | 51 MB each | EdgeTech 2205 JSF (Discover Bathymetric), towed OFOBS frame, five consecutive files |
| `awi_ofobs_ps124_026_07_ssh_10cm_part1.tiff` | 25 MB | CARIS HIPS and SIPS intensity mosaic of the profile, 0.10 m, float, UPS South |

The East Rockall line stores navigation as UTM zone 29N easting and
northing without saying so: read it with `nav_crs="EPSG:32629"`. The
mosaics' palettes draw strong echoes dark (Grand Bay, Solan Bank, East
Rockall); the palette index itself rises with echo strength. No data is
index 255 in the Grand Bay mosaic and 0 in the two Isis mosaics.

The London Gateway files store navigation as UTM zone 31N easting and
northing without saying so: convert them with `nav_crs="EPSG:32631"`.
They record no heading (use the course over ground), their samples
already carry the recorder's TVG, and their `Weight` fields are garbage,
which the reader ignores with a warning. Their positions jump back by 1
to 3 m at every GPS fix (`processing.smooth_navigation` removes that).
On the GeoDAS lines with no heading the recorder's tow-point offset
points south whatever the course; the reader detects it, uses the ship
positions and records the tow point (12.5 m aft) for the layback step.
The cable layback is not recorded: about 15 m brings opposite-direction
Dynamo passes together.
`datasets.BGS_LONDON_GATEWAY_WRECKS` holds the catalogued centre of each
wreck from the BGS record (UTM 31N meters); a line can show its wreck
tens of meters from that centre along track.

## Sources and licences

- **Wight-Barfleur Reef.** Joint Nature Conservation Committee, *Raw side
  scan sonar data from Wight-Barfleur Reef SCI* (CEND 03/13, 2013),
  <https://www.data.gov.uk/dataset/5298a3dd-af03-4509-b806-ab9f884816ae>.
  Contains public sector information licensed under the
  [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- **Mid-Irish Sea.** Joint Nature Conservation Committee, *Raw sidescan
  sonar data from Submarine structures in the mid-Irish Sea Area of
  Search* (CEND 11/08, 2008),
  <https://www.data.gov.uk/dataset/4a67eeb2-3dce-40b6-96ca-66cc25eeeada>.
  Contains public sector information licensed under the Open Government
  Licence v3.0.
- **London Gateway.** British Geological Survey, National Geoscience
  Data Centre, *Geophysical Survey WESSEX_London_Gateway: Thames Estuary
  (2002, 2007, 2008, 2011)*, item 140325,
  <https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html#item140325>.
  Acquired by the Port of London Authority and Wessex Archaeology.
  Contains data supplied by UKRI; Open Government Licence v3.0.
- **Solan Bank.** Joint Nature Conservation Committee, *Raw sidescan
  sonar data from Solan Bank Area of Search* (CEND 11/08, 2008),
  <https://www.data.gov.uk/dataset/5773e015-c014-416c-aa43-a0397a8fd9e9>,
  and *Processed sidescan mosaics from Solan Bank Area of Search*,
  <https://www.data.gov.uk/dataset/6c1e65a9-d45d-443e-b563-f723ac266ac4>.
  Contains public sector information licensed under the Open Government
  Licence v3.0.
- **East Rockall.** Joint Nature Conservation Committee, *Raw side scan
  data from Anton Dohrn and East Rockall Bank* (MV Franklin, 2009),
  <https://www.data.gov.uk/dataset/09203cee-92dd-486b-8838-a525507765fd>,
  and *Processed side scan data from Anton Dohrn and East Rockall Bank*,
  <https://www.data.gov.uk/dataset/efc2ea5f-c376-4904-b7fb-34cdde65744e>.
  Contains public sector information licensed under the Open Government
  Licence v3.0.
- **AWI OFOBS, PS124.** Hehemann, Laura; Purser, Autun; Boehringer,
  Lilian; Werner, Ellen; Wenzhöfer, Frank (2021): *Side scan sonar and
  swath bathymetry collected along OFOBS profiles PS124_26-7 during
  POLARSTERN cruise PS124, binned* [dataset]. PANGAEA,
  <https://doi.org/10.1594/PANGAEA.939172>, In: Hehemann, L et al.
  (2021): *Side scan sonar and swath bathymetry collected along OFOBS
  profiles during POLARSTERN cruise PS124* [dataset publication series].
  PANGAEA, <https://doi.org/10.1594/PANGAEA.939322>. The mosaic:
  Hehemann, Laura; Boehringer, Lilian; Purser, Autun; Werner, Ellen;
  Wenzhöfer, Frank (2021): *Seafloor bathymetric maps derived from side
  scan data collected along OFOBS profiles during POLARSTERN cruise PS124
  to the southern Weddell Sea* [dataset]. PANGAEA,
  <https://doi.org/10.1594/PANGAEA.939087>. Licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Grand Bay.** Locker, S.D., Forde, A.S., and Smith, C.G., 2018,
  *Subbottom and Sidescan Sonar Data Acquired in 2015 From Grand Bay,
  Mississippi and Alabama*: U.S. Geological Survey data release,
  <https://doi.org/10.5066/P9374DKQ>. Public domain; the U.S. Geological
  Survey is the originator of the data. The Theory pages also use the
  release's sidescan mosaic (CleanSweep 3.9, 0.30 m).

The figures on these pages are derived from those recordings under the
same terms.
