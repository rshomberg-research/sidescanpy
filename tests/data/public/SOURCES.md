# Public test fixtures

Small cuts of openly licensed sidescan recordings, used by
`tests/unit/io/test_public_fixtures.py` and `tests/unit/io/jsf/test_subset.py`.
**The files are not in the repository** (data stays out of the code
repo; a public archive with a DOI is planned). Put them in one directory
and set `SIDESCANPY_PUBLIC_DATA` to it; without it those tests skip. They
can be rebuilt from the public archives below. Each file was made with
`sidescanpy.io.jsf.write_subset` or `sidescanpy.io.xtf.write_subset`,
which copy the file header and a run of pings unchanged, byte for byte.
The ping range is part of each file name (`p<first>-<last>`, zero-based
in the source file).

Every file has to carry a licence that allows redistribution. Never add
one to the set without its source, licence and checksum below; a test
fails on an unlisted or changed file.

| File | Source file | Format, writer | Sonar | Licence |
|---|---|---|---|---|
| `jncc_wight_barfleur_20130321174522_p100-107.jsf` | `BoxB2/20130321174522.jsf` | JSF, EdgeTech Discover | EdgeTech, 219 + 521 kHz | OGL v3 |
| `jncc_wight_barfleur_20130321174522H_p100-107.xtf` | `BoxB2/20130321174522H.xtf` | XTF, Discover 7.22 (same pings as the JSF) | EdgeTech, high frequency | OGL v3 |
| `jncc_wight_barfleur_20130321174522L_p100-107.xtf` | `BoxB2/20130321174522L.xtf` | XTF, Discover 7.22 (same pings as the JSF) | EdgeTech, low frequency | OGL v3 |
| `jncc_mid_irish_sea_MIS13_2_p0-29.xtf` | `MIS13_2.XTF` | XTF, Isis 7.0.41 | Benthos SIS-1624, 100 + 400 kHz | OGL v3 |
| `usgs_grand_bay_15CCT03_SSS_153_150602205800_p0-59.xtf` | `15CCT03_SSS_153_150602205800.xtf` | XTF, KleinXlt 4.0 / SonarPro 12 | Klein (file header: "Klein 3000"; the data release says Klein 3900) | U.S. public domain |

## Sources and attribution

**Wight-Barfleur Reef.** Joint Nature Conservation Committee, *Raw side
scan sonar data from Wight-Barfleur Reef SCI* (CEND 03/13, 2013),
<https://www.data.gov.uk/dataset/5298a3dd-af03-4509-b806-ab9f884816ae>,
archive `JNCC2013_03WightBReefRawAcousticss_raw.zip`. Contains public
sector information licensed under the Open Government Licence v3.0
(<https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>).

**Mid-Irish Sea.** Joint Nature Conservation Committee, *Raw sidescan
sonar data from Submarine structures in the mid-Irish Sea Area of Search*
(CEND 11/08, 2008),
<https://www.data.gov.uk/dataset/4a67eeb2-3dce-40b6-96ca-66cc25eeeada>,
archive `JNCC2008_05IrishSolanGeophysSidescanMidIrishRawData.zip`.
Contains public sector information licensed under the Open Government
Licence v3.0.

**Grand Bay.** Locker, S.D., Forde, A.S., and Smith, C.G., 2018,
*Subbottom and Sidescan Sonar Data Acquired in 2015 From Grand Bay,
Mississippi and Alabama*: U.S. Geological Survey data release,
<https://doi.org/10.5066/P9374DKQ>, archive `2015-315-FA_xtf.zip`. Public
domain; the U.S. Geological Survey asks to be acknowledged as originator
of the data.

## Checksums (SHA-256)

```
fbfe2a3f5581d0a60051de8111957d667a9167b30d98a2f0284a5a240225b7c7  jncc_wight_barfleur_20130321174522_p100-107.jsf
8d5809ec40ff97ec8bd7f307b3516271f90f78331bc590c66500156fdfa83813  jncc_wight_barfleur_20130321174522H_p100-107.xtf
f4495e81d30ccdde0a601efe4bee80ab53f69cda7b3c8f6b5f5ea4aa6bce1f27  jncc_wight_barfleur_20130321174522L_p100-107.xtf
a51f6d1f336bb753b13d8f5097b034d5458dbdae545bb0b0992d8459d3b59d39  jncc_mid_irish_sea_MIS13_2_p0-29.xtf
a4262b37898fbb589f7e77d95a7ff91e7de9bf20e793926b02bb639e44b7fa5b  usgs_grand_bay_15CCT03_SSS_153_150602205800_p0-59.xtf
```
