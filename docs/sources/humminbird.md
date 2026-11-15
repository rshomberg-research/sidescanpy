# Humminbird

Humminbird fishfinders with side imaging (Helix, Solix, Apex and older
units) record each recording as a small `.DAT` header and a directory of
the same name with one `.SON` file per beam. sidescanpy converts the
side imaging beams into the [data format](../format/index.md).
Humminbird publishes no specification; the layout follows the open
readers PyHum and PINGVerter (PING-Mapper), and the reader has been
checked on Helix recordings: a public USFWS recording (Helix 15, 405 kHz)
and PING-Mapper's test recording (455 kHz).

```python
from sidescanpy.io import read_and_convert
from sidescanpy.io.humminbird import convert, read_humminbird

survey = read_and_convert(["R00013.DAT"], "humminbird")   # the .DAT file
# or, step by step:
recording = read_humminbird("R00013.DAT")                  # finds R00013/B00n.SON
survey = convert(recording)
```

Pass the `.DAT` file; its `.SON` files are found in the directory of the
same name next to it (`R00013.DAT` → `R00013/`). `sidescan-convert
survey` picks up every `.DAT` that has such a directory. `read_dat`,
`read_son` and `read_humminbird` give the header and each beam's frames
with their fields as recorded. `write_subset(src, dst, n_pings=,
start_ping=)` copies the `.DAT` and a run of ping frames of every beam to
a new recording, byte for byte (for making small test files).

## File structure

- **`.DAT`**: one header for the recording. 64 bytes, big-endian, on
  Helix and older units: water type, start time (Unix seconds, UTC),
  first position, recording name, ping count and duration. Solix units
  write 96 or 100 bytes, little-endian, with the same leading fields
  (read as PINGVerter lays them out; not tested on real files). Onix
  units write an XML header, which is not read.
- **`.SON`**: one per beam (`B000.SON` … `B004.SON`), a sequence of ping
  frames. A frame starts with the bytes `C0 DE AB 21`, then tagged
  big-endian fields: a 1-byte tag, then a 1-byte value when the tag's top
  bit is clear (tags 80-87) or a 4-byte value when it is set (tags
  128-160). Tag 160 is the sample count, the byte `0x21` ends the
  header, and that many 8-bit samples follow. The header is 67 bytes on
  older Helix files and 72 or 152 bytes on newer ones; the encoding is
  the same, so tags are decoded one by one, and unknown ones are kept
  as `tag_<n>`.
- **`.IDX`**: per-beam index of frame times and offsets. Not needed; the
  reader walks the frames.

## Channels and data streams

| Stream | Humminbird source | In the data format | Rate |
|---|---|---|---|
| Side imaging, port and starboard | `B002.SON` (port), `B003.SON` (starboard), paired by ping time | one `frequencies/hz_<Hz>` per recorded frequency (tag 146) | every ping |
| Position | tags 130, 131: Mercator easting, northing (m) | per-ping `latitude`, `longitude`; `navigation` record | every ping (held between GPS fixes) |
| Heading, speed | tag 132 (tenths of a degree), tag 133 (dm/s) | `heading_deg`; `navigation` record (`speed_knots`) | every ping |
| Depth below the transducer | tag 135 (dm) | `altitude_m`, `fathometer_reading_m` | every ping |
| 2D sonar, down imaging | `B000.SON`, `B001.SON`, `B004.SON` (one channel each) | not converted; named in a warning and `humminbird_beams_skipped` | |
| Attitude, sound velocity, DVL | none | NaN per-ping values, empty `attitude` and `dvl` records | |

Every frame repeats the latest GPS position and heading, so they change
only when the GPS updates (about once a second); count the real changes
with `processing.aux_update_events` before trusting a track
([Navigation](../examples/plot_navigation.rst)).

## Mapping to the data format

| Survey | Humminbird source |
|---|---|
| frequency key `hz_<Hz>` | tag 146 (Hz) of the side imaging frames |
| `backscatter_intensity` (ping, channel, sample) | `B002` (port) and `B003` (starboard) samples, the 8-bit values as float32; both stored near to far, so neither is reversed |
| `valid_samples` | tag 160; pings hold slightly different counts, padding beyond it is NaN |
| `latitude`, `longitude` | tags 131, 130 on a sphere of radius 6 378 388 m (International 1924), inverted as PyHum and PINGVerter do |
| `heading_deg` | tag 132, low 16 bits / 10 |
| `altitude_m`, `fathometer_reading_m` | tag 135 / 10; 0 is NaN |
| `depth_m`, `roll_deg`, `pitch_deg`, `heave_m`, `sound_vel_ms` | not recorded: NaN (downstream steps use 1500 m/s) |
| `sample_rate_hz` | not recorded: 38 594 Hz by default (see below) |
| `time` | `.DAT` start time + tag 129 (ms since the start) |
| `ping_number` | tag 128 of the port frame |
| `position_source` attribute | `unknown`: the position is the GPS antenna's, and the file does not say where the transducer is |

`pulse_length_us` is not recorded and is 0.0. Dataset attributes
prefixed `humminbird_` record the recording name, sonar id, water type
and SON header size, which beams were skipped, how the sample rate was
chosen, pings recorded on one side only, and counts of truncated frames
and resyncs.

## Conventions and caveats

- **Sample interval is assumed.** The files record neither the range
  nor the sample interval. Humminbird samples at a fixed rate whatever
  the range setting, which only changes how many samples a ping holds.
  Two checks on real Helix files agree: the first bottom return sits at
  about 1.9 cm per sample of recorded depth, and the time between pings
  grows by 26.3 µs per sample (a straight line, r = 1.000, over range
  settings from 1168 to 3058 samples), which is a sample rate of about
  38 050 Hz. `sample_rate_hz` defaults to `HUMMINBIRD_SAMPLE_RATE_HZ`
  = 38 594 Hz, the rate implied by the pixel size PyHum and PING-Mapper
  use (`2 asin(c / (L f)) / π` with `L` = 0.108 m, `f` = 455 kHz, which
  is `c / (2 fs)` with `fs = π L f / 4`). At 1500 m/s that is 1.94 cm
  per sample, within 2 % of both estimates. Pass
  `sample_rate_hz=` to `io.humminbird.convert` to use another value;
  `humminbird_sample_rate_source` records the choice.
- **Port and starboard pairing.** The unit writes both sides of a ping
  at the same millisecond, or the starboard one 1 ms later. Frames within
  `PAIR_TOLERANCE_MS` (5 ms) pair; a ping recorded on one side only keeps
  NaN on the other and is counted in `humminbird_pings_one_side`.
- **Port and starboard orientation.** Both beams store samples nearest
  first. On the recordings checked, the water column and the first
  bottom return are mirror images on the two sides and match the
  recorded depth. Which beam is port comes from the beam numbering in
  PyHum and PINGVerter; it has not been checked against a shoreline.
- **Time.** The `.DAT` start time is taken as UTC (on the public
  recordings it gives daylight local times; read as local time one would
  fall at night).
- **Unit gain.** The samples are 8-bit display values after the unit's
  own gain and contrast settings, which are not recorded. Measure what
  the unit applied with `analysis.fit_tvg_models` before correcting
  ([Limitations](../limitations.md#radiometric-correction)).
- **Recorded depth can be wrong.** The depth comes from the unit's own
  sounder and is not checked. One public recording reads 8 to 18 m for
  its first few thousand pings in a river 1 to 5 m deep; those pings get
  the wrong slant-to-ground conversion. Plot the depth over a waterfall
  first, and use the `bottom_track` step where it does not follow the
  first bottom return.
- **Heading** is the GPS course over ground unless a compass is
  connected to the unit; the file does not say which.
- **Down-looking beams** (2D sonar, down imaging) have one channel each
  and do not fit the port/starboard model; they are skipped.

## Reference

Humminbird publishes no specification. The file layout, the meaning of
the header tags and the position formula used here were worked out by
PyHum and PINGVerter, and sidescanpy's reader follows their description
(its code is its own):

- PyHum: Buscombe, D. (2017), Shallow water benthic imaging and substrate
  characterization using recreational-grade sidescan-sonar,
  *Environmental Modelling & Software* 89, 1-18,
  [doi:10.1016/j.envsoft.2016.12.003](https://doi.org/10.1016/j.envsoft.2016.12.003);
  <https://github.com/dbuscombe-usgs/PyHum> (U.S. public domain).
- PINGVerter and PING-Mapper: Bodine, C. S., et al.,
  <https://github.com/CameronBodine/PINGVerter>,
  <https://github.com/CameronBodine/PINGMapper> (MIT licence). Bodine et
  al. (2022), PING-Mapper: open-source software for automated benthic
  imaging and mapping using recreation-grade sonar, *Earth and Space
  Science* 9, e2022EA002469,
  [doi:10.1029/2022EA002469](https://doi.org/10.1029/2022EA002469).
