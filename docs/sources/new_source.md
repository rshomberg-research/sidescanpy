# Adding a source format

A new source needs one thing: a reader that converts its files into the
[data format](../format/index.md). Corrections, gridding, targets,
exports and analysis then work on it unchanged. This page is the
process the existing readers (JSF, SDF, XTF) followed, as a checklist
for the next one.

## 1. Understand the source before writing code

- **Get the specification** and at least one real file, and note which
  sonar and which acquisition software wrote it. Writers of the same
  format differ (the XTF page lists five), and most quirks belong to a
  writer, not to the format.
- **Look for a file you can share.** An openly licensed recording lets
  the reader be tested in CI and shown in the examples. Note its source,
  licence and checksum.
- **Draft the "Channels and data streams" table first**, as on the
  [JSF](jsf.md), [SDF](sdf.md) and [XTF](xtf.md) pages: every sonar
  channel and sensor stream in the file, where it lands in the data
  format, and how often it arrives. Note where one quantity is recorded
  twice (JSF copies position into every ping and also logs NMEA), and
  which copy is a live sensor and which is held between updates.

## 2. Write the reader and the converter

Code lives in `src/sidescanpy/io/<format>/`, in two layers:

| Module | Does | Does not |
|---|---|---|
| `reader.py` (plus `messages.py`, `pages.py` or `structs.py` for record layouts) | parses the file into typed records with the source's own names and units | convert units, pair channels or build datasets |
| `converter.py` | builds a `SideScanSurvey` from those records | correct anything: samples stay as recorded |
| `subset.py` (recommended) | `write_subset(src, dst, n_pings=, start_ping=)`: copies a run of pings byte for byte | |
| `split.py` (if the range can change mid-file) | splits the pings into runs of constant sample rate | |
| `targets.py` (if the source has target catalogs) | reads them into `Target` objects | |

Then register the extension in `io/dispatch.py` (`detect_format`,
`read_and_convert`, and `read_and_convert_segments` if it splits), which
also makes it available to `sidescan-convert`.

## 3. Meet the data format's contract

The converter's output passes `ds.sidescan.validate()` for every
frequency, and follows these rules:

- **Dimensions.** `(ping, channel, sample)` with `channel` always 2,
  `0` port and `1` starboard, and sample 0 nearest the sonar on both
  sides (reverse a side the source stores far to near). `sample` is an
  absolute index. One dataset per frequency, keyed `hz_<Hz>`, each with
  its own ping axis.
- **Samples.** `backscatter_intensity` as float32, scaled only by what
  the format defines (a weighting exponent, a bit mask), never by a
  gain or TVG. Shorter pings are NaN-padded, with `valid_samples`
  holding each ping's true length.
- **Required per-ping variables:** `latitude`, `longitude` (WGS84
  degrees), `heading_deg`, `roll_deg`, `pitch_deg` (nose up positive),
  `heave_m`, `depth_m`, `altitude_m`, `fathometer_reading_m`,
  `sound_vel_ms`, plus `time` (`datetime64[ns]`) and `ping_number`.
  Units go in the name suffix. A value the source does not record, or
  flags invalid, is NaN, never 0.
- **Required attributes:** `freq_hz_start`, `freq_hz_end`,
  `sample_rate_hz` (one per frequency; split the file if it changes),
  `pulse_length_us` (0 when not recorded).
- **Aux records** on their own `obs` time axis: `navigation`,
  `attitude`, `dvl`, and `sensors[...]` for anything else, each with a
  `source` attribute. A stream the file lacks gets an empty record
  (`make_empty_navigation_record` and friends), not one filled from
  another stream.
- **Position source.** Set `position_source` and
  `position_source_reason`: `towfish` or `ship` only when the file says
  so, otherwise `unknown`. Never guess `ship`: the `layback` step would
  then move positions that may already be the towfish's.
- **Provenance.** Reader-specific attributes carry the format's prefix
  (`xtf_`, `sdf_`), such as the recording program and counts of skipped
  or repaired records.
- **Logging.** DEBUG for routine decisions, WARNING for surprising
  data, and one summary warning per file for skipped record types. No
  `print()`.

If the source records something the data format cannot hold, say so in
a [GitHub issue](https://github.com/rshomberg-research/sidescanpy/issues):
that is exactly the feedback the format needs before 1.0.

## 4. Test it

- **Synthetic files, byte for byte.** `tests/unit/io/<format>/conftest.py`
  builds small files from the specification's record layouts; the tests
  cover parsing, every mapping rule above, the quirks, and a round trip
  through `.ss.h5`. The XTF and JSF test folders are the models.
- **A public fixture.** A cut of a real file made with `write_subset`,
  listed in `tests/data/public/SOURCES.md` with source, licence and
  checksum, and kept outside the repository (never commit survey data).
- **The gates.** `make test` also checks the import direction, the
  logging policy and that every module imports.

## 5. Check it on real data

Passing tests on synthetic files does not show the conversion is right.
On real files:

- Plot a waterfall of each frequency and side: the seabed starts where
  the altitude says, port is on the left, and nothing is mirrored.
- Count real updates in every aux stream with
  `processing.aux_update_events`, and plot the track
  ([What's in a file](../examples/plot_whats_in_a_file.rst),
  [Navigation](../examples/plot_navigation.rst)).
- If the same pings exist in another format (Discover writes JSF and
  XTF), convert both and compare: the samples should be identical
  ([Comparing results across sonars and file formats](../examples/plot_compare_sonars_and_formats.rst)).
- Measure whether the recorder applied a gain with
  `analysis.fit_tvg_models` before choosing corrections
  ([Investigating an unknown survey](../examples/plot_investigate_a_survey.rst)).

## 6. Document it

- A page in `docs/sources/` with the same sections as the others: file
  structure, channels and data streams, mapping to the data format,
  known quirks, reference.
- A row in the [Source formats](index.md) table, with what it was
  tested with.
- If the public file can be shared, register it in
  `sidescanpy.datasets` and list it in
  [Example data](../example_data.md).
- A `CHANGELOG.md` entry.
