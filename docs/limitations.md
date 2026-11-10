# Limitations

What sidescanpy does not do yet, and what is not settled. Each item says
what to do in the meantime. Format-specific quirks are on the
[JSF](sources/jsf.md), [SDF](sources/sdf.md) and [XTF](sources/xtf.md) pages.

## Radiometric correction

- **Recorder gain is unknown.** Many sonars apply some gain before
  writing the file, and the files do not say how much. EdgeTech JSF
  carries a per-ping gain factor that tracks the signal like an automatic
  gain control; XTF gain codes are read but not interpreted. Some
  recordings behave as if spreading, or both spreading and absorption,
  were already compensated. Measure each survey with
  `analysis.fit_range_decay` and turn off the TVG terms it says are
  already applied ([Getting started](getting_started.md#check-the-gain-before-correcting)
  shows how). Do not assume a sonar model behaves the same on every
  survey.
- **The spreading coefficient is a choice.** The default TVG uses
  30 log R: two-way spherical spreading (40 log R) less the 10 log R
  growth of the insonified seabed patch with range. Which fits real data
  varies by sonar and seabed. Set `spreading_loss_db` per survey after a
  fit; the value used is recorded in each file's history.
- **The decay fit picks a named candidate.** `fit_range_decay` compares
  four fixed loss models and does not fit absorption freely. Its fit
  window can run past the point where the signal reaches the noise
  floor, and the two frequencies of one line can get different answers.
  Check that the corrected data is flat with range before trusting the
  verdict.
- **No angle-dependent correction.** TVG depends on range only. There is
  no beam-pattern or grazing-angle correction, and no mosaic-wide
  normalization between lines.

## Geometry and navigation

- **Flat seabed.** Ground range is `sqrt(slant² - altitude²)`, which
  assumes a flat bottom under the swath. On slopes the across-track
  positions are off by the slope geometry.
- **No layback or lever arms.** Positions are used as recorded. There is
  no towfish layback model and no step for GPS-to-sensor offsets. If the
  file holds the ship's position rather than the towfish's, the
  georeferenced samples are offset by the layback.
- **Heading must be recorded.** There is no fallback to course over
  ground when heading is missing or all zero.
- **One sound velocity per ping.** Zero or missing values fall back to
  1500 m/s (always the case for Klein SDF). There is no water-column
  sound-speed profile or ray bending.
- **Altitude.** `fathometer_reading_m` is not attitude-corrected, and
  the SDF and XTF readers fill `altitude_m` with the same value. For files with no
  altitude, the `bottom_track` step measures it from the first bottom
  return; on sloping or very shallow seabed it reads slightly high,
  because the first return comes from the beam's near edge.

## Gridding

- **One UTM zone per grid.** A grid uses the zone of its bounding-box
  centre, so a survey that crosses a zone boundary puts its far edge in
  the neighbouring zone's frame.
- **No multi-line mosaics.** `grid_channel` grids one channel at a time.
  Stitching several lines, and filling the nadir gap from adjacent
  lines, is not built in.
- **griddata is slow.** The `griddata` resampler used by default for
  geometric dataset chips interpolates most pixels when the grid is finer
  than the ping spacing, is much slower than `kdtree` or `binned`, and
  uses the most memory on large swath windows.

## Files and performance

- **The `.ss.h5` format is not frozen.** See
  [Project status](status.md). Keep the raw files.
- **Re-correcting a georeferenced file is slow.** Adding a step to a
  file that already has per-sample latitude and longitude takes minutes
  per few thousand pings, against seconds for every step in one pass on
  a freshly converted file. Run all the steps you need in one pass from a
  fresh conversion.
- **Formats not read.** See [Source formats](sources/index.md#not-read-yet).

## Reporting a problem

Open an issue with the sonar model, the acquisition software and its
version, and the shortest file that shows the problem. `io.jsf.write_subset`
and `io.xtf.write_subset` cut a run of pings out of a file, byte for byte,
so you can share a few megabytes instead of a whole line.
