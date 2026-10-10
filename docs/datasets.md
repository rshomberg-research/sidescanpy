# Targets and datasets

## Target catalogs

`sidescanpy.targets.Target` describes one picked contact: id, position
(lat/lon and/or ping/channel/sample), optional measured `bbox`,
`position_confidence` (`unknown` < `approximate` < `verified`) and
`category`.

| Function | Purpose |
|---|---|
| `io.read_target_catalog(dir)` | read an EdgeTech `*.target` or Klein `Target_*.tgt` catalog from a directory |
| `targets.read_targets_json` / `write_targets_json` (and `_csv`) | the format-agnostic catalog file |
| `targets.correlate.build_position_index(ss_h5_dir)` | cache per-ping positions of every `.ss.h5` |
| `targets.correlate.correlate_target(target, index, max_dist_m)` | the closest ping (geodesic distance) across files and frequencies |
| `targets.correlate.correlate_catalog` / `place_catalog` | the same for a whole catalog, grouped by file, frequency and channel |
| `targets.correlate.resolve_targets_by_catalog_ping(targets, ss_h5_dir)` | Klein: place each target at the ping its catalog names (catalog ping time + number, no clock offset) |
| `targets.correlate.calibrate_clock_offset` | estimate a constant clock offset between a catalog's pick times and the survey (fallback for catalogs without ping fields) |

## Native-resolution windows

`export.windows.native` cuts windows that are square (chips) or a fixed
length (tiles) *in metres* from the native ping x sample grid and resizes
them with linear interpolation, for sonars whose ping spacing is much
coarser than their sample spacing:

| Function | Purpose |
|---|---|
| `native_chip(ds, ping=, channel=, sample=, size_m=, size_px=)` | one side, centred on a target |
| `swath_tiles(ds, tile_m=, px_m=, overlap_m=)` | the whole channel as dual-swath tiles (port mirrored left) |
| `NativeWindow.to_pixel` / `to_native` / `box_to_native` | map between image pixels and (ping, sample, channel), e.g. to turn annotation boxes back into sonar coordinates |

## Building a dataset

`export.ml.build_target_dataset(sources, targets, spec, out_dir)` renders
one image per target per style and writes COCO (`targets.coco.json`) and
YOLO (`labels/`, `data.yaml`) annotations. Everything that shapes the
dataset lives in one `DatasetSpec`, which round-trips to YAML.

| `DatasetSpec` field | Meaning |
|---|---|
| `styles` | `chip` (fixed crop on one channel), `single-swath` (one channel's full swath), `dual-swath` (both channels) |
| `render_level` | `uncorrected`, `db` or `radiometric` (native pixels); `geometric` or `georef` (real-world meters) |
| `size_px`, `swath_size_px` | output size, square |
| `resolution_m` | meters per pixel for geometric/georef; a list renders each value as its own dataset |
| `across_axis` | `ground` (ground range) or `slant` (slant range, no ground projection) for `geometric` |
| `swath_fill` | size swath frames to the across-track extent the data fills (no black margins) |
| `resample_method` | `griddata`, `kdtree` or `binned` for `geometric` |
| `jitter_px`, `n_augment`, `random_flip`, `seed` | augmentation: random offset of the target in the frame, draws per target, random horizontal/vertical flips |
| `box_size`, `box_size_m` | placeholder box for targets with no measured extent (pixels, or meters converted per image) |
| `categories`, `min_confidence` | class names, and which targets get annotations |
| `n_background`, `background_margin_m`, `background_max_nan_frac`, `background_ping_ranges` | background (no-target) images: windows whose footprint plus margin contains no catalog target |
| `val_fraction`, `split_block_pings`, `layout` | train/val split (targets by id hash, backgrounds by file and ping block) and `flat` or `ultralytics` folder layout |
| `bit_depth`, `gamma`, `pct_low`, `pct_high`, `normalization_scope` | how values become 8- or 16-bit pixels |

Each run writes `manifest.json` and returns a `DatasetResult` with
per-style counts and the targets that were skipped and why.

Background images are only as clean as the catalog: an object missing
from the catalog can appear in a background image and will be learned as
background.

## Lower-level pieces

`sidescanpy.export` also exposes the building blocks: window extraction
(`extract_chip`, `extract_full_channel`, `extract_geometric_window*`,
`extract_georef_window`), orientation (`orient_pixel_window`,
`orient_geo_window`), augmentation (`random_flip`, `jitter_offset_m`),
quantization (`to_uint8`, `to_uint16`, `from_uint8`), and
`export.ml.write_coco` / `write_yolo`.

`sidescan-export chip` is a thin command-line wrapper over
`build_target_dataset`; `examples/build_yolo_dataset.py` runs end to end
on a synthetic survey.
