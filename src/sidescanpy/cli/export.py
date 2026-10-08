"""
``sidescan-export``: waterfalls, gridded rasters, survey tracks, and
configurable annotated-dataset exports (chips / full-channel windows /
correction-step progression grids) from already-corrected ``.ss.h5``
files.

Export subcommands for track maps, waterfalls, grids, and target-centered
chips (``chip``/``survey-window``/``progression``).

Subcommands:

- ``waterfall`` — sample/ping/time-indexed (``--range-mode slant``, the
  old ``05_waterfalls.py``) or ground-range/along-track-indexed
  (``--range-mode ground``, the old ``07_ground_waterfalls.py``,
  requires geometric ranges already computed) waterfall PNGs, segmented
  by ping count.
- ``grid`` — gridded-raster PNGs on an auto-selected UTM grid, segmented
  by ping count (the old ``09_grid.py``, requires georeferencing).
- ``track`` — full-survey nadir track + scan-line plot per frequency,
  with an optional target-catalog overlay via ``--targets-json`` (merges
  the old ``03_track_maps.py``/``12_track_with_targets.py``; requires
  georeferencing for the scan-line rungs).
- ``chip`` — real, jittered target-positive chips, position-correlated
  to the real survey array by matching each target's recorded lat/lon
  (not by trusting a target catalog's own embedded ping index, which
  can be private to that catalog file — see this module's own git
  history for the JSF bug that motivated this). Exports up to three
  parallel COCO+YOLO datasets under ``output/<style>/`` (``--styles``):
  ``chip`` (tight single-channel crop), ``single-swath`` (one full
  channel, square), ``dual-swath`` (both channels combined, square).
  ``--render-level`` selects what every style renders from: ``uncorrected``/
  ``radiometric`` (pixel/sample space) or ``geometric``/``georef`` (real
  meters-square, via the same window extractors ``progression`` uses —
  requires that correction already run via ``sidescan-correct``).
  ``--with-context`` (independent of styles/render-level, always
  radiometric) saves ``context/context_<id>.png`` — the full ping
  neighborhood, both channels combined, with a real box drawn at the
  target's position.
- ``survey-window`` — full-channel / geometric / georef windows at
  evenly-spaced ping locations, uncorrelated to any target catalog, so
  every window is honestly a no-target negative.
- ``progression`` — a 5-step (uncorrected / dB / TVG / geometric / georef) x
  N-target comparison grid, for visual QA of the correction pipeline.

Usage:
    sidescan-export waterfall --ss-h5-dir data/active/unknown/ss_h5/ \\
        --output-dir data/active/unknown/exports/waterfall_radiometric/

    sidescan-export waterfall --range-mode ground \\
        --ss-h5-dir data/active/unknown/ss_h5/ \\
        --output-dir data/active/unknown/exports/waterfall_ground/

    sidescan-export grid --ss-h5-dir data/active/unknown/ss_h5/ \\
        --output-dir data/active/unknown/exports/gridded/

    sidescan-export track --ss-h5-dir data/active/unknown/ss_h5/ \\
        --output-dir data/active/unknown/exports/track/ \\
        --targets-json data/active/unknown/Targets/targets.json

    sidescan-export chip \\
        --targets-json data/active/unknown/Targets/targets.json \\
        --ss-h5-dir data/active/unknown/ss_h5/ \\
        --output data/active/unknown/Targets/dataset_chips/ \\
        --size-px 400 --gamma 2.5 --jitter-px 40

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from PIL import Image

from sidescanpy.cli._common import (
    add_logging_args,
    require_matplotlib,
    resolve_freq_keys,
    setup_logging_from_args,
    uncorrected_alias,
)
from sidescanpy.cli._config import apply_config_to_subparsers, find_config_path, load_config
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.core.units import linear_to_db
from sidescanpy.export import (
    ReferenceStats,
    WindowExtent,
    compute_reference_stats,
    extract_chip,
    extract_full_channel,
    extract_geometric_window,
    extract_geometric_window_binned,
    extract_geometric_window_griddata,
    extract_georef_window,
    locate_point_geometric,
    locate_point_georef,
    rescale_target_position,
    resize_to,
    targets_within_window,
    to_uint8,
    to_uint16,
)
from sidescanpy.export.ml import CocoImageEntry, write_coco, write_yolo
from sidescanpy.grids import (
    ResamplingMethod,
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.grids.definition import utm_crs_for_lonlat
from sidescanpy.io.hdf5 import concat_survey_frequency, read_survey
from sidescanpy.processing.bottom import mask_nadir_gap
from sidescanpy.processing.inversion import UNCORRECTED_VAR, uncorrected_backscatter
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.processing.radiometric import TVGParams, apply_tvg_to_amplitude
from sidescanpy.processing.track import build_survey_track
from sidescanpy.targets.correlate import (
    build_position_index,
    correlate_target,
    project_target_to_frequency,
)
from sidescanpy.targets.io import read_targets_json

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"unknown": 0, "approximate": 1, "verified": 2}
# Was 2000 -- not a multiple of HDF5_CHUNK_PING (the on-disk chunk size
# write_survey() actually stores, 256). run_grid's own --segment-size
# (this same default) feeds directly into read_survey(chunks=...), so a
# misaligned value made xr.open_dataset split real HDF5 chunks to match,
# degrading read performance and printing a UserWarning on every call
# (found from a real user report, 2026-09-28). 2048 = 8x256, close to
# the original round-number default, never splits a real chunk.
_DEFAULT_SEGMENT_SIZE = HDF5_CHUNK_PING * 8


# ============================================================
# waterfall (replaces 05_waterfalls.py / 07_ground_waterfalls.py)
# ============================================================

def _sample_spacing_m(ds, local_ping: int, channel_idx: int) -> float:
    """Mean across-track meters-per-sample for one ping/channel, used to
    convert a pixel-space box size into a real-world width in ground
    mode. Averaged over every finite (non-water-column) sample on the
    ping rather than a local two-point difference, so it's stable near
    the water-column boundary or the window's own edges."""
    at = ds["across_track_m"].isel(ping=local_ping, channel=channel_idx).values
    finite = at[np.isfinite(at)]
    if finite.size < 2:
        return 1.0
    return float(np.abs(np.diff(finite)).mean())


def _ping_spacing_m(along_track) -> float:
    """Median along-track meters-per-ping across the whole plotted
    window, used to convert a pixel-space box size into a real-world
    height in ground mode. Median, not mean, so it isn't skewed by a
    vessel-turn's occasional large ping-to-ping jump."""
    if along_track is None or along_track.size < 2:
        return 1.0
    diffs = np.abs(np.diff(along_track))
    return float(np.median(diffs)) if diffs.size else 1.0


def _target_waterfall_xy(target, ds, ping_idx: int, ping_offset: int, x_axis: str, y_axis: str, along_track=None) -> tuple[float, float, float, float] | None:
    """Where ``target`` lands in ``plot_waterfall``'s own *data*
    coordinates (pcolormesh, not raw pixel-array indices), for whichever
    of the two supported axis pairings is active, plus the (x, y) scale
    factors (real axis-units per pixel-space unit) a caller needs to
    convert a shared ``--box-size`` -- always a sample/ping *pixel*
    count -- into a comparably-sized box on whichever axes are active.

    ``x_axis="sample"``/``y_axis="ping"`` (slant mode): x = -sample
    (port, channel 0) / +sample (starboard, channel 1); y = the *global*
    ping index directly (``_compute_y_full`` already bakes
    ``ping_offset`` into the plotted y-axis, so ``ping_idx`` needs no
    re-basing here). Scale factors are both 1.0 -- the axes already
    *are* pixel-space, so ``box_size`` needs no conversion.

    ``x_axis="across_track"``/``y_axis="along_track"`` (ground mode,
    forced by ``--range-mode ground`` via ``resolve_waterfall_axes``):
    unlike the ping axis, ``along_track`` resets to 0 at ``ds``'s own
    first ping (see ``_along_track_distance``'s docstring) -- so the
    target's ping must be re-based to ``ds``'s local index
    (``ping_idx - ping_offset``) before it can index either
    ``across_track_m`` or the caller-supplied ``along_track`` array.
    x = the target's own real across-track position (already signed by
    channel), read directly off ``ds`` at its own ping/channel/sample --
    not derivable from ``target.sample`` alone the way slant mode's
    sample-index x is, since across-track distance is altitude/geometry
    dependent, not a fixed function of sample index. Scale factors are
    real meters-per-sample/meters-per-ping (see ``_sample_spacing_m``/
    ``_ping_spacing_m``) -- **without** these, a shared ``box_size`` of
    e.g. 50 (calibrated to look tight in slant mode's sample/ping pixel
    units) was being applied as a literal 50 *meters*, easily covering
    most of a swath only tens of meters wide.

    Returns ``None`` if the target's ping falls outside ``ds`` (already
    windowed) or (ground mode only) it has no across-track value at its
    own position (e.g. still in the water column).
    """
    if target.sample is None:
        return None
    channel_idx = target.channel if target.channel is not None else 0

    if x_axis == "sample" and y_axis == "ping":
        x_center = -target.sample if channel_idx == 0 else target.sample
        return float(x_center), float(ping_idx), 1.0, 1.0

    if x_axis == "across_track" and y_axis == "along_track":
        local_ping = ping_idx - ping_offset
        if not (0 <= local_ping < ds.sizes["ping"]) or not (0 <= target.sample < ds.sizes["sample"]):
            return None
        x_center = float(ds["across_track_m"].isel(ping=local_ping, channel=channel_idx, sample=target.sample).values)
        if not np.isfinite(x_center):
            return None
        y_center = float(along_track[local_ping]) if along_track is not None else float(local_ping)
        x_scale = _sample_spacing_m(ds, local_ping, channel_idx)
        y_scale = _ping_spacing_m(along_track)
        return x_center, y_center, x_scale, y_scale

    return None


def _waterfall_box_rects(ds, boxes, ping_offset, x_axis, y_axis, box_size) -> list[tuple[str, float, float, float, float]]:
    """``(label, x0, y0, width, height)`` in the waterfall's axis units for
    each ``(target, ping_idx)`` in ``boxes`` that can be placed on these
    axes -- see ``_target_waterfall_xy``."""
    # Computed once per window, not per box -- along_track_distance_m
    # does a real pyproj.Geod.inv call over every ping in ds.
    along_track = along_track_distance_m(ds) if boxes and y_axis == "along_track" else None
    rects = []
    for target, ping_idx in boxes:
        xy = _target_waterfall_xy(target, ds, ping_idx, ping_offset, x_axis, y_axis, along_track)
        if xy is None:
            continue
        x_center, y_center, x_scale, y_scale = xy
        half_x = (box_size / 2.0) * x_scale
        half_y = (box_size / 2.0) * y_scale
        rects.append((str(target.target_id), x_center - half_x, y_center - half_y, 2 * half_x, 2 * half_y))
    return rects


def _export_waterfall_png(ds, output_path: Path, *, x_axis, y_axis, ping_offset, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, dpi, boxes=None, box_size=40.0, value_var="backscatter_intensity"):
    """``boxes``, if given: list of ``(target, ping_idx)`` (global ping
    index within whatever ``ds`` this call was handed) to draw a red box
    + target_id label for, on top of the rendered waterfall. Supports
    both the default sample/ping axes and ground mode's across_track/
    along_track axes -- see ``_target_waterfall_xy``."""
    from sidescanpy.cli import _figures

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _figures.save_waterfall_png(
        ds, output_path, x_axis=x_axis, y_axis=y_axis, ping_offset=ping_offset,
        cmap=cmap, vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
        gamma=gamma, dpi=dpi, value_var=value_var,
        box_rects=_waterfall_box_rects(ds, boxes or [], ping_offset, x_axis, y_axis, box_size),
    )


def run_waterfall(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _figures
    from sidescanpy.viz.waterfall import resolve_waterfall_axes

    if args.range_mode == "ground":
        # Forces the distance-axis branch regardless of --x-axis/--y-axis
        # -- geometric ranges have no visible effect on a sample/ping-
        # indexed plot (only color values change under TVG, not pixel
        # geometry), matching the old 07_ground_waterfalls.py.
        x_axis, y_axis = resolve_waterfall_axes(frozenset({"geometric"}), args.x_axis, args.y_axis)
    else:
        x_axis, y_axis = args.x_axis, args.y_axis

    targets = []
    index: dict[Path, dict] = {}
    if args.targets_json:
        supported_axes = (x_axis == "sample" and y_axis == "ping") or (x_axis == "across_track" and y_axis == "along_track")
        if not supported_axes:
            logger.error(
                "--targets-json only supports the default sample/ping pixel axes or "
                f"ground mode's across_track/along_track axes -- got x_axis={x_axis!r}, "
                f"y_axis={y_axis!r}."
            )
            return 1
        targets = read_targets_json(args.targets_json)
        index = build_position_index(args.ss_h5_dir)

    files = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1
    logger.info(f"Found {len(files)} .ss.h5 file(s) in {args.ss_h5_dir}")

    stride = args.window_pings - args.overlap_pings
    if stride <= 0:
        logger.error(f"--overlap-pings ({args.overlap_pings}) must be smaller than --window-pings ({args.window_pings})")
        return 1

    value_var = UNCORRECTED_VAR if args.value_var == "uncorrected" else "backscatter_intensity"

    n_written = 0
    # Correlate every target once (position-based, format-agnostic), then
    # re-derive its position on *each* frequency below -- its own native
    # mark where it was actually correlated, or a physical projection
    # (same ping, slant-range-equivalent sample) onto every other
    # frequency of that same file, so a target shows up on every
    # rendered frequency, not just the one it happened to be picked on.
    matched = []
    for t in targets:
        m = correlate_target(t, index, args.max_match_dist_m)
        if m is not None:
            matched.append((t,) + m)  # (t, path, native_freq_key, local_ping, dist_m)
    if targets:
        logger.info(f"Overlaying {len(matched)}/{len(targets)} target(s) matched to a real ping")

    ds_first = read_survey(files[0], chunks={"ping": 512})
    freq_keys = resolve_freq_keys(ds_first.frequencies, args.frequency)

    for freq_key in freq_keys:
        ds_all, file_start_ping = concat_survey_frequency(files, freq_key, chunks={"ping": 512})
        if ds_all is None:
            logger.warning(f"{freq_key}: no file has this frequency, skipping")
            continue
        n_pings_total = ds_all.sizes["ping"]
        if value_var == UNCORRECTED_VAR:
            ds_all = _with_uncorrected_var(ds_all)

        boxes_by_ping: dict[int, list] = defaultdict(list)
        for t, path, native_freq_key, local_ping, _dist_m in matched:
            if path not in file_start_ping:
                continue  # this file has no freq_key channel at all
            if native_freq_key == freq_key:
                projected = t
            else:
                src = index[path][native_freq_key]["sample_interval_m"]
                dst = index[path][freq_key]["sample_interval_m"]
                n_samples_dst = index[path][freq_key]["n_samples"]
                projected = project_target_to_frequency(t, source_mps=src, target_mps=dst, n_samples=n_samples_dst)
                if projected is None:
                    continue  # genuinely out of this frequency's recorded range
            boxes_by_ping[file_start_ping[path] + local_ping].append(projected)

        if args.overview_rows:
            overview_stride = max(1, n_pings_total // args.overview_rows)
            breaks = sorted(p for p in file_start_ping.values() if p > 0)
            out_path = args.output_dir / freq_key / "overview.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _figures.save_survey_overview_png(
                ds_all, out_path, dpi=args.dpi,
                ping_stride=overview_stride, break_ping_indices=breaks,
                x_axis=x_axis, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma,
                value_var=value_var,
            )
            n_written += 1
            logger.info(f"{freq_key}: wrote overview.png ({n_pings_total} pings, 1:{overview_stride} compressed)")

        n_windows = 0
        for start in range(0, n_pings_total, stride):
            stop = min(start + args.window_pings, n_pings_total)
            ds_window = ds_all.isel(ping=slice(start, stop)).load()
            out_path = args.output_dir / freq_key / f"segment_{n_windows + 1:03d}.png"
            window_boxes = [(bt, p) for p in range(start, stop) for bt in boxes_by_ping.get(p, [])]
            try:
                _export_waterfall_png(
                    ds_window, out_path,
                    x_axis=x_axis, y_axis=y_axis, ping_offset=start,
                    cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                    vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma,
                    dpi=args.dpi, boxes=window_boxes, box_size=args.box_size,
                    value_var=value_var,
                )
                n_written += 1
            except Exception as e:
                logger.error(f"{freq_key}: window starting at ping {start} failed: {type(e).__name__}: {e}")
            n_windows += 1
            if stop == n_pings_total:
                break
        logger.info(f"{freq_key}: {n_windows} window(s) processed ({n_pings_total} total pings, window={args.window_pings}, overlap={args.overlap_pings})")

    logger.info("=" * 60)
    logger.info(f"export waterfall ({args.range_mode}, {args.value_var}) complete: {n_written} PNG(s) written")
    return 0 if n_written > 0 else 1


# ============================================================
# grid (replaces 09_grid.py)
# ============================================================

def _export_gridded_segment(ds, output_path: Path, *, resolution_m, method, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, dpi, show_land=False, show_locator=False):
    bbox_wgs84 = bbox_from_channel_dataset(ds)
    grid_def = utm_grid_from_wgs84_bbox(bbox_wgs84, resolution_m=resolution_m)
    raster = resample_channel_to_grid(ds, grid_def, method=ResamplingMethod(method), backscatter_var="backscatter_intensity")
    from sidescanpy.cli import _figures

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _figures.save_gridded_png(
        raster, str(output_path), dpi=dpi, cmap=cmap, vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct, gamma=gamma,
        show_land=show_land, show_locator=show_locator,
    )


def run_grid(args: argparse.Namespace) -> int:
    files = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1
    logger.info(f"Found {len(files)} .ss.h5 file(s) in {args.ss_h5_dir}")

    n_written = 0
    for path in files:
        # chunks= keeps this lazy (dask-backed) -- without it, the whole
        # file loads eagerly before any segmenting happens below,
        # defeating --segment-size's own purpose on a large real survey
        # (avoids OOM; every other exporter in
        # this file -- waterfall/track/chip -- already reads this way,
        # this one didn't). `.load()` on each segment slice below then
        # only ever materializes that one segment's data.
        survey = read_survey(path, chunks={"ping": args.segment_size})
        freq_keys = resolve_freq_keys(survey.frequencies, args.frequency)
        for freq_key in freq_keys:
            if freq_key not in survey.frequencies:
                continue
            ds = survey.get_frequency(freq_key).dataset
            n_pings = ds.sizes["ping"]
            n_segments = max(1, -(-n_pings // args.segment_size))
            for seg_idx in range(n_segments):
                start = seg_idx * args.segment_size
                stop = min(start + args.segment_size, n_pings)
                ds_segment = ds.isel(ping=slice(start, stop)).load()
                out_path = args.output_dir / freq_key / path.stem / f"segment_{seg_idx + 1:03d}.png"
                try:
                    _export_gridded_segment(
                        ds_segment, out_path,
                        resolution_m=args.grid_resolution_m, method=args.grid_method,
                        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                        vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma,
                        dpi=args.dpi, show_land=args.show_land, show_locator=args.show_locator,
                    )
                    n_written += 1
                except Exception as e:
                    logger.error(f"[{path.name}/{freq_key}] segment {seg_idx + 1} failed: {type(e).__name__}: {e}")
            logger.info(f"[{path.name}/{freq_key}] {n_segments} segment(s) processed")

    logger.info("=" * 60)
    logger.info(f"export grid complete: {n_written} PNG(s) written")
    return 0 if n_written > 0 else 1


# ============================================================
# track (replaces 03_track_maps.py / 12_track_with_targets.py)
# ============================================================

def _resolve_frequencies(surveys: dict, requested: str) -> list[str]:
    freqs: set[str] = set()
    for survey in surveys.values():
        freqs.update(survey.frequencies)
    return resolve_freq_keys(freqs, requested)


def run_track(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _figures

    paths = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not paths:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1
    logger.info(f"Loading {len(paths)} .ss.h5 file(s) (lazy)")
    surveys = {p: read_survey(p, chunks={"ping": _DEFAULT_SEGMENT_SIZE}) for p in paths}

    freq_keys = _resolve_frequencies(surveys, args.frequency)
    if not freq_keys:
        logger.error("No frequencies found across input files")
        return 1

    targets = read_targets_json(args.targets_json) if args.targets_json else None
    if targets is not None:
        logger.info(f"Overlaying {len(targets)} target(s)")

    n_written = 0
    for freq_key in freq_keys:
        logger.info(f"=== {freq_key} ===")
        track = build_survey_track(surveys, freq_key, scanline_stride=args.scanline_stride)
        if track is None:
            logger.warning(f"  No file contains {freq_key}; skipping")
            continue

        out_path = args.output_dir / freq_key / "track.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if targets is not None:
                _figures.save_track_with_targets_png(
                    track, targets, str(out_path), dpi=args.dpi,
                    show_land=args.show_land, show_locator=args.show_locator,
                )
            else:
                _figures.save_track_png(
                    track, str(out_path), dpi=args.dpi,
                    show_land=args.show_land, show_locator=args.show_locator,
                )
            logger.info(f"  Wrote {out_path}")
            n_written += 1
        except Exception as e:
            logger.error(f"  [{freq_key}] track export failed: {type(e).__name__}: {e}")

    logger.info("=" * 60)
    logger.info(f"export track complete: {n_written}/{len(freq_keys)} frequency track(s) written")
    return 0 if n_written > 0 else 1


# ============================================================
# chip / survey-window / progression
# ============================================================

def _save_grayscale(img_8bit: np.ndarray, path: Path) -> None:
    """Save a true single-channel 8-bit (or, for a ``uint16`` array,
    16-bit) grayscale PNG.

    Deliberately not ``matplotlib.pyplot.imsave`` — even with
    ``cmap="gray"`` it always writes RGB(A), which is redundant-channel
    grayscale, not the single-channel 8-bit file the dataset spec calls
    for. ``PIL.Image`` mode ``"L"`` writes the real thing.
    """
    if img_8bit.dtype == np.uint16:
        Image.fromarray(img_8bit).save(path)  # PIL mode "I;16"
        return
    Image.fromarray(img_8bit, mode="L").save(path)


def _reference_stats_for_scope(scope: str, arrays: list[np.ndarray], pct_low: float, pct_high: float) -> ReferenceStats | None:
    if scope == "image":
        return None
    if scope == "shared":
        return compute_reference_stats(*arrays, pct_low=pct_low, pct_high=pct_high)
    raise ValueError(f"unknown --normalization-scope: {scope!r}")


_CHIP_STYLES = ("chip", "single-swath", "dual-swath")
_RENDER_LEVELS = ("uncorrected", "db", "radiometric", "geometric", "georef")
# Native sample/ping pixel-space levels (no resolution_m concept) -- the
# rest resample onto a real-world grid.
_PIXEL_LEVELS = ("uncorrected", "db", "radiometric")
# Level/--value-var name -> backscatter variable. Neither "uncorrected"
# nor "backscatter_db" is stored on disk: "uncorrected" is the
# uncorrected signal, inverted from processing_history by _with_uncorrected_var /
# _mask_nadir_gap; "backscatter_db" is 20*log10 of it with no TVG gain, added
# per loaded neighborhood by _with_db_var (the same raw->dB step
# apply_tvg_to_amplitude does before adding gain -- the progression grid's
# own "db" step).
_LEVEL_VARS = {"uncorrected": UNCORRECTED_VAR, "db": "backscatter_db", "radiometric": "backscatter_intensity"}



def _with_uncorrected_var(ds: xr.Dataset) -> xr.Dataset:
    """Add ``uncorrected`` (the uncorrected signal, inverted from
    ``processing_history``; lazy on dask data) unless ``ds`` already has it."""
    if UNCORRECTED_VAR in ds:
        return ds
    return ds.assign({UNCORRECTED_VAR: uncorrected_backscatter(ds)})


def _mask_nadir_gap(ds: xr.Dataset) -> xr.Dataset:
    """``mask_nadir_gap`` plus ``uncorrected``, masked identically (derived
    from the masked intensity)."""
    return _with_uncorrected_var(mask_nadir_gap(ds, value_vars=("backscatter_intensity", UNCORRECTED_VAR)))


def _with_db_var(ds: xr.Dataset) -> xr.Dataset:
    """Add ``backscatter_db`` (``20*log10`` of the uncorrected signal, no TVG) to ``ds``."""
    if "backscatter_db" in ds:
        return ds
    return ds.assign(backscatter_db=linear_to_db(_with_uncorrected_var(ds)[UNCORRECTED_VAR]))


# Geometric/georef single-swath/dual-swath: source cells kept per output
# pixel (per axis) after the pre-resample block average in
# _load_swath_neighborhood. 2 keeps the resample well-fed without
# carrying the full native density.
_SWATH_OVERSAMPLE = 2.0
_SWATH_LOAD_BLOCK_PINGS = 2048
# Variables the geometric/georef resample (and mask_nadir_gap) read --
# everything else in a corrected .ss.h5 is left on disk.
_SWATH_LOAD_VARS = (
    "backscatter_intensity", "ground_range_m", "across_track_m",
    "latitude_per_sample", "longitude_per_sample", "latitude", "longitude", "sound_vel_ms",
)


def _neighborhood_ping_spacing_m(ds_full: xr.Dataset, lo: int, hi: int) -> float:
    lat_lon = ds_full[["latitude", "longitude"]].isel(ping=slice(lo, hi)).load()
    along = along_track_distance_m(lat_lon, fill_nav_stalls=True)
    return abs(float(along[-1] - along[0])) / max(along.size - 1, 1) if along.size > 1 else 0.0


def _geo_decimation(
    ds_full: xr.Dataset, ping_idx: int, style: str, *, size_px: int, jitter_px: int, resolution_m: float | None,
    across_axis: str = "ground", swath_fill: bool = False,
) -> tuple[int, int, int]:
    """(half_span_pings, fp, fs) for one geometric/georef style around
    ``ping_idx``: a ping neighborhood covering the output window in
    *meters* (``size_px * resolution_m`` along-track plus jitter, with
    slack for turns and edge-shifting -- sizing it in pings instead
    left geometric chips short of data at their top/bottom, which
    griddata's nearest fill smeared into streaks), and (ping, sample)
    block-average factors that keep ~``_SWATH_OVERSAMPLE`` source
    cells per output pixel per axis. ``resolution_m`` ``None``
    estimates the auto value ``_extract_geo_style_window`` will pick
    (chip: native ``max(mps, ping_spacing)``; swath styles: ground
    swath width / ``size_px``)."""
    n_pings = ds_full.sizes["ping"]
    mps = _sample_interval_m(ds_full)
    probe = 500
    ping_spacing_m = _neighborhood_ping_spacing_m(ds_full, max(0, ping_idx - probe), min(n_pings, ping_idx + probe)) or mps
    if resolution_m is None:
        if style == "chip":
            resolution_m = max(mps, ping_spacing_m)
        else:
            across = _across_values(ds_full.isel(ping=ping_idx), across_axis).values
            if style != "dual-swath":
                across = across[0]
            finite = across[np.isfinite(across)]
            swath_width_m = float(finite.max() - finite.min()) if finite.size else size_px * mps
            # --swath-fill drops the ping-spacing floor (see
            # _extract_geo_style_window); estimate the same way so the
            # sample-axis averaging stays fine enough.
            resolution_m = swath_width_m / size_px if swath_fill else max(swath_width_m / size_px, ping_spacing_m)
    half_extent_m = (size_px / 2.0 + jitter_px) * resolution_m
    half_span = int(1.5 * half_extent_m / ping_spacing_m) + 50
    fs = max(1, int(resolution_m / (_SWATH_OVERSAMPLE * mps)))
    fp = max(1, int(resolution_m / (_SWATH_OVERSAMPLE * ping_spacing_m)))
    # Never average an axis down to (almost) nothing -- happens when the
    # other axis's spacing sets resolution_m (e.g. a swath narrower than
    # one along-track pixel).
    fs = min(fs, max(1, ds_full.sizes["sample"] // 8))
    fp = min(fp, max(1, (2 * half_span) // 8))
    return half_span, fp, fs


@dataclass
class _GeoView:
    """A ``_load_geo_neighborhood`` result, in its own averaged units."""

    ds: xr.Dataset
    target: object
    local_ping: int
    mps: float
    items: dict
    lo: int


def _load_geo_neighborhood(
    ds_full: xr.Dataset, t, ping_idx: int, items_by_channel: dict, styles: list[str], *,
    size_px: int, jitter_px: int, resolution_m: float | None, with_db: bool, across_axis: str = "ground",
    per_sample_latlon: bool = True, swath_fill: bool = False,
) -> _GeoView:
    """Block-averaged neighborhood around ``ping_idx`` serving every
    style in ``styles`` (widest span, finest factors), with ``t`` and
    ``items_by_channel`` translated into its averaged sample/ping units.
    Global ping ``p`` maps to ``lo + (p - lo) // fp``, so
    ``_geo_targets_within_window``'s own ``p - lo`` lands on the right
    averaged row.

    ``across_axis="slant"`` replaces the loaded ``across_track_m`` with
    signed slant range (see ``_across_values``), so every downstream
    geometric extraction, co-target lookup and swath-width estimate
    works in slant-range meters instead of ground-projected meters."""
    n_pings = ds_full.sizes["ping"]
    plans = [
        _geo_decimation(ds_full, ping_idx, st, size_px=size_px, jitter_px=jitter_px, resolution_m=resolution_m, across_axis=across_axis, swath_fill=swath_fill)
        for st in styles
    ]
    half_span = max(p[0] for p in plans)
    fp = min(p[1] for p in plans)
    fs = min(p[2] for p in plans)
    lo = max(0, ping_idx - half_span)
    hi = min(n_pings, ping_idx + half_span)
    extra_vars = ("slant_range_m",) if across_axis == "slant" else ()
    # Per-sample lat/lon are only read by georef; the geometric level
    # skips them (two full float64 per-sample arrays per neighborhood).
    skip_vars = () if per_sample_latlon else ("latitude_per_sample", "longitude_per_sample")
    ds = _load_swath_neighborhood(ds_full, lo, hi, fp, fs, with_db=with_db, extra_vars=extra_vars, skip_vars=skip_vars)
    if across_axis == "slant":
        ds["across_track_m"] = _across_values(ds, "slant")
    items = {
        ch: [(_decimate_target(tt, fs, fp), lo + (p - lo) // fp) for tt, p in ch_its]
        for ch, ch_its in items_by_channel.items()
    }
    logger.debug(f"target {t.target_id} {styles}: {hi - lo} pings averaged {fp}x{fs} (ping x sample) -> {dict(ds.sizes)}")
    return _GeoView(ds=ds, target=_decimate_target(t, fs, fp), local_ping=(ping_idx - lo) // fp, mps=_sample_interval_m(ds), items=items, lo=lo)


def _load_swath_neighborhood(
    ds_full: xr.Dataset, lo: int, hi: int, fp: int, fs: int, *, with_db: bool, extra_vars: tuple[str, ...] = (),
    skip_vars: tuple[str, ...] = (),
) -> xr.Dataset:
    """Nadir-masked ``[lo, hi)`` ping neighborhood, block-averaged by
    ``fp`` pings x ``fs`` samples before any geometric/georef resample.

    Loads in ``_SWATH_LOAD_BLOCK_PINGS``-sized pieces, only
    ``_SWATH_LOAD_VARS``, masking each piece at native resolution
    before averaging -- a full-swath neighborhood is ~2x n_samples
    pings, and loading every per-sample array in a corrected file at
    once was ~3GB per target on a real survey, with the resample's
    own working set on top (44GB RSS observed). ``sample_rate_hz`` is
    divided by ``fs`` so ``_sample_interval_m`` reports the averaged
    cell size.
    """
    keep = [v for v in (*_SWATH_LOAD_VARS, *extra_vars) if v in ds_full and v not in skip_vars]
    step = fp * max(1, _SWATH_LOAD_BLOCK_PINGS // fp)
    blocks = []
    for b0 in range(lo, hi, step):
        blk = _mask_nadir_gap(ds_full[keep].isel(ping=slice(b0, min(hi, b0 + step))).load())
        if with_db:
            blk = _with_db_var(blk)
        if fp > 1 or fs > 1:
            blk = blk.coarsen(ping=fp, sample=fs, boundary="trim").mean(keep_attrs=True)
        blocks.append(blk)
    out = xr.concat(blocks, dim="ping") if len(blocks) > 1 else blocks[0]
    out.attrs = {**ds_full.attrs, "sample_rate_hz": float(ds_full.attrs["sample_rate_hz"]) / fs}
    return out


def _across_values(ds: xr.Dataset, across_axis: str):
    """Across-track coordinate (port negative, starboard positive) for a
    geometric window: ``across_track_m`` (ground-projected, the default)
    or, for ``across_axis="slant"``, ``slant_range_m`` carrying the same
    sign convention -- the slant-range (no ground projection) analogue,
    still in real meters, so a ``size_px * resolution_m`` window is the
    same real extent on both axes. Water column is NOT removed in slant
    mode (``slant_range_m`` is finite everywhere); ``mask_nadir_gap``
    still blanks the detected nadir gap's backscatter either way."""
    if across_axis == "ground":
        return ds["across_track_m"]
    if across_axis != "slant":
        raise ValueError(f"unknown across_axis: {across_axis!r}")
    if "slant_range_m" not in ds:
        raise KeyError("slant_range_m")
    sign = xr.DataArray([-1.0, 1.0], dims="channel")
    return ds["slant_range_m"] * sign


# --swath-fill: percentile of per-ping reach that sets the swath frame's
# edges -- the far edge is where at least (100 - this)% of pings still
# have data, the near (nadir-side, single-swath only) edge where at least
# that share already do. Longer-reaching pings are cropped slightly
# instead of every row carrying a black margin.
_SWATH_FILL_PCT = 10.0


def _swath_fill_extent(
    across: np.ndarray, values: np.ndarray, *, combine: bool, channel_idx: int | None,
) -> tuple[float, float] | None:
    """``(lo, hi)`` across-track meters that the recorded data fills, for
    a swath style's frame (``--swath-fill``). ``across``/``values``:
    ``(ping, channel, sample)``. A sample counts only where both the
    coordinate and the backscatter are finite -- in slant mode the
    coordinate is finite everywhere, including the masked nadir gap.

    Dual swath spans port's far edge to starboard's far edge (not
    centered on nadir: a side that reaches farther keeps its range). A
    side with no data at all (one-sided recording) mirrors the other, so
    nadir stays centered. Single swath spans that channel's near edge to
    its far edge. ``None`` if there is no usable data."""
    valid = np.isfinite(across) & np.isfinite(values)
    mag = np.where(valid, np.abs(across), np.nan)

    def reach(ch):
        m = mag[:, ch, :]
        rows = np.isfinite(m).any(axis=1)
        if not rows.any():
            return None
        with np.errstate(all="ignore"):
            far = float(np.percentile(np.nanmax(m[rows], axis=1), _SWATH_FILL_PCT))
            near = float(np.percentile(np.nanmin(m[rows], axis=1), 100.0 - _SWATH_FILL_PCT))
        return near, far

    if combine:
        port, stbd = reach(0), reach(1)
        if port is None and stbd is None:
            return None
        port_far = port[1] if port else stbd[1]
        stbd_far = stbd[1] if stbd else port[1]
        return -port_far, stbd_far
    r = reach(channel_idx)
    if r is None:
        return None
    near, far = r
    if not far > near:
        return None
    return (-far, -near) if channel_idx == 0 else (near, far)


def _random_flip(window: np.ndarray, co_targets: list, rng: np.random.Generator, *, edge_coords: bool) -> tuple[np.ndarray, list, bool, bool]:
    """Independently flip ``window`` horizontally and vertically, each
    with probability 0.5, remapping every co-target's ``sample`` (col) /
    ``window_row`` (row) to match -- a training-time augmentation, applied
    after display orientation, so the annotations stay exact.

    ``edge_coords``: geometric/georef windows locate targets in
    continuous pixel-edge coordinates (``locate_point``: pixel ``i``
    spans ``[i, i+1)``), mirrored as ``n - x``; pixel-space windows use
    integer sample/ping indices (pixel centers), mirrored as ``n - 1 - x``.
    Returns ``(window, co_targets, flipped_h, flipped_v)``."""
    flip_h, flip_v = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    n_rows, n_cols = window.shape
    off = 0 if edge_coords else 1
    if flip_h:
        window = window[:, ::-1]
        co_targets = [replace(ct, sample=n_cols - off - ct.sample) for ct in co_targets]
    if flip_v:
        window = window[::-1, :]
        co_targets = [
            replace(ct, extra_attributes={**ct.extra_attributes, "window_row": n_rows - off - ct.extra_attributes["window_row"]})
            for ct in co_targets
        ]
    return np.ascontiguousarray(window), co_targets, flip_h, flip_v


def _augmented_file_name(target_id: str, aug_idx: int, n_augment: int) -> str:
    """``target_<id>.png`` (single draw, unchanged naming) or
    ``target_<id>_a<k>.png`` for one of ``n_augment`` draws."""
    return f"target_{target_id}.png" if n_augment <= 1 else f"target_{target_id}_a{aug_idx}.png"


def _decimate_target(t, fs: int, fp: int):
    """``t`` in a ``_load_swath_neighborhood`` array's (fp, fs)-averaged
    sample/ping pixel units -- ``sample`` and ``bbox`` (ping axis = y)."""
    bbox = None if t.bbox is None else (t.bbox[0] / fs, t.bbox[1] / fp, t.bbox[2] / fs, t.bbox[3] / fp)
    return replace(t, sample=None if t.sample is None else t.sample // fs, bbox=bbox)


def _style_size_px(style: str, args: argparse.Namespace) -> int:
    return args.size_px if style == "chip" else args.swath_size_px


def _orient_pixel_window(window: np.ndarray, co_targets: list, channel_idx: int, style: str) -> tuple[np.ndarray, list]:
    """Match ``targets.display.orient_for_display``'s two display
    conventions (found by comparing rendered output against EdgeTech
    Discover's own target thumbnails) for a chip/full-swath export
    window, which — unlike ``orient_for_display`` itself — can carry
    several co-located targets sharing one window and so needs every
    one of them remapped together, not just a single primary target.

    Vertical (ping-axis) flip always applies: ``targets_within_window``'s
    row convention is ascending ping order, top-down, the opposite of
    Discover's own display.

    Horizontal (port-mirror) flip applies only for a single physical
    channel's own window (``channel_idx == 0``, i.e. port) -- a
    "dual-swath" window is already port-mirrored by
    ``_combine_channels_mirrored`` at combination time, so flipping
    again here would un-mirror it.
    """
    n_rows, n_cols = window.shape
    window = window[::-1, :]
    co_targets = [
        replace(ct, extra_attributes={**ct.extra_attributes, "window_row": n_rows - 1 - ct.extra_attributes["window_row"]})
        for ct in co_targets
    ]
    if style != "dual-swath" and channel_idx == 0:
        window = window[:, ::-1]
        co_targets = [replace(ct, sample=n_cols - 1 - ct.sample) for ct in co_targets]
    return window, co_targets


def _geo_targets_within_window(
    candidates: list[tuple], masked, lo: int, along_dist: np.ndarray, size_px: int, meta: dict, locate_fn,
    along_size_px: int | None = None,
) -> list:
    """Real-world equivalent of ``targets_within_window`` for a
    geometric window: which of ``candidates`` (``(Target, global_ping_idx)``
    pairs, as gathered in ``run_chip``'s ``items_by_channel``) physically
    fall inside this specific window's along/across bounds, translated to
    window-local ``(row, col)`` via ``locate_fn`` (pre-flip -- see
    ``_flip_geo_vertical``). Mirrors ``_extract_geo_style_window``'s own
    per-target lookup, so a bad/out-of-range candidate is skipped, not a
    crash.

    ``along_size_px``, default ``None`` (falls back to ``size_px``,
    reproducing the prior square-only behavior exactly): the grid's real
    row count, when it differs from ``size_px`` (the column count) --
    real bug, found 2026-09-29 while making "geometric" render-level
    output genuinely non-square (anisotropic along-track pixel
    resolution, scipy binned): this function's row bound check used to
    reuse ``size_px`` for BOTH axes, silently accepting/rejecting
    co-targets against the wrong row limit once the two diverged.
    """
    row_limit = along_size_px if along_size_px is not None else size_px
    included = []
    for candidate_t, ping_idx in candidates:
        local_ping = ping_idx - lo
        if not (0 <= local_ping < masked.sizes["ping"]):
            continue
        c_idx = candidate_t.channel if candidate_t.channel is not None else 0
        if candidate_t.sample is None or not (0 <= candidate_t.sample < masked.sizes["sample"]):
            continue
        try:
            t_across = float(masked["across_track_m"].isel(ping=local_ping, channel=c_idx, sample=candidate_t.sample).values)
        except (KeyError, IndexError):
            continue
        if not np.isfinite(t_across):
            continue
        t_along = float(along_dist[local_ping])
        row, col = locate_fn(t_along, t_across, meta)
        if 0 <= row < row_limit and 0 <= col < size_px:
            included.append(replace(candidate_t, sample=col, extra_attributes={**candidate_t.extra_attributes, "window_row": row}))
    return included


def _flip_geo_vertical(grid: np.ndarray, positioned_targets: list, along_size_px: int) -> tuple[np.ndarray, list]:
    """Match Discover's ping-order display convention (see
    ``_orient_pixel_window``'s docstring for the raw-pixel equivalent) for
    a geometric window: ``extract_geometric_window``/
    ``extract_geometric_window_binned`` build row 0 = the minimum
    along-track distance (ascending ping order, top-down), the opposite
    of Discover's own display. ``positioned_targets`` must already carry
    window-local ``sample``/``extra_attributes["window_row"]`` (e.g. from
    ``locate_point_geometric`` via ``_geo_targets_within_window``).

    ``along_size_px`` -- the grid's actual ROW count (real bug, found
    2026-09-29: this used to be a ``size_px`` param reused for both axes,
    which silently mis-flipped ``window_row`` once along-track pixel
    resolution went anisotropic and row count stopped matching column
    count -- ``size_px - row`` only correctly mirrors an index when
    ``size_px`` IS the row count).
    """
    grid = grid[::-1, :]
    positioned_targets = [
        replace(pt, extra_attributes={**pt.extra_attributes, "window_row": along_size_px - pt.extra_attributes["window_row"]})
        for pt in positioned_targets
    ]
    return grid, positioned_targets


def _orient_geo_window(
    grid: np.ndarray, positioned_targets: list, size_px: int, *, combine: bool, channel_idx: int | None,
    along_size_px: int | None = None,
) -> tuple[np.ndarray, list]:
    """Full orientation correction for a "geometric" render-level window:
    the vertical flip (``_flip_geo_vertical``, always) plus a horizontal
    port-mirror flip for a single port channel's own window.

    Real, confirmed finding (2026-09-29, direct coordinate inspection,
    not a guess): ``extract_geometric_window`` builds columns in
    ascending ``across_track_m`` order. Port's across_track_m is
    negative, so ascending order puts the LEAST negative (nearest-nadir)
    value LAST -- near-range lands on the right edge. Starboard's is
    positive, so ascending order puts the smallest (nearest-nadir) value
    FIRST -- near-range lands on the left edge. A port window and a
    starboard window are therefore genuine mirror images of each other
    with no correction applied, which is exactly what every "geometric"
    render-level output (chip/single-swath/dual-swath's own per-channel
    windows, and this project's processing-slide tooling) has been doing
    -- confirmed via real target coordinates (0001, port, across_min=
    -16.28m; 0002, starboard, across_min=+15.55m -- near-range on
    opposite sides).

    This is a DIFFERENT reason than ``_orient_pixel_window``'s own
    port-mirror flip, which exists purely to match EdgeTech Discover's
    vendor display convention for pixel-space renders (sample-index
    ordering has no inherent near/far-range sign issue to begin with).
    There is no Discover convention to match in real-meters geometric
    space -- this fix exists purely for genuine near/far-range
    consistency between channels, with no vendor precedent either way.

    combine=True (dual-swath, channel_idx=None) never needs this: a
    pooled real-meters point cloud spans both true signs at once, so
    nadir is already correctly placed in the middle from the data's own
    real coordinates -- same reasoning ``_orient_pixel_window`` already
    applies for its own dual-swath case (already mirrored at combination
    time by ``_combine_channels_mirrored``).
    """
    grid, positioned_targets = _flip_geo_vertical(grid, positioned_targets, along_size_px if along_size_px is not None else size_px)
    if not combine and channel_idx == 0:
        n_cols = grid.shape[1]
        grid = grid[:, ::-1]
        positioned_targets = [replace(pt, sample=n_cols - 1 - pt.sample) for pt in positioned_targets]
    return grid, positioned_targets


def _rescale_co_targets(co_targets: list, native_shape: tuple[int, int], output_shape: tuple[int, int]) -> list:
    """Move every co-target's (row, col) from a native-resolution window
    into its resized output space, matching :func:`resize_to`'s own
    transform -- see :func:`rescale_target_position`."""
    rescaled = []
    for ct in co_targets:
        row, col = rescale_target_position(ct.extra_attributes["window_row"], ct.sample, native_shape, output_shape)
        rescaled.append(replace(ct, sample=col, extra_attributes={**ct.extra_attributes, "window_row": row}))
    return rescaled


def _with_bbox_size_px(t, scale_x: float, scale_y: float):
    """Stash ``t.bbox``'s own real, measured ``(w, h)`` -- converted from
    whatever native unit ``scale_x``/``scale_y`` convert from (source
    sample/ping pixels for the raw-pixel styles, meters-per-sample/
    meters-per-ping for the geo styles) into this specific window's
    actual output pixel size -- into ``extra_attributes["bbox_size_px"]``.

    Previously ``sidescan-export chip`` always wrote
    every target's COCO/YOLO box at the shared ``--box-size`` (a fixed
    placeholder), discarding a real, measured extent such as a
    catalog's 4-corner columns give every target.

    No-op (returns ``t`` unchanged) when ``t.bbox`` is ``None`` -- a
    center-only pick (e.g. JSF/EdgeTech Discover's own catalog, which
    has no measured extent) still falls back to the shared
    ``--box-size`` placeholder exactly as before this existed.
    """
    if t.bbox is None:
        return t
    _x, _y, w, h = t.bbox
    return replace(t, extra_attributes={**t.extra_attributes, "bbox_size_px": (float(w) * scale_x, float(h) * scale_y)})


def _extract_pixel_style_window(
    masked, t, channel_idx: int, n_samples_single: int, local_ping: int, lo: int, style: str, level: str,
    size_px: int, jitter_px: int, rng: np.random.Generator, placed_by_channel: dict, placed_combined: list,
) -> tuple[np.ndarray, list] | None:
    """``uncorrected``/``radiometric`` (sample/ping pixel space) window + the
    co-targets that land inside it, for one target/style. Returns
    ``None`` if this level's backscatter variable isn't present on
    ``masked`` (e.g. ``radiometric`` requested against a never-TVG'd
    file), or if ``masked``'s own ping neighborhood is too short for a
    full-swath style's native square window (a short source file/segment
    -- see ``extract_full_channel``'s own docstring) -- caller logs and
    skips, doesn't crash the whole run.
    """
    value_var = _LEVEL_VARS[level]
    if value_var not in masked:
        return None

    if style == "dual-swath":
        port_2d = masked[value_var].isel(channel=0).values
        stbd_2d = masked[value_var].isel(channel=1).values
        arr_2d = _combine_channels_mirrored(port_2d, stbd_2d)
        target_col = _target_combined_col(t.sample, n_samples_single, channel_idx)
        co_source = placed_combined
    else:
        arr_2d = masked[value_var].isel(channel=channel_idx).values
        target_col = t.sample
        co_source = placed_by_channel[channel_idx]

    try:
        if style == "chip":
            window, extent = extract_chip(arr_2d, center_row=local_ping, center_col=target_col, size_px=size_px, jitter_px=jitter_px, rng=rng)
        else:
            window, extent = extract_full_channel(arr_2d, center_ping=float(local_ping), jitter_px=jitter_px, rng=rng)
    except ValueError:
        return None

    extent_global = WindowExtent(row_start=extent.row_start + lo, col_start=extent.col_start, n_rows=extent.n_rows, n_cols=extent.n_cols)
    co_targets = targets_within_window(co_source, extent_global)
    if not co_targets:
        return None
    window, co_targets = _orient_pixel_window(window, co_targets, channel_idx, style)

    native_shape = window.shape
    # bbox_size_px scale: native window pixels ARE the same sample(x)/
    # ping(y) units t.bbox was recorded in (this style never resamples,
    # only crops) -- 1:1 until/unless the final resize below changes it.
    row_scale, col_scale = 1.0, 1.0
    if native_shape != (size_px, size_px):
        row_scale, col_scale = size_px / native_shape[0], size_px / native_shape[1]
        co_targets = _rescale_co_targets(co_targets, native_shape, (size_px, size_px))
        window = resize_to(window, size_px)
    co_targets = [_with_bbox_size_px(ct, scale_x=col_scale, scale_y=row_scale) for ct in co_targets]
    return window, co_targets


def _extract_geo_style_window(
    masked, t, channel_idx: int, local_ping: int, style: str, level: str, size_px: int, jitter_px: int,
    rng: np.random.Generator, mps: float, items_by_channel: dict, lo: int,
    resolution_m_override: float | None = None,
    backscatter_var: str = "backscatter_intensity",
    resample_method: str = "griddata",
    swath_fill: bool = False,
) -> tuple[np.ndarray, list] | None:
    """``geometric``/``georef`` (real-meters) window + the primary target's
    own position within it.

    ``geometric`` searches for co-occurring targets and applies the same
    Discover-matching vertical flip the raw-pixel styles get (see
    ``_geo_targets_within_window``/``_flip_geo_vertical``) -- ``georef``
    does neither yet: checking every cataloged target's real-world
    position against an arbitrarily-shaped resampled grid is more work
    for georef's own (lat/lon-based, CRS-projected) grid construction,
    and georef's row-axis convention (northing-based) isn't obviously
    the same "match Discover's ping order" question the geometric/
    raw-pixel styles share -- needs its own separate look before
    assuming the same fix applies. Documented, not silently missing.
    Returns ``None`` if this window
    can't be built (e.g. no finite geometry in this ping neighborhood, or
    ``level="georef"``/``"geometric"`` requested against a file that was
    never run through ``sidescan-correct --geometric``/``--georef``, so
    ``across_track_m``/``latitude_per_sample`` etc. don't exist at all).
    Also ``None`` if ``t.sample`` doesn't fit this *specific* channel's own
    sample axis -- real bug found running this against Klein SDF data: a
    target whose catalog entry carries no ``ping_meters_per_sample``
    (``match_frequency`` then has no basis to prefer one frequency
    channel over another and picks arbitrarily) can be correlated to a
    frequency/file whose own sample count is smaller than the sample
    index the target's position was originally captured against on a
    *different* channel -- an exact ``.isel(sample=t.sample)`` then raises
    ``IndexError`` where the pixel-space styles merely clamp silently
    (``extract_chip``/``extract_full_channel``'s own edge-shifting
    window placement never indexes out of bounds). Skipping here matches
    that same "don't crash the whole run over one bad target" contract,
    not a fix to the underlying frequency-matching ambiguity itself.

    ``resolution_m_override``, when given, is used directly as
    ``resolution_m`` for every style instead of the auto-calculated
    value -- see ``--resolution-m``'s own help text in ``main()`` for
    why the auto-calculation is unreliable per-target. ``ping_spacing_m``
    is still computed either way; it also feeds ``bbox_size_px``'s own
    unit conversion, independent of which resolution the crop itself
    renders at.

    ``backscatter_var``, added 2026-09-28 (real user ask: uncorrected vs.
    radiometric values weren't independently selectable once a
    geometric/georef *coordinate* system was chosen -- render-level
    conflated the two axes), selects which variable the resampled grid
    is actually built from -- passed straight through to
    ``extract_geometric_window``/``extract_georef_window``, which
    already supported this at the library level; only the CLI wiring
    was missing.

    ``resample_method``, added 2026-09-29 -- selects which of the three
    ``extract_geometric_window*`` functions the ``"geometric"`` render
    level uses: ``"kdtree"``, ``"binned"``, or ``"griddata"`` (default).
    Only affects ``level="geometric"`` -- ``"georef"`` always uses
    ``extract_georef_window``'s own isotropic KDTree radius search,
    unaffected by this parameter. See that branch's own inline comment
    for what each option actually trades off.
    """
    if t.sample is None or not (0 <= t.sample < masked.sizes["sample"]):
        return None
    try:
        along_dist = along_track_distance_m(masked, fill_nav_stalls=True)
        target_along_m = float(along_dist[local_ping])
        target_across_m = float(masked["across_track_m"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)
    except KeyError:
        return None

    ping_span_m = float(along_dist[-1] - along_dist[0]) if along_dist.size > 1 else 0.0
    ping_spacing_m = abs(ping_span_m) / max(along_dist.size - 1, 1) if along_dist.size > 1 else mps
    along_diffs = np.abs(np.diff(along_dist)) if along_dist.size > 1 else np.array([])
    nonzero_along_diffs = along_diffs[along_diffs > 0]
    gap_p95_m = float(np.percentile(nonzero_along_diffs, 95)) if nonzero_along_diffs.size else 0.0

    combine = style == "dual-swath"
    if style == "chip":
        resolution_m = resolution_m_override if resolution_m_override is not None else max(mps, ping_spacing_m)
        center_across_m = target_across_m + _jitter_offset_m(jitter_px, resolution_m, rng)
    elif swath_fill and resolution_m_override is None and (
        extent := _swath_fill_extent(masked["across_track_m"].values, masked[backscatter_var].values, combine=combine, channel_idx=channel_idx)
    ):
        # --swath-fill: the frame's width is exactly where the data is, at
        # square pixels -- no ping-spacing floor on resolution (that floor
        # left the swath narrower than the image whenever pings are
        # coarser than swath_width / size_px), and not pinned to nadir.
        lo_m, hi_m = extent
        resolution_m = max((hi_m - lo_m) / size_px, 1e-6)
        center_across_m = (lo_m + hi_m) / 2.0
    else:
        across_all = masked["across_track_m"].values if combine else masked["across_track_m"].isel(channel=channel_idx).values
        across_finite = across_all[np.isfinite(across_all)]
        swath_width_m = float(across_finite.max() - across_finite.min()) if across_finite.size else size_px * mps
        resolution_m = resolution_m_override if resolution_m_override is not None else max(swath_width_m / size_px, ping_spacing_m, 1e-6)
        center_across_m = 0.0 if combine else (float((across_finite.max() + across_finite.min()) / 2.0) if across_finite.size else 0.0)
    # geo_search_radius_m: kept for the georef branch below (Tier 2, a
    # harder, separate fix -- see docs/planning/anisotropic-crop-search-
    # radius.md), which still uses extract_georef_window's own isotropic
    # KDTree radius search.
    geo_search_radius_m = max(1.5 * resolution_m, gap_p95_m)

    if level == "geometric":
        # Selectable resampling method (2026-09-29, real user direction
        # after reviewing a real-data comparison of all three): square
        # size_px x size_px output for every method, one fixed
        # resolution_m on both axes -- no anisotropic-pixel-resolution
        # widening (tried in an earlier version of this function, walked
        # back: it made output non-square, which the user confirmed is
        # unusable for training regardless of the gap tradeoff it was
        # solving for).
        #
        # - "kdtree" (extract_geometric_window): real physical average
        #   of nearby measurements within along_search_radius_m/
        #   across_search_radius_m. Zero fabricated values, some
        #   directional blur at fine resolution (the "ellipse footprint"
        #   this family's earlier sessions investigated).
        # - "binned" (extract_geometric_window_binned): real measured
        #   means only, strict rectangle membership. Zero fabricated
        #   values, but a bin with no real sample inside it is NaN --
        #   real, honest gaps, not filled. Can be the majority of the
        #   grid at a resolution much finer than real along-track ping
        #   spacing (confirmed on real data: up to ~66% NaN at 0.03m).
        # - "griddata" (extract_geometric_window_griddata, DEFAULT):
        #   scipy.interpolate.griddata full interpolation, never NaN inside the recorded coverage (NaN only where no real sample is nearby -- past max range, nadir).
        #   Real user choice after reviewing the real-data comparison
        #   directly -- accepted tradeoff: at a resolution finer than
        #   real along-track spacing, most pixels are interpolated
        #   (constructed from nearby real points), not measured.
        #
        # See sidescanpy's CHANGELOG.md "resample_method" entry for the
        # real-data comparison that motivated offering all three rather
        # than picking one silently.
        geom_channel_idx = None if combine else channel_idx
        center_along_m = target_along_m + _jitter_offset_m(jitter_px, resolution_m, rng)
        across_search_radius_m = max(1.5 * resolution_m, 1.5 * mps)
        try:
            if resample_method == "kdtree":
                grid, meta = extract_geometric_window(
                    masked, size_px, resolution_m=resolution_m, center_along_m=center_along_m,
                    center_across_m=center_across_m, channel_idx=geom_channel_idx,
                    along_search_radius_m=geo_search_radius_m, across_search_radius_m=across_search_radius_m,
                    avoid_edge_gaps=True, backscatter_var=backscatter_var,
                )
            elif resample_method == "binned":
                grid, meta = extract_geometric_window_binned(
                    masked, size_px, resolution_m=resolution_m, center_along_m=center_along_m,
                    center_across_m=center_across_m, channel_idx=geom_channel_idx,
                    avoid_edge_gaps=True, backscatter_var=backscatter_var,
                )
            else:
                grid, meta = extract_geometric_window_griddata(
                    masked, size_px, resolution_m=resolution_m, center_along_m=center_along_m,
                    center_across_m=center_across_m, channel_idx=geom_channel_idx,
                    avoid_edge_gaps=True, backscatter_var=backscatter_var,
                )
        except ValueError:
            return None
        row, col = locate_point_geometric(target_along_m, target_across_m, meta)
        primary_local = replace(t, sample=col, extra_attributes={**t.extra_attributes, "window_row": row})
        all_items = [item for items in items_by_channel.values() for item in items] if combine else items_by_channel.get(channel_idx, [])
        candidates = [item for item in all_items if item[0].target_id != t.target_id]
        co_targets = _geo_targets_within_window(candidates, masked, lo, along_dist, size_px, meta, locate_point_geometric)
        grid, targets_out = _orient_geo_window(grid, [primary_local] + co_targets, size_px, combine=combine, channel_idx=channel_idx)
        # bbox_size_px scale: t.bbox's (w, h) is in source sample(x)/
        # ping(y) pixel counts -- convert to real meters via mps/
        # ping_spacing_m, then to this window's own output pixels via
        # resolution_m (meters per output pixel) -- same on both axes
        # again now that every resample_method produces square output.
        targets_out = [_with_bbox_size_px(ct, scale_x=mps / resolution_m, scale_y=ping_spacing_m / resolution_m) for ct in targets_out]
        targets_out = [replace(ct, extra_attributes={**ct.extra_attributes, "window_resolution_m": resolution_m}) for ct in targets_out]
        return grid, targets_out
    else:  # georef
        try:
            target_lat = float(masked["latitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)
            target_lon = float(masked["longitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)
        except KeyError:
            return None
        north_jitter_m = _jitter_offset_m(jitter_px, resolution_m, rng)
        east_jitter_m = _jitter_offset_m(jitter_px, resolution_m, rng)
        center_lat, center_lon = _jitter_latlon(target_lat, target_lon, north_jitter_m, east_jitter_m)
        georef_input = masked
        if not combine:
            excluded = 1 - channel_idx
            georef_input = masked.copy()
            georef_input[backscatter_var] = georef_input[backscatter_var].copy()
            georef_input[backscatter_var][{"channel": excluded}] = np.nan
        try:
            raster, meta = extract_georef_window(
                georef_input, size_px, resolution_m=resolution_m, center_lat=center_lat, center_lon=center_lon,
                max_search_radius_m=geo_search_radius_m, backscatter_var=backscatter_var,
            )
        except (ValueError, KeyError):
            return None
        grid = raster.dataset["backscatter"].values
        if grid.shape != (size_px, size_px):
            padded = np.full((size_px, size_px), np.nan)
            r = min(grid.shape[0], size_px)
            c = min(grid.shape[1], size_px)
            padded[:r, :c] = grid[:r, :c]
            grid = padded
        row, col = locate_point_georef(target_lat, target_lon, meta)
        primary_local = replace(t, sample=col, extra_attributes={**t.extra_attributes, "window_row": row})
        # Same conversion as the "geometric" branch above -- mps/
        # ping_spacing_m/resolution_m are all computed once, shared by
        # both branches.
        primary_local = _with_bbox_size_px(primary_local, scale_x=mps / resolution_m, scale_y=ping_spacing_m / resolution_m)
        primary_local = replace(primary_local, extra_attributes={**primary_local.extra_attributes, "window_resolution_m": resolution_m})
        return grid, [primary_local]


def _parse_categories(value) -> list[str] | None:
    """``--categories a,b`` (or a list from a config file) -> ordered names, or ``None``."""
    if value is None:
        return None
    names = [v.strip() for v in value.split(",")] if isinstance(value, str) else [str(v) for v in value]
    names = [n for n in names if n]
    return names or None


def run_chip(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _figures
    from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context

    resolution_m_override = getattr(args, "resolution_m", None)
    if resolution_m_override is not None and args.render_level in _PIXEL_LEVELS:
        logger.warning(
            f"--resolution-m={resolution_m_override} has no effect under --render-level {args.render_level!r} "
            "(native pixel crop, no resolution_m concept) -- ignoring it",
        )
        resolution_m_override = None
    resample_method = getattr(args, "resample_method", "griddata")

    # --render-level uncorrected/radiometric already fully determines pixel-style
    # values (see _extract_pixel_style_window); --value-var only matters
    # once a geometric/georef coordinate system is in play (chip styles,
    # and --with-context in either coordinate mode -- see run_chip's own
    # --with-context handling below).
    value_var = getattr(args, "value_var", "radiometric")
    if value_var != "radiometric" and args.render_level in _PIXEL_LEVELS:
        logger.warning(
            f"--value-var={value_var} has no effect under --render-level {args.render_level!r} -- that flag alone already "
            "selects uncorrected vs. radiometric for pixel-space styles; ignoring --value-var",
        )
    backscatter_var = _LEVEL_VARS[value_var]

    args.output.mkdir(parents=True, exist_ok=True)
    targets = read_targets_json(args.targets_json)
    if args.min_confidence != "unknown":
        targets = [t for t in targets if _CONFIDENCE_RANK[t.position_confidence] >= _CONFIDENCE_RANK[args.min_confidence]]
    logger.info(f"Read {len(targets)} target(s) at or above min-confidence={args.min_confidence!r}")
    categories = _parse_categories(getattr(args, "categories", None))
    if categories is not None:
        # Fail before any expensive rendering rather than at the final write.
        unknown = sorted({t.category for t in targets if t.category is not None} - set(categories))
        if unknown:
            raise SystemExit(f"error: target categories {unknown} are not in --categories {categories}")

    index = build_position_index(args.ss_h5_dir)

    matched = []
    for t in targets:
        m = correlate_target(t, index, args.max_match_dist_m)
        if m is None:
            logger.warning(f"target {t.target_id}: no ss_h5 ping within {args.max_match_dist_m}m of its recorded position, skipping")
            continue
        path, freq_key, ping_idx, dist_m = m
        logger.debug(f"target {t.target_id} -> {path.name} {freq_key} ping {ping_idx} ({dist_m:.1f}m)")
        matched.append((t, path, freq_key, ping_idx))

    if not matched:
        logger.error("No targets correlated to any ss_h5 ping -- nothing to export")
        return 1

    # Group by (file, frequency) only -- NOT also channel, unlike before:
    # a "dual-swath" style window spans both physical channels, so
    # co-target lookup for it needs both channels' matched targets
    # together, not split into separate per-channel groups.
    groups: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for t, path, freq_key, ping_idx in matched:
        channel_idx = t.channel if t.channel is not None else 0
        groups[(path, freq_key)][channel_idx].append((t, ping_idx))

    styles = list(args.styles)
    for style in styles:
        (args.output / style).mkdir(parents=True, exist_ok=True)
    if args.with_context:
        (args.output / "context").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    geo = args.render_level not in _PIXEL_LEVELS
    need_db = "db" in (args.render_level, value_var)
    n_augment = max(1, int(getattr(args, "n_augment", 1) or 1))
    random_flip = bool(getattr(args, "random_flip", False))
    across_axis = getattr(args, "across_axis", "ground")
    box_size_m = getattr(args, "box_size_m", None)
    swath_fill = bool(getattr(args, "swath_fill", False))
    if across_axis != "ground" and args.render_level != "geometric":
        logger.warning(f"--across-axis={across_axis} only applies to --render-level geometric -- ignoring it")
        across_axis = "ground"
    if box_size_m is not None and not geo:
        logger.warning("--box-size-m only applies to geometric/georef render levels -- ignoring it, using --box-size")
        box_size_m = None

    # Per-style output state. Under --normalization-scope image (default)
    # every window is rendered and written as soon as it's extracted, so
    # peak memory doesn't grow with target count x --n-augment (32 targets
    # x 4 draws of 1024 px float64 windows is ~1GB if held to the end);
    # "shared" needs every window first and keeps the old hold-then-write path.
    stream = args.normalization_scope == "image"
    pending: dict[str, list] = {s: [] for s in styles}  # (window, co_targets, primary, freq_key, file_name, aug_idx)
    coco_entries: dict[str, list] = {s: [] for s in styles}
    sheet: dict[str, tuple[list, list, list]] = {s: ([], [], []) for s in styles}  # thumbnails, captions, boxes
    aug_log: dict[str, list] = {s: [] for s in styles}
    n_written: dict[str, int] = {s: 0 for s in styles}

    def emit(style, window, co_targets, primary, freq_key, file_name, aug_idx, reference_stats=None):
        img_8bit, _meta = to_uint8(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high, reference_stats=reference_stats)
        style_dir = args.output / style
        if getattr(args, "bit_depth", 8) == 16:
            # Dataset image at 16-bit; the contact sheet still uses the
            # 8-bit render (same stretch/gamma).
            img_16bit, _meta = to_uint16(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high, reference_stats=reference_stats)
            _save_grayscale(img_16bit, style_dir / file_name)
        else:
            _save_grayscale(img_8bit, style_dir / file_name)
        for co in co_targets:
            coco_entries[style].append(
                CocoImageEntry(
                    target=co, file_name=file_name, width=img_8bit.shape[1], height=img_8bit.shape[0],
                    bbox_size=co.extra_attributes.get("bbox_size_px"),
                )
            )
        # Contact sheet: first draw per target only -- one row of every
        # augmentation of every target is unreadable and, at 1024 px,
        # a multi-GB figure.
        if aug_idx == 0:
            thumbnails, captions, sheet_boxes = sheet[style]
            thumbnails.append(img_8bit)
            sheet_boxes.append(co_targets)
            n_co = len(co_targets)
            captions.append(f"{primary.target_id} ({freq_key}, {n_co} target{'s' if n_co != 1 else ''})")
        n_written[style] += 1

    for (path, freq_key), items_by_channel in groups.items():
        survey = read_survey(path, chunks={"ping": 512})
        ds_full = survey.get_frequency(freq_key).dataset
        n_pings = ds_full.sizes["ping"]
        n_samples_single = ds_full.sizes["sample"]

        # Full-file-ping-coordinate target lists for co-target lookup:
        # one per physical channel (chip/single-swath stay same-channel-
        # only, matching a real operator's own recorded side), plus one
        # combined-column version spanning both channels (dual-swath).
        placed_by_channel = {
            ch: [replace(t, extra_attributes={**t.extra_attributes, "window_row": ping_idx}) for t, ping_idx in its]
            for ch, its in items_by_channel.items()
        }
        placed_combined = [
            replace(t, sample=_target_combined_col(t.sample, n_samples_single, ch), extra_attributes={**t.extra_attributes, "window_row": ping_idx})
            for ch, its in items_by_channel.items()
            for t, ping_idx in its
        ]

        # Geometric/georef levels read their own block-averaged, meter-
        # sized neighborhoods (_load_geo_neighborhood): one for the chip
        # style, one shared by the swath styles and the geometric
        # context view (a dual-swath extraction). Only the pixel-space
        # levels need the native-resolution `masked` below.
        swath_geo_styles = [s for s in styles if s != "chip"] if geo else []
        if geo and args.with_context:
            swath_geo_styles = list(dict.fromkeys(swath_geo_styles + ["dual-swath"]))
        max_native_span = 0
        if not geo:
            if "chip" in styles:
                max_native_span = max(max_native_span, args.size_px)
            if "single-swath" in styles:
                max_native_span = max(max_native_span, n_samples_single)
            if "dual-swath" in styles:
                max_native_span = max(max_native_span, 2 * n_samples_single)
        half_span = max_native_span // 2 + args.jitter_px + 50
        orig_bbox = {tt.target_id: tt.bbox for its in items_by_channel.values() for tt, _ in its}

        for channel_idx, its in items_by_channel.items():
            for t, ping_idx in its:
                # Slice a bounded ping neighborhood before loading/masking
                # -- this project's real files run up to 14.8GB with
                # every correction step's arrays included; loading one
                # target's neighborhood is bounded regardless of file
                # size, loading the whole file per group isn't.
                if not geo:
                    lo = max(0, ping_idx - half_span)
                    hi = min(n_pings, ping_idx + half_span)
                    masked = _mask_nadir_gap(ds_full.isel(ping=slice(lo, hi)).load())
                    if need_db:
                        masked = _with_db_var(masked)
                    local_ping = ping_idx - lo
                    _sample_interval_m(masked)
                else:
                    views = {}
                    try:
                        if "chip" in styles:
                            views["chip"] = _load_geo_neighborhood(
                                ds_full, t, ping_idx, items_by_channel, ["chip"], size_px=args.size_px,
                                jitter_px=args.jitter_px, resolution_m=resolution_m_override, with_db=need_db,
                                across_axis=across_axis, per_sample_latlon=args.render_level == "georef",
                                swath_fill=swath_fill,
                            )
                        if swath_geo_styles:
                            views["swath"] = _load_geo_neighborhood(
                                ds_full, t, ping_idx, items_by_channel, swath_geo_styles, size_px=args.swath_size_px,
                                jitter_px=args.jitter_px, resolution_m=resolution_m_override, with_db=need_db,
                                across_axis=across_axis, per_sample_latlon=args.render_level == "georef",
                                swath_fill=swath_fill,
                            )
                    except KeyError as exc:
                        logger.warning(f"target {t.target_id}: {path.name} has no {exc} (run sidescan-correct --geometric) -- skipping")
                        continue

                def geo_window(style, size_px, jitter_px, channel_idx=channel_idx, orig_bbox=orig_bbox):
                    # B023: only called within this loop iteration; `views` is unbound on
                    # the non-geo branch so it cannot be bound as a default arg.
                    v = views["chip" if style == "chip" else "swath"]  # noqa: B023
                    result = _extract_geo_style_window(
                        v.ds, v.target, channel_idx, v.local_ping, style, args.render_level, size_px, jitter_px, rng, v.mps,
                        v.items, v.lo, resolution_m_override=resolution_m_override, backscatter_var=backscatter_var,
                        resample_method=resample_method, swath_fill=swath_fill and args.render_level == "geometric",
                    )
                    if result is None:
                        return None
                    grid, targets_out = result
                    return grid, [replace(ct, bbox=orig_bbox.get(ct.target_id, ct.bbox)) for ct in targets_out]

                if args.with_context:
                    if args.render_level in _PIXEL_LEVELS:
                        # Native pixel-space QA view -- always the FULL
                        # loaded neighborhood (both channels, mirrored/
                        # combined), never a jittered/cropped target
                        # window, matching its "ground truth" purpose.
                        # --value-var (default radiometric, matching this
                        # branch's own long-standing behavior) is the only
                        # thing that varies here -- coordinate system was
                        # always native pixel space and stays that way.
                        port_2d = masked[backscatter_var].isel(channel=0).values
                        stbd_2d = masked[backscatter_var].isel(channel=1).values
                        combined = _combine_channels_mirrored(port_2d, stbd_2d)
                        combined_col = _target_combined_col(t.sample, n_samples_single, channel_idx)
                        context_target = replace(t, sample=combined_col, extra_attributes={**t.extra_attributes, "window_row": local_ping})
                        # Native, un-resized pixels: t.bbox's own (w, h) is already in this window's units.
                        context_target = _with_bbox_size_px(context_target, scale_x=1.0, scale_y=1.0)
                        # "combined" is the same port-mirrored dual-swath layout
                        # _extract_pixel_style_window builds -- reuse the same
                        # orientation fix (style="dual-swath" skips the
                        # already-applied horizontal mirror, only flips vertically).
                        combined, (context_target,) = _orient_pixel_window(combined, [context_target], channel_idx, "dual-swath")
                        fig = render_target_context(combined, context_target, box_size=(args.box_size, args.box_size), coordinate_system="pixel")
                        _figures.save_figure(fig, args.output / "context" / f"context_{t.target_id}.png", dpi=150)
                    else:
                        # Geometric/georef QA view (2026-09-28, real user
                        # ask): the same "whole loaded neighborhood, both
                        # channels pooled" shape as the pixel branch above,
                        # but resampled into real-world coordinates --
                        # reuses "dual-swath"'s own extraction (already
                        # pools both channels into one real-meters grid
                        # covering the swath), just with jitter_px=0 so the
                        # drawn box sits at the TRUE position, matching
                        # this feature's QA purpose (chip's own jitter is a
                        # training-data augmentation, not appropriate
                        # here). Uses --swath-size-px, same as single-swath/
                        # dual-swath chip styles, not a separate size knob.
                        # When dual-swath is itself being exported, the
                        # context reuses that window (rendered after the
                        # style loop below) instead of a second, identical
                        # griddata pass (~11s per target on
                        # a real survey) -- jittered, but the box
                        # still sits at the target's true position.
                        if "dual-swath" not in styles:
                            context_result = geo_window("dual-swath", args.swath_size_px, 0)
                            if context_result is None:
                                logger.warning(f"target {t.target_id}: context/{args.render_level} window unavailable, skipping context")
                            else:
                                grid, context_targets = context_result
                                fig = render_target_context(grid, context_targets[0], box_size=(args.box_size, args.box_size), coordinate_system=args.render_level,
                                                    resolution_m=context_targets[0].extra_attributes.get("window_resolution_m"))
                                _figures.save_figure(fig, args.output / "context" / f"context_{t.target_id}.png", dpi=150)

                # --n-augment: independent jitter (+ optional flip) draws
                # of the same target from the same loaded neighborhood --
                # the expensive load above happens once per target, only
                # the resample repeats.
                for aug_idx in range(n_augment):
                    for style in styles:
                        size_px = _style_size_px(style, args)
                        if args.render_level in _PIXEL_LEVELS:
                            result = _extract_pixel_style_window(
                                masked, t, channel_idx, n_samples_single, local_ping, lo, style, args.render_level,
                                size_px, args.jitter_px, rng, placed_by_channel, placed_combined,
                            )
                        else:
                            result = geo_window(style, size_px, args.jitter_px)
                        if result is None:
                            logger.warning(f"target {t.target_id}: {style}/{args.render_level} window unavailable, skipping")
                            continue
                        window, co_targets = result
                        if style == "dual-swath" and geo and args.with_context and aug_idx == 0:
                            primary = next((ct for ct in co_targets if ct.target_id == t.target_id), co_targets[0])
                            fig = render_target_context(window, primary, box_size=(args.box_size, args.box_size), coordinate_system=args.render_level,
                                                        resolution_m=primary.extra_attributes.get("window_resolution_m"))
                            _figures.save_figure(fig, args.output / "context" / f"context_{t.target_id}.png", dpi=150)
                        if box_size_m is not None:
                            # Meters placeholder for center-only picks (no
                            # measured bbox): same real extent at every
                            # resolution, unlike --box-size's fixed pixels.
                            co_targets = [
                                ct if "bbox_size_px" in ct.extra_attributes or not ct.extra_attributes.get("window_resolution_m")
                                else replace(ct, extra_attributes={**ct.extra_attributes, "bbox_size_px": (box_size_m / ct.extra_attributes["window_resolution_m"],) * 2})
                                for ct in co_targets
                            ]
                        flip_h = flip_v = False
                        if random_flip:
                            window, co_targets, flip_h, flip_v = _random_flip(window, co_targets, rng, edge_coords=geo)
                        file_name = _augmented_file_name(t.target_id, aug_idx, n_augment)
                        primary = next((ct for ct in co_targets if ct.target_id == t.target_id), None)
                        aug_log[style].append({
                            "file_name": file_name, "target_id": t.target_id, "augmentation": aug_idx,
                            "source_file": path.name, "freq_key": freq_key, "flip_h": flip_h, "flip_v": flip_v,
                            "target_row": None if primary is None else round(float(primary.extra_attributes["window_row"]), 2),
                            "target_col": None if primary is None else round(float(primary.sample), 2),
                            "resolution_m": None if primary is None else primary.extra_attributes.get("window_resolution_m"),
                            "n_targets": len(co_targets),
                        })
                        if stream:
                            emit(style, window, co_targets, t, freq_key, file_name, aug_idx)
                        else:
                            pending[style].append((window, co_targets, t, freq_key, file_name, aug_idx))

    total_written = 0
    for style in styles:
        if pending[style]:
            reference_stats = _reference_stats_for_scope(args.normalization_scope, [p[0] for p in pending[style]], args.pct_low, args.pct_high)
            for window, co_targets, primary, freq_key, file_name, aug_idx in pending[style]:
                emit(style, window, co_targets, primary, freq_key, file_name, aug_idx, reference_stats=reference_stats)
            pending[style] = []
        if not n_written[style]:
            logger.warning(f"{style}: no windows produced at render-level={args.render_level!r} -- skipping this style's COCO/YOLO export")
            continue

        style_dir = args.output / style
        thumbnails, captions, sheet_boxes = sheet[style]
        sheet_fig = contact_sheet(
            thumbnails, captions, boxes=sheet_boxes, box_size=(args.box_size, args.box_size),
            ncols=6, suptitle=f"{style}: {len(thumbnails)} chips" + (f" (first of {n_augment} draws each)" if n_augment > 1 else ""),
        )
        _figures.save_figure(sheet_fig, style_dir / "contact_sheet.png", dpi=150)
        sheet[style] = ([], [], [])

        if n_augment > 1 or random_flip:
            (style_dir / "augmentations.json").write_text(json.dumps(aug_log[style], indent=1))

        # write_coco()/write_yolo() have their own independent
        # min_confidence filter (default "verified", separate from the
        # target-list filter above) -- pass args.min_confidence through
        # explicitly, or a format whose targets are never "verified"
        # (e.g. Klein's .tgt, which is unconditionally "unknown") gets a
        # real crop for every target but zero COCO/YOLO annotations for
        # any of them, silently.
        coco = write_coco(coco_entries[style], style_dir / "targets.coco.json", box_size=(args.box_size, args.box_size), min_confidence=args.min_confidence, categories=categories)
        logger.info(f"{style}: COCO export: {len(coco['images'])} image(s), {len(coco['annotations'])} annotation(s) -> {style_dir / 'targets.coco.json'}")
        yolo = write_yolo(coco_entries[style], style_dir / "targets.yolo", box_size=(args.box_size, args.box_size), min_confidence=args.min_confidence, categories=categories)
        logger.info(f"{style}: YOLO export: {len(yolo['images'])} image(s) -> {style_dir / 'targets.yolo'}")
        logger.info(f"{style}: wrote {n_written[style]} chip(s) + contact sheet to {style_dir}")
        total_written += n_written[style]

    if total_written == 0:
        logger.error(f"No style produced any window at render-level={args.render_level!r} -- nothing exported")
        return 1
    return 0

def run_survey_window(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _figures
    from sidescanpy.viz.contact_sheet import contact_sheet

    args.output.mkdir(parents=True, exist_ok=True)
    survey = read_survey(args.ss_h5, chunks={"ping": 512})
    channel = survey.get_frequency(args.freq_key)
    ds = channel.dataset
    n_pings = ds.sizes["ping"]

    rng = np.random.default_rng(args.seed)

    n_samples = ds.sizes["sample"]

    raw_windows = []
    metas = []
    for i, center_ping in enumerate(np.linspace(0, n_pings - 1, args.n_windows + 2)[1:-1]):
        if args.mode == "full-channel":
            # Bound the ping neighborhood before loading+masking (real
            # files here run up to 14.8GB with every correction step's
            # arrays included) -- native window height == n_samples, plus
            # jitter room.
            half_span = n_samples // 2 + args.jitter_px + 50
            lo = max(0, int(center_ping) - half_span)
            hi = min(n_pings, int(center_ping) + half_span)
            masked = _mask_nadir_gap(ds.isel(ping=slice(lo, hi)).load())
            array_2d = masked["backscatter_intensity"].isel(channel=args.channel_idx).values
            local_ping = center_ping - lo
            try:
                window, extent = extract_full_channel(
                    array_2d, center_ping=float(local_ping), jitter_px=args.jitter_px, rng=rng,
                )
            except ValueError:
                logger.warning(
                    f"window {i} at ping {center_ping:.0f}: survey/channel too short for a square "
                    "full-channel window at this sample count, skipping"
                )
                continue
        elif args.mode == "geometric":
            sliced = _mask_nadir_gap(ds.isel(ping=slice(max(0, int(center_ping) - 100), int(center_ping) + 100)).load())
            try:
                window = extract_geometric_window(sliced, args.output_size_px, channel_idx=args.channel_idx)
            except ValueError:
                logger.warning(f"window {i} at ping {center_ping:.0f}: no usable geometry, skipping")
                continue
        elif args.mode == "georef":
            # extract_georef_window wraps resample_channel_to_grid, which
            # by design pools port+starboard as independent observations
            # into one grid (no per-channel selection) — unlike the other
            # two modes here, --channel-idx doesn't apply.
            sliced = _mask_nadir_gap(ds.isel(ping=slice(max(0, int(center_ping) - 100), int(center_ping) + 100)).load())
            try:
                raster = extract_georef_window(sliced, args.output_size_px)
                window = raster.dataset["backscatter"].values
            except (ValueError, KeyError):
                logger.warning(f"window {i} at ping {center_ping:.0f}: no usable georeferencing, skipping")
                continue
        else:
            raise ValueError(f"unknown --mode: {args.mode!r}")

        raw_windows.append(window)
        metas.append(i)

    if not raw_windows:
        logger.error("No windows extracted -- nothing to export")
        return 1

    # Percentile/gamma stretch on the native-resolution values *first* —
    # full-channel's native window can be far larger than output_size_px,
    # and stretching after downsampling measures percentiles against an
    # already-blurred, low-variance array. georef's own grid is already
    # sized near output_size_px by extract_georef_window, so this reorder
    # doesn't change its behavior, just applies consistently.
    reference_stats = _reference_stats_for_scope(args.normalization_scope, raw_windows, args.pct_low, args.pct_high)

    thumbnails, captions = [], []
    for i, window in zip(metas, raw_windows):
        img_8bit, _meta = to_uint8(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high, reference_stats=reference_stats)
        if img_8bit.shape != (args.output_size_px, args.output_size_px):
            img_8bit = resize_to(img_8bit, args.output_size_px)
            img_8bit = np.clip(img_8bit, 0, 255).astype(np.uint8)
        file_name = f"window_{args.mode}_{i:03d}.png"
        _save_grayscale(img_8bit, args.output / file_name)
        thumbnails.append(img_8bit)
        captions.append(f"no target ({args.mode})")

    sheet_fig = contact_sheet(thumbnails, captions, ncols=6, suptitle=f"{len(thumbnails)} {args.mode} windows")
    _figures.save_figure(sheet_fig, args.output / "contact_sheet.png", dpi=150)

    # No real target ground truth is correlated to this array (see module
    # docstring) — every window here is honestly a no-target negative, so
    # the COCO/YOLO export has zero annotations, demonstrating that path
    # rather than faking positive boxes.
    empty_entries: list[CocoImageEntry] = []
    write_coco(empty_entries, args.output / "empty.coco.json")
    logger.info(f"Wrote {len(raw_windows)} {args.mode} window(s) + contact sheet to {args.output}")
    return 0


# ============================================================
# Correction-step progression grid (uncorrected / dB / TVG / geometric / georef)
# ============================================================

_PROGRESSION_STEPS = ("uncorrected", "db", "tvg", "geometric", "georef")


def _sample_interval_m(ds) -> float:
    sound_vel = float(ds["sound_vel_ms"].values[0])
    if not np.isfinite(sound_vel) or sound_vel <= 0:
        sound_vel = 1500.0
    return sound_vel / (2.0 * float(ds.attrs["sample_rate_hz"]))


def _combine_channels_mirrored(port_2d: np.ndarray, starboard_2d: np.ndarray) -> np.ndarray:
    """Port (mirrored) + starboard side by side, nadir landing in the
    middle column -- both real pixel arrays, not a shared real-world
    coordinate system (that's what geometric/georef are for), so this is
    a plain array concatenation, not a resample.
    """
    return np.concatenate([port_2d[:, ::-1], starboard_2d], axis=1)


def _mask_bright_ping_rows(array_2d: np.ndarray, mad_k: float = 3.5) -> np.ndarray:
    """NaN out entire ping rows whose typical amplitude is an extreme
    outlier relative to the rest of the array.

    Klein SDF hardware runs active per-ping automatic gain control -- an
    occasional ping's overall gain jumps well above its neighbors,
    showing up as a uniformly bright, full-width horizontal streak.
    Because the effect multiplies an entire ping equally, a plain value
    clip only dampens the streak's brightness without removing it (the
    row stays visible, just less extreme) -- masking the whole outlier
    row is what actually removes it. Median/MAD-based (not mean/std),
    with the literature-backed ``mad_k=3.5`` (Iglewicz & Hoaglin
    "definite outlier" threshold) default.
    """
    row_level = np.nanmedian(array_2d, axis=1)
    finite = np.isfinite(row_level)
    if finite.sum() < 5:
        return array_2d
    med = np.nanmedian(row_level[finite])
    mad = np.nanmedian(np.abs(row_level[finite] - med))
    if mad == 0:
        return array_2d
    z = 0.6745 * (row_level - med) / mad
    bright_rows = np.abs(z) > mad_k
    if not bright_rows.any():
        return array_2d
    out = array_2d.copy()
    out[bright_rows, :] = np.nan
    return out


def _jitter_offset_m(jitter_px: int, resolution_m: float, rng: np.random.Generator) -> float:
    """A real-world jitter draw, same convention as ``extract_chip``/
    ``extract_full_channel``'s own pixel jitter (a uniform integer pixel
    count) but expressed in meters at whichever resolution the calling
    geometric/georef window is being resampled at."""
    if not jitter_px:
        return 0.0
    return float(rng.integers(-jitter_px, jitter_px + 1)) * resolution_m


def _jitter_latlon(lat: float, lon: float, north_jitter_m: float, east_jitter_m: float) -> tuple[float, float]:
    """Offset a ``(lat, lon)`` point by an exact real-world distance in
    UTM meters (auto-selected zone), for georef jitter -- ``center_lat``/
    ``center_lon`` are the only center sidescanpy.export.extract_georef_window
    accepts, so a meters-based jitter has to round-trip through UTM
    rather than applying a small-angle degrees approximation."""
    if north_jitter_m == 0.0 and east_jitter_m == 0.0:
        return lat, lon
    crs = utm_crs_for_lonlat(lon, lat)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_latlon = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    easting, northing = to_utm.transform(lon, lat)
    new_lon, new_lat = to_latlon.transform(easting + east_jitter_m, northing + north_jitter_m)
    return new_lat, new_lon


def _target_combined_col(t_sample: int, n_samples_single: int, channel_idx: int) -> int:
    """Where a target's own single-channel sample index lands in
    :func:`_combine_channels_mirrored`'s output column space."""
    if channel_idx == 0:
        return n_samples_single - 1 - t_sample
    return n_samples_single + t_sample


def run_progression(args: argparse.Namespace) -> None:
    """Build a 7(-ish)-target x 5-step progression grid (uncorrected / dB-converted
    / TVG-applied / geometric-corrected / georef-corrected) at either
    full-channel or chip scale, so the same real target is directly
    comparable across every correction step.
    """
    resample_method = getattr(args, "resample_method", "griddata")
    args.output.mkdir(parents=True, exist_ok=True)
    targets = read_targets_json(args.targets_json)
    if args.min_confidence != "unknown":
        targets = [t for t in targets if _CONFIDENCE_RANK[t.position_confidence] >= _CONFIDENCE_RANK[args.min_confidence]]
    if args.channel != "both":
        want = 0 if args.channel == "port" else 1
        targets = [t for t in targets if t.channel == want]

    index = build_position_index(args.ss_h5_dir)
    file_n_pings = {path: max(len(c["lat"]) for c in channels.values()) for path, channels in index.items()}

    matched = []
    for t in targets:
        m = correlate_target(t, index, args.max_match_dist_m)
        if m is None:
            continue
        path, freq_key, ping_idx, _dist_m = m
        if file_n_pings[path] < args.min_file_pings:
            continue
        if args.scale == "full" or args.match_full_scale_margin:
            n_samples_single = index[path][freq_key]["n_samples"]
            full_half_span = n_samples_single + args.jitter_px + 50
            if ping_idx < full_half_span or (file_n_pings[path] - ping_idx) < full_half_span:
                continue
        matched.append((t, path, freq_key, ping_idx))
        if len(matched) >= args.n_targets:
            break

    if len(matched) < args.n_targets:
        logger.warning(f"only found {len(matched)}/{args.n_targets} targets in a file with >= {args.min_file_pings} pings")

    rng = np.random.default_rng(args.seed)
    columns = []

    for t, path, freq_key, ping_idx in matched:
        channel_idx = t.channel if t.channel is not None else 0
        survey = read_survey(path, chunks={"ping": 512})
        ds_full = survey.get_frequency(freq_key).dataset
        n_pings = ds_full.sizes["ping"]
        n_samples_single = ds_full.sizes["sample"]

        geo_size_px = args.output_size_px if args.scale == "full" else args.geo_chip_size_px
        combine_sides = args.scale == "full" and args.channel == "both"

        if args.scale == "full":
            combined_n_samples = 2 * n_samples_single if combine_sides else n_samples_single
            half_span = combined_n_samples // 2 + args.jitter_px + 50
        else:
            half_span = args.chip_size_px // 2 + args.jitter_px + 50

        lo = max(0, ping_idx - half_span)
        hi = min(n_pings, ping_idx + half_span)
        masked = _mask_nadir_gap(ds_full.isel(ping=slice(lo, hi)).load())
        local_ping = ping_idx - lo
        mps = _sample_interval_m(masked)

        step_images: dict[str, np.ndarray] = {}
        step_positions: dict[str, tuple[float, float] | None] = {}

        if args.scale == "full":
            if combine_sides:
                raw_port = masked["uncorrected"].isel(channel=0).values
                raw_stbd = masked["uncorrected"].isel(channel=1).values
                raw_2d = _combine_channels_mirrored(raw_port, raw_stbd)
                intensity_port = masked["backscatter_intensity"].isel(channel=0).values
                intensity_stbd = masked["backscatter_intensity"].isel(channel=1).values
                intensity_2d = _combine_channels_mirrored(intensity_port, intensity_stbd)
                target_col = _target_combined_col(t.sample, n_samples_single, channel_idx)
            else:
                raw_2d = masked["uncorrected"].isel(channel=channel_idx).values
                intensity_2d = masked["backscatter_intensity"].isel(channel=channel_idx).values
                target_col = t.sample
            if args.mask_bright_rows:
                raw_2d = _mask_bright_ping_rows(raw_2d, mad_k=args.mask_bright_rows_k)
                intensity_2d = _mask_bright_ping_rows(intensity_2d, mad_k=args.mask_bright_rows_k)
            _rw, extent = extract_full_channel(raw_2d, center_ping=float(local_ping), jitter_px=args.jitter_px, rng=rng)
        else:
            raw_2d = masked["uncorrected"].isel(channel=channel_idx).values
            intensity_2d = masked["backscatter_intensity"].isel(channel=channel_idx).values
            target_col = t.sample
            if args.mask_bright_rows:
                raw_2d = _mask_bright_ping_rows(raw_2d, mad_k=args.mask_bright_rows_k)
                intensity_2d = _mask_bright_ping_rows(intensity_2d, mad_k=args.mask_bright_rows_k)
            _rw, extent = extract_chip(raw_2d, center_row=local_ping, center_col=t.sample, size_px=args.chip_size_px, jitter_px=args.jitter_px, rng=rng)

        row_sl = slice(extent.row_start, extent.row_start + extent.n_rows)
        col_sl = slice(extent.col_start, extent.col_start + extent.n_cols)
        raw_window = raw_2d[row_sl, col_sl]
        intensity_window = intensity_2d[row_sl, col_sl]
        db_window = apply_tvg_to_amplitude(raw_window, mps, params=TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0))
        target_local = (local_ping - extent.row_start, target_col - extent.col_start)

        for step, window in (("uncorrected", raw_window), ("db", db_window), ("tvg", intensity_window)):
            img, _meta = to_uint8(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high)
            pos = target_local
            if args.scale == "full":
                native_shape = window.shape
                img = np.clip(resize_to(img, args.output_size_px), 0, 255).astype(np.uint8)
                pos = rescale_target_position(target_local[0], target_local[1], native_shape, (args.output_size_px, args.output_size_px))
            step_images[step] = img
            step_positions[step] = pos

        along_dist = along_track_distance_m(masked, fill_nav_stalls=True)
        target_along_m = float(along_dist[local_ping])
        target_across_m = float(masked["across_track_m"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)

        ping_span_m = float(along_dist[-1] - along_dist[0]) if along_dist.size > 1 else 0.0
        ping_spacing_m = abs(ping_span_m) / max(along_dist.size - 1, 1) if along_dist.size > 1 else mps

        along_diffs = np.abs(np.diff(along_dist)) if along_dist.size > 1 else np.array([])
        nonzero_along_diffs = along_diffs[along_diffs > 0]
        gap_p95_m = float(np.percentile(nonzero_along_diffs, 95)) if nonzero_along_diffs.size else 0.0

        if args.scale == "full":
            if combine_sides:
                across_all = masked["across_track_m"].values
                geo_channel_idx = None
                geom_center_across_m = 0.0
            else:
                across_all = masked["across_track_m"].isel(channel=channel_idx).values
                geo_channel_idx = channel_idx
            across_finite = across_all[np.isfinite(across_all)]
            swath_width_m = float(across_finite.max() - across_finite.min()) if across_finite.size else geo_size_px * mps
            geo_resolution_m = max(swath_width_m / geo_size_px, ping_spacing_m, 1e-6)
            if not combine_sides:
                geom_center_across_m = float((across_finite.max() + across_finite.min()) / 2.0) if across_finite.size else 0.0
            geom_center_along_m = target_along_m + _jitter_offset_m(args.jitter_px, geo_resolution_m, rng)
        else:
            geo_resolution_m = max(mps, ping_spacing_m)
            geo_channel_idx = channel_idx
            geom_center_along_m = target_along_m + _jitter_offset_m(args.jitter_px, geo_resolution_m, rng)
            geom_center_across_m = target_across_m + _jitter_offset_m(args.jitter_px, geo_resolution_m, rng)

        # geo_search_radius_m: kept for the georef branch below, which
        # still uses extract_georef_window's own isotropic KDTree radius
        # search -- see _extract_geo_style_window's own comment for the
        # same three-way resample_method dispatch used here. Square
        # geo_size_px x geo_size_px output for every method, matching
        # _extract_geo_style_window's own (real user direction: no
        # non-square output, regardless of resample_method).
        geo_search_radius_m = max(1.5 * geo_resolution_m, gap_p95_m)

        try:
            if resample_method == "kdtree":
                geom_grid, geom_meta = extract_geometric_window(
                    masked, geo_size_px, resolution_m=geo_resolution_m,
                    center_along_m=geom_center_along_m, center_across_m=geom_center_across_m, channel_idx=geo_channel_idx,
                    search_radius_m=geo_search_radius_m,
                )
            elif resample_method == "binned":
                geom_grid, geom_meta = extract_geometric_window_binned(
                    masked, geo_size_px, resolution_m=geo_resolution_m,
                    center_along_m=geom_center_along_m, center_across_m=geom_center_across_m, channel_idx=geo_channel_idx,
                )
            else:
                geom_grid, geom_meta = extract_geometric_window_griddata(
                    masked, geo_size_px, resolution_m=geo_resolution_m,
                    center_along_m=geom_center_along_m, center_across_m=geom_center_across_m, channel_idx=geo_channel_idx,
                )
            geom_img, _ = to_uint8(geom_grid, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high)
            geom_pos = locate_point_geometric(target_along_m, target_across_m, geom_meta)
        except ValueError:
            geom_img = np.zeros((geo_size_px, geo_size_px), dtype=np.uint8)
            geom_pos = None
        step_images["geometric"] = geom_img
        step_positions["geometric"] = geom_pos

        try:
            target_lat = float(masked["latitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)
            target_lon = float(masked["longitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=t.sample).values)
            georef_north_jitter_m = _jitter_offset_m(args.jitter_px, geo_resolution_m, rng)
            georef_east_jitter_m = _jitter_offset_m(args.jitter_px, geo_resolution_m, rng)
            georef_center_lat, georef_center_lon = _jitter_latlon(target_lat, target_lon, georef_north_jitter_m, georef_east_jitter_m)
            georef_input = masked
            if args.channel != "both":
                excluded = 1 - channel_idx
                georef_input = masked.copy()
                georef_input["backscatter_intensity"] = georef_input["backscatter_intensity"].copy()
                georef_input["backscatter_intensity"][{"channel": excluded}] = np.nan
            georef_raster, georef_meta = extract_georef_window(
                georef_input, geo_size_px, resolution_m=geo_resolution_m, center_lat=georef_center_lat, center_lon=georef_center_lon,
                max_search_radius_m=geo_search_radius_m,
            )
            georef_grid = georef_raster.dataset["backscatter"].values
            if georef_grid.shape != (geo_size_px, geo_size_px):
                padded = np.full((geo_size_px, geo_size_px), np.nan)
                r = min(georef_grid.shape[0], geo_size_px)
                c = min(georef_grid.shape[1], geo_size_px)
                padded[:r, :c] = georef_grid[:r, :c]
                georef_grid = padded
            georef_img, _ = to_uint8(georef_grid, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high)
            georef_pos = locate_point_georef(target_lat, target_lon, georef_meta)
        except (ValueError, KeyError):
            georef_img = np.zeros((geo_size_px, geo_size_px), dtype=np.uint8)
            georef_pos = None
        step_images["georef"] = georef_img
        step_positions["georef"] = georef_pos

        for step in _PROGRESSION_STEPS:
            file_name = f"{t.target_id}_{step}.png"
            _save_grayscale(step_images[step], args.output / file_name)

        columns.append({
            "target_id": t.target_id,
            "source_file": path.name,
            "positions": {k: (list(v) if v is not None else None) for k, v in step_positions.items()},
        })
        logger.info(f"target {t.target_id}: wrote all {len(_PROGRESSION_STEPS)} steps ({path.name})")

    with open(args.output / "progression_meta.json", "w") as f:
        json.dump({"scale": args.scale, "steps": list(_PROGRESSION_STEPS), "columns": columns}, f, indent=2)
    logger.info(f"Wrote progression grid ({args.scale} scale, {len(columns)} targets) to {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    common = dict(gamma=1.0, pct_low=1.0, pct_high=99.0)

    p_wf = subparsers.add_parser("waterfall", help="Sample/ping or ground-range/along-track waterfall PNGs, spanning the whole survey")
    p_wf.add_argument("--ss-h5-dir", required=True, type=Path)
    p_wf.add_argument("--output-dir", required=True, type=Path)
    p_wf.add_argument("--frequency", type=str, default="all")
    p_wf.add_argument(
        "--window-pings", type=int, default=_DEFAULT_SEGMENT_SIZE,
        help="Pings per window. Windows are cut from the whole survey (every .ss.h5 file in "
             "--ss-h5-dir, lazily concatenated in filename order) -- not restarted per file.",
    )
    p_wf.add_argument(
        "--overlap-pings", type=int, default=0,
        help="Pings shared between consecutive windows (must be < --window-pings). 0 (default): "
             "the old non-overlapping behavior.",
    )
    p_wf.add_argument(
        "--overview-rows", type=int, default=None,
        help="Also write one overview.png per frequency: the whole survey compressed to "
             "approximately this many rows (stride computed as total_pings // this value, so "
             "every survey's overview comes out about the same size regardless of length), with "
             "a marker line at every file boundary. Omit to skip the overview.",
    )
    p_wf.add_argument(
        "--value-var", choices=["uncorrected", "radiometric"], type=uncorrected_alias("raw"), default="radiometric",
        help="uncorrected = uncorrected signal, inverted from processing_history (no TVG); radiometric = backscatter_intensity (TVG-corrected, "
             "the old default). Combine with --range-mode ground for a geometrically-axed view.",
    )
    p_wf.add_argument("--range-mode", choices=["slant", "ground"], default="slant", help="ground requires geometric ranges already computed (see sidescan-correct --geometric)")
    p_wf.add_argument("--x-axis", choices=["sample", "ground_range", "across_track"], default="sample", help="Ignored when --range-mode ground")
    p_wf.add_argument("--y-axis", choices=["ping", "time", "along_track"], default="ping", help="Ignored when --range-mode ground")
    p_wf.add_argument("--cmap", type=str, default="copper")
    p_wf.add_argument("--vmin", type=float, default=None)
    p_wf.add_argument("--vmax", type=float, default=None)
    p_wf.add_argument("--vmin-pct", type=float, default=2.0)
    p_wf.add_argument("--vmax-pct", type=float, default=98.0)
    p_wf.add_argument("--gamma", type=float, default=2.5)
    p_wf.add_argument("--dpi", type=int, default=100)
    p_wf.add_argument(
        "--targets-json", type=Path, default=None,
        help="Optional: draw a box + label at each matched target's real position. Only supported "
             "with the default sample/ping axes (--range-mode slant, default --x-axis/--y-axis).",
    )
    p_wf.add_argument("--max-match-dist-m", type=float, default=25.0, help="Reject a target/ping match farther than this (meters); only used with --targets-json")
    p_wf.add_argument("--box-size", type=float, default=40.0, help="Drawn box size in pixels; only used with --targets-json")
    p_wf.set_defaults(func=run_waterfall)

    p_grid = subparsers.add_parser("grid", help="Gridded-raster PNGs on an auto-selected UTM grid")
    p_grid.add_argument("--ss-h5-dir", required=True, type=Path)
    p_grid.add_argument("--output-dir", required=True, type=Path)
    p_grid.add_argument("--frequency", type=str, default="all")
    p_grid.add_argument("--segment-size", type=int, default=_DEFAULT_SEGMENT_SIZE)
    p_grid.add_argument("--grid-resolution-m", type=float, default=1.0)
    p_grid.add_argument("--grid-method", choices=["mean", "median", "max", "nearest", "idw"], default="mean")
    p_grid.add_argument("--cmap", type=str, default="copper")
    p_grid.add_argument("--vmin", type=float, default=None)
    p_grid.add_argument("--vmax", type=float, default=None)
    p_grid.add_argument("--vmin-pct", type=float, default=2.0)
    p_grid.add_argument("--vmax-pct", type=float, default=98.0)
    p_grid.add_argument("--gamma", type=float, default=2.5)
    p_grid.add_argument("--dpi", type=int, default=100)
    p_grid.add_argument("--show-land", action="store_true", help="Draw land/water context behind the raster (requires the 'maps' extra)")
    p_grid.add_argument("--show-locator", action="store_true", help="Add a zoomed-out locator inset (requires the 'maps' extra)")
    p_grid.set_defaults(func=run_grid)

    p_track = subparsers.add_parser("track", help="Full-survey nadir track + scan-lines, optionally with a target-catalog overlay")
    p_track.add_argument("--ss-h5-dir", required=True, type=Path)
    p_track.add_argument("--output-dir", required=True, type=Path)
    p_track.add_argument("--frequency", type=str, default="all")
    p_track.add_argument("--targets-json", type=Path, default=None, help="Optional: overlay target-catalog positions")
    p_track.add_argument("--scanline-stride", type=int, default=50)
    p_track.add_argument("--dpi", type=int, default=150)
    p_track.add_argument("--show-land", action="store_true", help="Draw land/water context behind the track (requires the 'maps' extra)")
    p_track.add_argument("--show-locator", action="store_true", help="Add a zoomed-out locator inset (requires the 'maps' extra)")
    p_track.set_defaults(func=run_track)

    p_chip = subparsers.add_parser("chip", help="Real, jittered target-positive chips, position-correlated to the real survey array")
    p_chip.add_argument("--targets-json", required=True, type=Path)
    p_chip.add_argument("--ss-h5-dir", required=True, type=Path, help="Directory of already-corrected .ss.h5 files to search for each target's real position")
    p_chip.add_argument("--max-match-dist-m", type=float, default=25.0, help="Reject a target/ping match farther than this (meters)")
    p_chip.add_argument("--output", required=True, type=Path)
    p_chip.add_argument(
        "--styles", nargs="+", choices=list(_CHIP_STYLES), default=list(_CHIP_STYLES),
        help="Which chip style(s) to export as parallel COCO/YOLO datasets under output/<style>/: "
             "'chip' (tight single-channel crop, --size-px), 'single-swath' (one full channel, square), "
             "'dual-swath' (both channels combined, square, port+starboard). Default: all three.",
    )
    p_chip.add_argument(
        "--render-level", choices=list(_RENDER_LEVELS), type=uncorrected_alias("raw"), default="radiometric",
        help="Processing level every style renders from this run (applies to both COCO and YOLO): "
             "'uncorrected' (uncorrected signal, linear, no TVG), 'db' (20*log10 of it, no TVG), "
             "'radiometric' (TVG-corrected, the old default), 'geometric' (across/along-track meters -- requires --geometric already run via "
             "sidescan-correct), 'georef' (lat/lon-projected meters -- requires --georef already run). "
             "Under geometric/georef every style's image is a real size_px*resolution_m meters square, "
             "not a pixel crop.",
    )
    p_chip.add_argument("--size-px", type=int, default=400, help="Output size (both axes) for the 'chip' style")
    p_chip.add_argument("--swath-size-px", type=int, default=512, help="Output size (both axes) for the 'single-swath'/'dual-swath' styles")
    p_chip.add_argument(
        "--resolution-m", type=float, default=None,
        help="Real-world meters per output pixel for 'geometric'/'georef' render levels, applied to "
             "every style. When set, used directly instead of the per-target auto-calculated value "
             "(swath width / output size for single-swath/dual-swath, native sample/ping spacing for "
             "chip) -- that auto-calculation has a known failure mode: it derives ping spacing from an "
             "average over a wide ping neighborhood, so a target sitting in a locally slow/dense-ping "
             "region (e.g. where the vessel slowed for a closer look) can get an inflated, too-coarse "
             "resolution that makes the whole crop look zoomed far out. Set this from sidescan-analyze's "
             "findings for the dataset, not by trial and error. No effect (logged, not an error) under "
             "'uncorrected'/'radiometric' render level, which crops in native pixel space and has no resolution_m "
             "concept at all.",
    )
    p_chip.add_argument(
        "--value-var", choices=["uncorrected", "db", "radiometric"], type=uncorrected_alias("raw"), default="radiometric",
        help="uncorrected signal vs. dB (20*log10 of it, no TVG) vs. backscatter_intensity for 'geometric'/'georef' render levels (chip "
             "styles AND --with-context, in either coordinate mode) -- independent of --render-level, "
             "which already fully determines this for 'uncorrected'/'radiometric' render levels (no effect there, "
             "logged not an error). Added 2026-09-28 so coordinate system and value processing are "
             "independently selectable, instead of --render-level conflating both.",
    )
    p_chip.add_argument(
        "--resample-method", choices=["kdtree", "binned", "griddata"], default="griddata",
        help="Which resampling function fills the 'geometric' render level's square, fixed-resolution_m "
             "grid (no effect on 'georef', which always uses extract_georef_window's own isotropic KDTree "
             "radius search, or on 'uncorrected'/'radiometric', which have no resolution_m concept). "
             "'kdtree' (extract_geometric_window): real physical average of nearby measurements, zero "
             "fabricated values, some directional blur at fine resolution. 'binned' "
             "(extract_geometric_window_binned): real measured means only, zero fabricated values, but a "
             "cell with no real sample inside it is left as NaN -- can be the majority of the grid at a "
             "tight resolution_m on real data. 'griddata' (extract_geometric_window_griddata, "
             "DEFAULT): scipy.interpolate.griddata full interpolation, no gaps inside the recorded swath (cells with no real sample nearby, e.g. past max range, stay no-data) -- real, chosen tradeoff: "
             "at a resolution finer than real along-track ping spacing, most pixels are interpolated from "
             "nearby real points, not individually measured. Real-data comparison of all three (2026-09-29) "
             "is in sidescanpy's CHANGELOG.md.",
    )
    p_chip.add_argument("--box-size", type=float, default=40.0, help="COCO/YOLO annotation box size AND (with --with-context) the drawn box size, in pixels")
    p_chip.add_argument(
        "--with-context", action="store_true",
        help="Also save context/context_<id>.png -- the full loaded ping neighborhood, BOTH channels "
             "combined (port+starboard), with a real box drawn at the target's true position, via "
             "sidescanpy.viz.render_target_context. Independent of --styles -- but, as of 2026-09-28, "
             "follows --render-level's own coordinate choice (native pixel space for 'uncorrected'/'radiometric', "
             "a real-meters resampled grid -- same shape as the 'dual-swath' style -- for "
             "'geometric'/'georef') and --value-var's uncorrected/radiometric choice, rather than always being a "
             "fixed radiometric pixel-space view.",
    )
    p_chip.add_argument("--jitter-px", type=int, default=0)
    p_chip.add_argument(
        "--n-augment", type=int, default=1,
        help="Independent jitter draws per target per style (each its own image, target_<id>_a<k>.png when > 1). "
             "The target neighborhood is loaded once per target; only the resample repeats. With --jitter-px near "
             "half the image size, the target lands anywhere in the frame across draws.",
    )
    p_chip.add_argument(
        "--random-flip", action="store_true",
        help="Flip each image horizontally and vertically, each independently with probability 0.5, with every "
             "annotation remapped to match. Draws are recorded per image in <style>/augmentations.json.",
    )
    p_chip.add_argument(
        "--across-axis", choices=["ground", "slant"], default="ground",
        help="'geometric' render level only: across-track axis of the real-meters grid. 'ground' (default) = "
             "across_track_m, ground-projected. 'slant' = signed slant range, no ground projection (water column "
             "kept), still meters so --resolution-m means the same real extent on both axes.",
    )
    p_chip.add_argument(
        "--box-size-m", type=float, default=None,
        help="geometric/georef only: annotation box size in meters for targets with no measured bbox (center-only "
             "picks), converted per image by its own resolution -- same real extent at every scale. Targets with a "
             "measured bbox keep it. Default: use --box-size pixels.",
    )
    p_chip.add_argument(
        "--swath-fill", action="store_true",
        help="'geometric' level, single-swath/dual-swath without --resolution-m: size the frame to the across-track "
             "extent the data fills (the far edge most pings reach; dual swath spans port to starboard, not "
             "centered on nadir), at square pixels -- no black side margins. Default keeps the old frame: the "
             "farthest single ping's width, nadir-centered, never finer than the ping spacing.",
    )
    p_chip.add_argument("--gamma", type=float, default=common["gamma"])
    p_chip.add_argument(
        "--bit-depth", type=int, choices=[8, 16], default=8,
        help="Dataset image bit depth (single-channel grayscale PNG). 16 keeps the same percentile "
             "stretch and --gamma but 65536 gray levels instead of 256; contact sheets stay 8-bit. "
             "Check that the training data loader reads 16-bit PNGs unchanged -- many default to "
             "converting to 8-bit on load.",
    )
    p_chip.add_argument("--pct-low", type=float, default=common["pct_low"])
    p_chip.add_argument("--pct-high", type=float, default=common["pct_high"])
    p_chip.add_argument("--normalization-scope", choices=["image", "shared"], default="image")
    p_chip.add_argument("--min-confidence", choices=["unknown", "approximate", "verified"], default="verified")
    p_chip.add_argument(
        "--categories", default=None,
        help="Comma-separated ordered class names (COCO ids 1..N, YOLO indices 0..N-1) so ids stay stable across separately-exported datasets; "
             "a target whose category is not listed is an error. Default: sorted set of target categories present.",
    )
    p_chip.add_argument("--seed", type=int, default=None)
    p_chip.set_defaults(func=run_chip)

    p_sw = subparsers.add_parser("survey-window", help="Full-channel / geometric / georef windows, no-target negatives")
    p_sw.add_argument("--ss-h5", required=True, type=Path)
    p_sw.add_argument("--freq-key", required=True, help="e.g. hz_1495000")
    p_sw.add_argument(
        "--channel-idx", type=int, default=0, choices=[0, 1],
        help="Port(0)/starboard(1) for full-channel and geometric modes; ignored for georef "
        "(resample_channel_to_grid pools both sides by design)",
    )
    p_sw.add_argument("--mode", required=True, choices=["full-channel", "geometric", "georef"])
    p_sw.add_argument("--output", required=True, type=Path)
    p_sw.add_argument("--output-size-px", type=int, default=400)
    p_sw.add_argument("--n-windows", type=int, default=5)
    p_sw.add_argument("--jitter-px", type=int, default=0)
    p_sw.add_argument("--gamma", type=float, default=common["gamma"])
    p_sw.add_argument("--pct-low", type=float, default=common["pct_low"])
    p_sw.add_argument("--pct-high", type=float, default=common["pct_high"])
    p_sw.add_argument("--normalization-scope", choices=["image", "shared"], default="image")
    p_sw.add_argument("--seed", type=int, default=None)
    p_sw.set_defaults(func=run_survey_window)

    p_prog = subparsers.add_parser(
        "progression",
        help="7-target x 5-step grid (uncorrected / dB / TVG / geometric / georef), one shared jitter draw per target",
    )
    p_prog.add_argument("--targets-json", required=True, type=Path)
    p_prog.add_argument("--ss-h5-dir", required=True, type=Path)
    p_prog.add_argument("--max-match-dist-m", type=float, default=25.0)
    p_prog.add_argument("--output", required=True, type=Path)
    p_prog.add_argument("--scale", required=True, choices=["full", "chip"])
    p_prog.add_argument("--n-targets", type=int, default=7)
    p_prog.add_argument("--chip-size-px", type=int, default=400)
    p_prog.add_argument("--output-size-px", type=int, default=400)
    p_prog.add_argument("--geo-chip-size-px", type=int, default=200)
    p_prog.add_argument("--min-file-pings", type=int, default=10000)
    p_prog.add_argument("--jitter-px", type=int, default=0)
    p_prog.add_argument("--gamma", type=float, default=common["gamma"])
    p_prog.add_argument("--pct-low", type=float, default=common["pct_low"])
    p_prog.add_argument("--pct-high", type=float, default=common["pct_high"])
    p_prog.add_argument("--min-confidence", choices=["unknown", "approximate", "verified"], default="verified")
    p_prog.add_argument("--match-full-scale-margin", action="store_true")
    p_prog.add_argument("--channel", choices=["port", "starboard", "both"], default="both")
    p_prog.add_argument("--mask-bright-rows", action="store_true")
    p_prog.add_argument("--mask-bright-rows-k", type=float, default=3.5)
    p_prog.add_argument(
        "--resample-method", choices=["kdtree", "binned", "griddata"], default="griddata",
        help="Which resampling function fills the geometric step's square, fixed-resolution grid -- "
             "same three options/tradeoffs as sidescan-export chip's own --resample-method (see that "
             "flag's help text). No effect on the georef step.",
    )
    p_prog.add_argument("--seed", type=int, default=None)
    p_prog.set_defaults(func=run_progression)

    all_subparsers = (p_wf, p_grid, p_track, p_chip, p_sw, p_prog)
    for p in all_subparsers:
        p.add_argument("--config", type=Path, default=None, help="YAML or JSON file seeding this subcommand's defaults -- explicit flags still override")
        p.add_argument("--strict-config", action="store_true", help="Treat config-file keys this subcommand does not recognize as an error instead of a warning")
        # default=SUPPRESS: see sidescanpy/cli/convert.py's matching
        # block for why a plain default=False here would silently
        # overwrite a real -v already set before the subcommand.
        add_logging_args(p, subcommand=True)

    add_logging_args(parser)

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        config = load_config(config_path)
        apply_config_to_subparsers(all_subparsers, config, argv)

    args = parser.parse_args(argv)

    setup_logging_from_args(args)

    if args.func is not run_progression and not require_matplotlib(f"sidescan-export {args.subcommand}"):
        return 1

    result = args.func(args)
    logger.info(f"export {args.subcommand} complete")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
