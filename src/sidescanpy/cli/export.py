"""
``sidescan-export``: waterfalls, gridded rasters, survey tracks, and
configurable annotated-dataset exports (chips / full-channel windows /
processing-stage progression grids) from already-corrected ``.ss.h5``
files.

Replaces the old per-project ``03_track_maps.py``, ``05_waterfalls.py``,
``07_ground_waterfalls.py``, ``09_grid.py``, ``12_track_with_targets.py``
(confirmed byte-identical between `project-b` and `project-a`'s own
copies except docstring text) and promotes
``project-a/scripts/14_export_dataset.py``'s ``chip``/``survey-window``/
``progression`` subcommands essentially unchanged — that script was
already format-agnostic and cross-project proven (run directly against
`project-b` data via CLI args, no copy, no code change) before this
promotion.

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
  ``--render-level`` selects what every style renders from: ``raw``/
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
- ``progression`` — a 5-stage (raw / dB / TVG / geometric / georef) x
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
from dataclasses import replace
from pathlib import Path

import pyproj

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr
from PIL import Image

from sidescanpy.cli._common import resolve_freq_keys
from sidescanpy.cli._config import apply_config_defaults, find_config_path, load_config
from sidescanpy.export import (
    ReferenceStats,
    WindowExtent,
    compute_reference_stats,
    extract_chip,
    extract_full_channel,
    extract_geometric_window,
    extract_georef_window,
    locate_point_geometric,
    locate_point_georef,
    rescale_target_position,
    resize_to,
    targets_within_window,
    to_uint8,
)
from sidescanpy.analysis.investigation import mask_nadir_gap
from sidescanpy.grids import ResamplingMethod, bbox_from_channel_dataset, resample_channel_to_grid, utm_grid_from_wgs84_bbox
from sidescanpy.grids.definition import utm_crs_for_lonlat
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.radiometric import TVGParams, apply_tvg_to_amplitude
from sidescanpy.targets.core import project_target_to_frequency
from sidescanpy.targets.interchange import CocoImageEntry, write_coco, write_yolo
from sidescanpy.targets.io import read_targets_json
from sidescanpy.viz import plot_survey_overview, plot_waterfall, resolve_waterfall_axes
from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context
from sidescanpy.viz.gridded import export_gridded_png
from sidescanpy.viz.targets import export_track_with_targets_png
from sidescanpy.viz.track import build_survey_track, export_track_and_scanlines_png
from sidescanpy.viz.waterfall import along_track_distance_m

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"unknown": 0, "approximate": 1, "verified": 2}
_DEFAULT_SEGMENT_SIZE = 2000


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


def _export_waterfall_png(ds, output_path: Path, *, x_axis, y_axis, ping_offset, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, dpi, boxes=None, box_size=40.0, value_var="backscatter_intensity"):
    """``boxes``, if given: list of ``(target, ping_idx)`` (global ping
    index within whatever ``ds`` this call was handed) to draw a red box
    + target_id label for, on top of the rendered waterfall. Supports
    both the default sample/ping axes and ground mode's across_track/
    along_track axes -- see ``_target_waterfall_xy``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        plot_waterfall(
            ds, x_axis=x_axis, y_axis=y_axis, ping_offset=ping_offset,
            cmap=cmap, vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
            gamma=gamma, ax=ax, value_var=value_var,
        )
        boxes = boxes or []
        # Computed once per window, not per box -- along_track_distance_m
        # does a real pyproj.Geod.inv call over every ping in ds.
        along_track = along_track_distance_m(ds) if boxes and y_axis == "along_track" else None
        for target, ping_idx in boxes:
            xy = _target_waterfall_xy(target, ds, ping_idx, ping_offset, x_axis, y_axis, along_track)
            if xy is None:
                continue
            x_center, y_center, x_scale, y_scale = xy
            half_x = (box_size / 2.0) * x_scale
            half_y = (box_size / 2.0) * y_scale
            ax.add_patch(Rectangle((x_center - half_x, y_center - half_y), 2 * half_x, 2 * half_y, fill=False, edgecolor="red", linewidth=1.2))
            ax.annotate(str(target.target_id), (x_center - half_x, y_center - half_y), color="red", fontsize=11, fontweight="bold", xytext=(2, 2), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _concat_survey_frequency(files: list[Path], freq_key: str, chunks: dict) -> tuple[xr.Dataset | None, dict]:
    """Lazily concatenate ``freq_key``'s channel across every file in
    ``files`` (sorted, so typically time order) into one dask-backed
    dataset spanning the whole survey -- no per-file restart.

    Returns ``(concatenated_ds, file_start_ping)`` where
    ``file_start_ping[path]`` is that file's own first ping's position in
    the concatenated dataset's global ping axis (0 for the first file).
    ``concatenated_ds`` is ``None`` if no file has this frequency at all.
    """
    datasets = []
    file_start_ping: dict[Path, int] = {}
    cumulative = 0
    for path in files:
        survey = read_survey(path, chunks=chunks)
        if freq_key not in survey.frequencies:
            continue
        ds = survey.get_frequency(freq_key).dataset
        file_start_ping[path] = cumulative
        cumulative += ds.sizes["ping"]
        datasets.append(ds)
    if not datasets:
        return None, {}
    return xr.concat(datasets, dim="ping"), file_start_ping


def run_waterfall(args: argparse.Namespace) -> int:
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
        index = _build_position_index(args.ss_h5_dir)

    files = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1
    logger.info(f"Found {len(files)} .ss.h5 file(s) in {args.ss_h5_dir}")

    stride = args.window_pings - args.overlap_pings
    if stride <= 0:
        logger.error(f"--overlap-pings ({args.overlap_pings}) must be smaller than --window-pings ({args.window_pings})")
        return 1

    value_var = "backscatter_raw" if args.value_var == "raw" else "backscatter_intensity"

    n_written = 0
    # Correlate every target once (position-based, format-agnostic), then
    # re-derive its position on *each* frequency below -- its own native
    # mark where it was actually correlated, or a physical projection
    # (same ping, slant-range-equivalent sample) onto every other
    # frequency of that same file, so a target shows up on every
    # rendered frequency, not just the one it happened to be picked on.
    matched = []
    for t in targets:
        m = _correlate_target(t, index, args.max_match_dist_m)
        if m is not None:
            matched.append((t,) + m)  # (t, path, native_freq_key, local_ping, dist_m)
    if targets:
        logger.info(f"Overlaying {len(matched)}/{len(targets)} target(s) matched to a real ping")

    ds_first = read_survey(files[0], chunks={"ping": 512})
    freq_keys = resolve_freq_keys(ds_first.frequencies, args.frequency)

    for freq_key in freq_keys:
        ds_all, file_start_ping = _concat_survey_frequency(files, freq_key, chunks={"ping": 512})
        if ds_all is None:
            logger.warning(f"{freq_key}: no file has this frequency, skipping")
            continue
        n_pings_total = ds_all.sizes["ping"]

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
            fig, ax = plt.subplots(figsize=(14, 8))
            try:
                plot_survey_overview(
                    ds_all, ping_stride=overview_stride, break_ping_indices=breaks,
                    x_axis=x_axis, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                    vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma, ax=ax,
                    value_var=value_var,
                )
                fig.tight_layout()
                fig.savefig(out_path, dpi=args.dpi)
                n_written += 1
                logger.info(f"{freq_key}: wrote overview.png ({n_pings_total} pings, 1:{overview_stride} compressed)")
            finally:
                plt.close(fig)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_gridded_png(
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
        # (found via a real field OOM report; every other exporter in
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
    paths = sorted(args.ss_h5_dir.glob("*.ss.h5"))
    if not paths:
        logger.error(f"No .ss.h5 files found in {args.ss_h5_dir}")
        return 1
    logger.info(f"Loading {len(paths)} .ss.h5 file(s) (lazy)")
    surveys = {p: read_survey(p, chunks={"ping": 2000}) for p in paths}

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
                export_track_with_targets_png(
                    track, targets, str(out_path), dpi=args.dpi,
                    show_land=args.show_land, show_locator=args.show_locator,
                )
            else:
                export_track_and_scanlines_png(
                    track["track_lat"], track["track_lon"],
                    track["scan_port_lat"], track["scan_port_lon"],
                    track["scan_stbd_lat"], track["scan_stbd_lon"],
                    str(out_path), dpi=args.dpi,
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
# (promoted from project-a/scripts/14_export_dataset.py, unchanged)
# ============================================================

def _save_grayscale(img_8bit: np.ndarray, path: Path) -> None:
    """Save a true single-channel 8-bit grayscale PNG.

    Deliberately not ``matplotlib.pyplot.imsave`` — even with
    ``cmap="gray"`` it always writes RGB(A), which is redundant-channel
    grayscale, not the single-channel 8-bit file the dataset spec calls
    for. ``PIL.Image`` mode ``"L"`` writes the real thing.
    """
    Image.fromarray(img_8bit, mode="L").save(path)


def _reference_stats_for_scope(scope: str, arrays: list[np.ndarray], pct_low: float, pct_high: float) -> ReferenceStats | None:
    if scope == "image":
        return None
    if scope == "shared":
        return compute_reference_stats(*arrays, pct_low=pct_low, pct_high=pct_high)
    raise ValueError(f"unknown --normalization-scope: {scope!r}")


def _build_position_index(ss_h5_dir: Path) -> dict[Path, dict]:
    """For every ``.ss.h5`` in ``ss_h5_dir``, cache each frequency
    channel's per-ping lat/lon and computed sample interval (meters per
    sample, at the standard 1500 m/s fallback whenever ``sound_vel_ms``
    isn't usable — matching ``compute_geometric_ranges``'s own fallback)
    — used to locate a cataloged target's real position and match it to
    the right frequency channel.
    """
    index = {}
    for path in sorted(ss_h5_dir.glob("*.ss.h5")):
        # chunks= keeps this lazy (dask-backed) -- large surveys' .ss.h5
        # files carry raw+intensity+geometric+georef arrays together
        # (10x+ the base backscatter size); read_survey(chunks=None)
        # loads that eagerly, which is real memory risk on a
        # multi-gigabyte file when only tiny per-ping lat/lon is needed
        # here.
        survey = read_survey(path, chunks={"ping": 512})
        channels = {}
        for freq_key, ch in survey.freq_channels.items():
            ds = ch.dataset
            sound_vel = float(ds["sound_vel_ms"].values[0])
            if not np.isfinite(sound_vel) or sound_vel <= 0:
                sound_vel = 1500.0
            sample_interval_m = sound_vel / (2.0 * float(ds.attrs["sample_rate_hz"]))
            channels[freq_key] = {
                "lat": ds["latitude"].values.astype(float),
                "lon": ds["longitude"].values.astype(float),
                "sample_interval_m": sample_interval_m,
                "n_samples": ds.sizes["sample"],
            }
        index[path] = channels
    return index


def _match_frequency(channels: dict, target_mps: float | None) -> str:
    if target_mps is None:
        return next(iter(channels))
    return min(channels, key=lambda fk: abs(channels[fk]["sample_interval_m"] - target_mps))


def _correlate_target(target, index: dict[Path, dict], max_dist_m: float):
    """Find ``(ss_h5_path, freq_key, ping_idx, dist_m)`` for ``target`` by
    matching its recorded lat/lon against every candidate file+frequency's
    own per-ping track. Returns ``None`` if nothing lands within
    ``max_dist_m`` (or the target has no recorded position at all).
    """
    if target.latitude is None or target.longitude is None:
        return None
    target_mps = target.extra_attributes.get("ping_meters_per_sample")
    best = None
    for path, channels in index.items():
        freq_key = _match_frequency(channels, target_mps)
        c = channels[freq_key]
        d_m = np.sqrt((c["lat"] - target.latitude) ** 2 + (c["lon"] - target.longitude) ** 2) * 111000.0
        idx = int(np.argmin(d_m))
        if best is None or d_m[idx] < best[3]:
            best = (path, freq_key, idx, float(d_m[idx]))
    if best is None or best[3] > max_dist_m:
        return None
    return best


_CHIP_STYLES = ("chip", "single-swath", "dual-swath")
_RENDER_LEVELS = ("raw", "radiometric", "geometric", "georef")


def _style_size_px(style: str, args: argparse.Namespace) -> int:
    return args.size_px if style == "chip" else args.swath_size_px


def _orient_pixel_window(window: np.ndarray, co_targets: list, channel_idx: int, style: str) -> tuple[np.ndarray, list]:
    """Match ``targets.core.orient_for_display``'s two display
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
) -> list:
    """Real-world equivalent of ``targets_within_window`` for a
    geometric window: which of ``candidates`` (``(Target, global_ping_idx)``
    pairs, as gathered in ``run_chip``'s ``items_by_channel``) physically
    fall inside this specific window's along/across bounds, translated to
    window-local ``(row, col)`` via ``locate_fn`` (pre-flip -- see
    ``_flip_geo_vertical``). Mirrors ``_extract_geo_style_window``'s own
    per-target lookup, so a bad/out-of-range candidate is skipped, not a
    crash.
    """
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
        if 0 <= row < size_px and 0 <= col < size_px:
            included.append(replace(candidate_t, sample=col, extra_attributes={**candidate_t.extra_attributes, "window_row": row}))
    return included


def _flip_geo_vertical(grid: np.ndarray, positioned_targets: list, size_px: int) -> tuple[np.ndarray, list]:
    """Match Discover's ping-order display convention (see
    ``_orient_pixel_window``'s docstring for the raw-pixel equivalent) for
    a geometric window: ``extract_geometric_window`` builds row 0 = the
    minimum along-track distance (ascending ping order, top-down), the
    opposite of Discover's own display. ``positioned_targets`` must
    already carry window-local ``sample``/``extra_attributes["window_row"]``
    (e.g. from ``locate_point_geometric`` via ``_geo_targets_within_window``).
    """
    grid = grid[::-1, :]
    positioned_targets = [
        replace(pt, extra_attributes={**pt.extra_attributes, "window_row": size_px - pt.extra_attributes["window_row"]})
        for pt in positioned_targets
    ]
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


def _extract_pixel_style_window(
    masked, t, channel_idx: int, n_samples_single: int, local_ping: int, lo: int, style: str, level: str,
    size_px: int, jitter_px: int, rng: np.random.Generator, placed_by_channel: dict, placed_combined: list,
) -> tuple[np.ndarray, list] | None:
    """``raw``/``radiometric`` (sample/ping pixel space) window + the
    co-targets that land inside it, for one target/style. Returns
    ``None`` if this level's backscatter variable isn't present on
    ``masked`` (e.g. ``radiometric`` requested against a never-TVG'd
    file), or if ``masked``'s own ping neighborhood is too short for a
    full-swath style's native square window (a short source file/segment
    -- see ``extract_full_channel``'s own docstring) -- caller logs and
    skips, doesn't crash the whole run.
    """
    value_var = "backscatter_raw" if level == "raw" else "backscatter_intensity"
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
    if native_shape != (size_px, size_px):
        co_targets = _rescale_co_targets(co_targets, native_shape, (size_px, size_px))
        window = resize_to(window, size_px)
    return window, co_targets


def _extract_geo_style_window(
    masked, t, channel_idx: int, local_ping: int, style: str, level: str, size_px: int, jitter_px: int,
    rng: np.random.Generator, mps: float, items_by_channel: dict, lo: int,
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
    (``_match_frequency`` then has no basis to prefer one frequency
    channel over another and picks arbitrarily) can be correlated to a
    frequency/file whose own sample count is smaller than the sample
    index the target's position was originally captured against on a
    *different* channel -- an exact ``.isel(sample=t.sample)`` then raises
    ``IndexError`` where the pixel-space styles merely clamp silently
    (``extract_chip``/``extract_full_channel``'s own edge-shifting
    window placement never indexes out of bounds). Skipping here matches
    that same "don't crash the whole run over one bad target" contract,
    not a fix to the underlying frequency-matching ambiguity itself.
    """
    if t.sample is None or not (0 <= t.sample < masked.sizes["sample"]):
        return None
    try:
        along_dist = along_track_distance_m(masked)
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
        resolution_m = max(mps, ping_spacing_m)
        center_across_m = target_across_m + _jitter_offset_m(jitter_px, resolution_m, rng)
    else:
        across_all = masked["across_track_m"].values if combine else masked["across_track_m"].isel(channel=channel_idx).values
        across_finite = across_all[np.isfinite(across_all)]
        swath_width_m = float(across_finite.max() - across_finite.min()) if across_finite.size else size_px * mps
        resolution_m = max(swath_width_m / size_px, ping_spacing_m, 1e-6)
        center_across_m = 0.0 if combine else (float((across_finite.max() + across_finite.min()) / 2.0) if across_finite.size else 0.0)
    geo_search_radius_m = max(1.5 * resolution_m, gap_p95_m)

    if level == "geometric":
        geom_channel_idx = None if combine else channel_idx
        center_along_m = target_along_m + _jitter_offset_m(jitter_px, resolution_m, rng)
        try:
            grid, meta = extract_geometric_window(
                masked, size_px, resolution_m=resolution_m, center_along_m=center_along_m,
                center_across_m=center_across_m, channel_idx=geom_channel_idx, search_radius_m=geo_search_radius_m,
                avoid_edge_gaps=True,
            )
        except ValueError:
            return None
        row, col = locate_point_geometric(target_along_m, target_across_m, meta)
        primary_local = replace(t, sample=col, extra_attributes={**t.extra_attributes, "window_row": row})
        all_items = [item for items in items_by_channel.values() for item in items] if combine else items_by_channel.get(channel_idx, [])
        candidates = [item for item in all_items if item[0].target_id != t.target_id]
        co_targets = _geo_targets_within_window(candidates, masked, lo, along_dist, size_px, meta, locate_point_geometric)
        grid, targets_out = _flip_geo_vertical(grid, [primary_local] + co_targets, size_px)
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
            georef_input["backscatter_intensity"] = georef_input["backscatter_intensity"].copy()
            georef_input["backscatter_intensity"][{"channel": excluded}] = np.nan
        try:
            raster, meta = extract_georef_window(
                georef_input, size_px, resolution_m=resolution_m, center_lat=center_lat, center_lon=center_lon,
                max_search_radius_m=geo_search_radius_m,
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
        return grid, [primary_local]


def run_chip(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    targets = read_targets_json(args.targets_json)
    if args.min_confidence != "unknown":
        targets = [t for t in targets if _CONFIDENCE_RANK[t.position_confidence] >= _CONFIDENCE_RANK[args.min_confidence]]
    logger.info(f"Read {len(targets)} target(s) at or above min-confidence={args.min_confidence!r}")

    index = _build_position_index(args.ss_h5_dir)

    matched = []
    for t in targets:
        m = _correlate_target(t, index, args.max_match_dist_m)
        if m is None:
            logger.warning(f"target {t.target_id}: no ss_h5 ping within {args.max_match_dist_m}m of its recorded position, skipping")
            continue
        path, freq_key, ping_idx, dist_m = m
        logger.debug(f"target {t.target_id} -> {path.name} {freq_key} ping {ping_idx} ({dist_m:.1f}m)")
        matched.append((t, path, freq_key, ping_idx))

    if not matched:
        logger.error("No targets correlated to any ss_h5 ping — nothing to export")
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
    raw_windows: dict[str, list] = {s: [] for s in styles}
    window_meta: dict[str, list] = {s: [] for s in styles}  # (co_targets, primary, file_stem, freq_key)

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

        max_native_span = 0
        if "chip" in styles:
            max_native_span = max(max_native_span, args.size_px)
        if "single-swath" in styles:
            max_native_span = max(max_native_span, n_samples_single)
        if "dual-swath" in styles:
            max_native_span = max(max_native_span, 2 * n_samples_single)
        half_span = max_native_span // 2 + args.jitter_px + 50

        for channel_idx, its in items_by_channel.items():
            for t, ping_idx in its:
                # Slice a bounded ping neighborhood before loading/masking
                # -- this project's real files run up to 14.8GB with
                # every correction stage's arrays included; loading one
                # target's neighborhood is bounded regardless of file
                # size, loading the whole file per group isn't.
                lo = max(0, ping_idx - half_span)
                hi = min(n_pings, ping_idx + half_span)
                masked = mask_nadir_gap(ds_full.isel(ping=slice(lo, hi)).load())
                local_ping = ping_idx - lo
                mps = _sample_interval_m(masked)

                if args.with_context:
                    port_2d = masked["backscatter_intensity"].isel(channel=0).values
                    stbd_2d = masked["backscatter_intensity"].isel(channel=1).values
                    combined = _combine_channels_mirrored(port_2d, stbd_2d)
                    combined_col = _target_combined_col(t.sample, n_samples_single, channel_idx)
                    context_target = replace(t, sample=combined_col, extra_attributes={**t.extra_attributes, "window_row": local_ping})
                    # "combined" is the same port-mirrored dual-swath layout
                    # _extract_pixel_style_window builds -- reuse the same
                    # orientation fix (style="dual-swath" skips the
                    # already-applied horizontal mirror, only flips vertically).
                    combined, (context_target,) = _orient_pixel_window(combined, [context_target], channel_idx, "dual-swath")
                    fig = render_target_context(combined, context_target, box_size=(args.box_size, args.box_size))
                    fig.savefig(args.output / "context" / f"context_{t.target_id}.png", dpi=150)
                    plt.close(fig)

                for style in styles:
                    size_px = _style_size_px(style, args)
                    if args.render_level in ("raw", "radiometric"):
                        result = _extract_pixel_style_window(
                            masked, t, channel_idx, n_samples_single, local_ping, lo, style, args.render_level,
                            size_px, args.jitter_px, rng, placed_by_channel, placed_combined,
                        )
                    else:
                        result = _extract_geo_style_window(
                            masked, t, channel_idx, local_ping, style, args.render_level, size_px, args.jitter_px, rng, mps,
                            items_by_channel, lo,
                        )
                    if result is None:
                        logger.warning(f"target {t.target_id}: {style}/{args.render_level} window unavailable, skipping")
                        continue
                    window, co_targets = result
                    raw_windows[style].append(window)
                    window_meta[style].append((co_targets, t, path.stem, freq_key))

    total_written = 0
    for style in styles:
        windows = raw_windows[style]
        if not windows:
            logger.warning(f"{style}: no windows produced at render-level={args.render_level!r} — skipping this style's COCO/YOLO export")
            continue

        reference_stats = _reference_stats_for_scope(args.normalization_scope, windows, args.pct_low, args.pct_high)

        coco_entries = []
        thumbnails, captions, sheet_boxes = [], [], []
        style_dir = args.output / style
        for window, (co_targets, primary, file_stem, freq_key) in zip(windows, window_meta[style]):
            img_8bit, _meta = to_uint8(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high, reference_stats=reference_stats)
            file_name = f"target_{primary.target_id}.png"
            _save_grayscale(img_8bit, style_dir / file_name)

            for co in co_targets:
                coco_entries.append(
                    CocoImageEntry(target=co, file_name=file_name, width=img_8bit.shape[1], height=img_8bit.shape[0])
                )
            thumbnails.append(img_8bit)
            sheet_boxes.append(co_targets)
            n_co = len(co_targets)
            captions.append(f"{primary.target_id} ({freq_key}, {n_co} target{'s' if n_co != 1 else ''})")

        sheet_fig = contact_sheet(
            thumbnails, captions, boxes=sheet_boxes, box_size=(args.box_size, args.box_size),
            ncols=6, suptitle=f"{style}: {len(thumbnails)} chips",
        )
        sheet_fig.savefig(style_dir / "contact_sheet.png", dpi=150)
        plt.close(sheet_fig)

        # write_coco()/write_yolo() have their own independent
        # min_confidence filter (default "verified", separate from the
        # target-list filter above) -- pass args.min_confidence through
        # explicitly, or a format whose targets are never "verified"
        # (e.g. Klein's .tgt, which is unconditionally "unknown") gets a
        # real crop for every target but zero COCO/YOLO annotations for
        # any of them, silently.
        coco = write_coco(coco_entries, style_dir / "targets.coco.json", box_size=(args.box_size, args.box_size), min_confidence=args.min_confidence)
        logger.info(f"{style}: COCO export: {len(coco['images'])} image(s), {len(coco['annotations'])} annotation(s) -> {style_dir / 'targets.coco.json'}")
        yolo = write_yolo(coco_entries, style_dir / "targets.yolo", box_size=(args.box_size, args.box_size), min_confidence=args.min_confidence)
        logger.info(f"{style}: YOLO export: {len(yolo['images'])} image(s) -> {style_dir / 'targets.yolo'}")
        logger.info(f"{style}: wrote {len(windows)} chip(s) + contact sheet to {style_dir}")
        total_written += len(windows)

    if total_written == 0:
        logger.error(f"No style produced any window at render-level={args.render_level!r} — nothing exported")
        return 1
    return 0


def run_survey_window(args: argparse.Namespace) -> int:
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
            # files here run up to 14.8GB with every correction stage's
            # arrays included) -- native window height == n_samples, plus
            # jitter room.
            half_span = n_samples // 2 + args.jitter_px + 50
            lo = max(0, int(center_ping) - half_span)
            hi = min(n_pings, int(center_ping) + half_span)
            masked = mask_nadir_gap(ds.isel(ping=slice(lo, hi)).load())
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
            sliced = mask_nadir_gap(ds.isel(ping=slice(max(0, int(center_ping) - 100), int(center_ping) + 100)).load())
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
            sliced = mask_nadir_gap(ds.isel(ping=slice(max(0, int(center_ping) - 100), int(center_ping) + 100)).load())
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
        logger.error("No windows extracted — nothing to export")
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
    sheet_fig.savefig(args.output / "contact_sheet.png", dpi=150)
    plt.close(sheet_fig)

    # No real target ground truth is correlated to this array (see module
    # docstring) — every window here is honestly a no-target negative, so
    # the COCO/YOLO export has zero annotations, demonstrating that path
    # rather than faking positive boxes.
    empty_entries: list[CocoImageEntry] = []
    write_coco(empty_entries, args.output / "empty.coco.json")
    logger.info(f"Wrote {len(raw_windows)} {args.mode} window(s) + contact sheet to {args.output}")
    return 0


# ============================================================
# Processing-stage progression grid (raw / dB / TVG / geometric / georef)
# ============================================================

_PROGRESSION_STAGES = ("raw", "db", "tvg", "geometric", "georef")


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
    """Build a 7(-ish)-target x 5-stage progression grid (raw / dB-converted
    / TVG-applied / geometric-corrected / georef-corrected) at either
    full-channel or chip scale, so the same real target is directly
    comparable across every processing stage. See old
    ``project-a/scripts/14_export_dataset.py``'s (superseded) copy of
    this function for the full design rationale — unchanged here.
    """
    args.output.mkdir(parents=True, exist_ok=True)
    targets = read_targets_json(args.targets_json)
    if args.min_confidence != "unknown":
        targets = [t for t in targets if _CONFIDENCE_RANK[t.position_confidence] >= _CONFIDENCE_RANK[args.min_confidence]]
    if args.channel != "both":
        want = 0 if args.channel == "port" else 1
        targets = [t for t in targets if t.channel == want]

    index = _build_position_index(args.ss_h5_dir)
    file_n_pings = {path: max(len(c["lat"]) for c in channels.values()) for path, channels in index.items()}

    matched = []
    for t in targets:
        m = _correlate_target(t, index, args.max_match_dist_m)
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
        masked = mask_nadir_gap(ds_full.isel(ping=slice(lo, hi)).load())
        local_ping = ping_idx - lo
        mps = _sample_interval_m(masked)

        stage_images: dict[str, np.ndarray] = {}
        stage_positions: dict[str, tuple[float, float] | None] = {}

        if args.scale == "full":
            if combine_sides:
                raw_port = masked["backscatter_raw"].isel(channel=0).values
                raw_stbd = masked["backscatter_raw"].isel(channel=1).values
                raw_2d = _combine_channels_mirrored(raw_port, raw_stbd)
                intensity_port = masked["backscatter_intensity"].isel(channel=0).values
                intensity_stbd = masked["backscatter_intensity"].isel(channel=1).values
                intensity_2d = _combine_channels_mirrored(intensity_port, intensity_stbd)
                target_col = _target_combined_col(t.sample, n_samples_single, channel_idx)
            else:
                raw_2d = masked["backscatter_raw"].isel(channel=channel_idx).values
                intensity_2d = masked["backscatter_intensity"].isel(channel=channel_idx).values
                target_col = t.sample
            if args.mask_bright_rows:
                raw_2d = _mask_bright_ping_rows(raw_2d, mad_k=args.mask_bright_rows_k)
                intensity_2d = _mask_bright_ping_rows(intensity_2d, mad_k=args.mask_bright_rows_k)
            _rw, extent = extract_full_channel(raw_2d, center_ping=float(local_ping), jitter_px=args.jitter_px, rng=rng)
        else:
            raw_2d = masked["backscatter_raw"].isel(channel=channel_idx).values
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

        for stage, window in (("raw", raw_window), ("db", db_window), ("tvg", intensity_window)):
            img, _meta = to_uint8(window, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high)
            pos = target_local
            if args.scale == "full":
                native_shape = window.shape
                img = np.clip(resize_to(img, args.output_size_px), 0, 255).astype(np.uint8)
                pos = rescale_target_position(target_local[0], target_local[1], native_shape, (args.output_size_px, args.output_size_px))
            stage_images[stage] = img
            stage_positions[stage] = pos

        along_dist = along_track_distance_m(masked)
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

        geo_search_radius_m = max(1.5 * geo_resolution_m, gap_p95_m)

        try:
            geom_grid, geom_meta = extract_geometric_window(
                masked, geo_size_px, resolution_m=geo_resolution_m,
                center_along_m=geom_center_along_m, center_across_m=geom_center_across_m, channel_idx=geo_channel_idx,
                search_radius_m=geo_search_radius_m,
            )
            geom_img, _ = to_uint8(geom_grid, gamma=args.gamma, pct_low=args.pct_low, pct_high=args.pct_high)
            geom_pos = locate_point_geometric(target_along_m, target_across_m, geom_meta)
        except ValueError:
            geom_img = np.zeros((geo_size_px, geo_size_px), dtype=np.uint8)
            geom_pos = None
        stage_images["geometric"] = geom_img
        stage_positions["geometric"] = geom_pos

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
        stage_images["georef"] = georef_img
        stage_positions["georef"] = georef_pos

        for stage in _PROGRESSION_STAGES:
            file_name = f"{t.target_id}_{stage}.png"
            _save_grayscale(stage_images[stage], args.output / file_name)

        columns.append({
            "target_id": t.target_id,
            "source_file": path.name,
            "positions": {k: (list(v) if v is not None else None) for k, v in stage_positions.items()},
        })
        logger.info(f"target {t.target_id}: wrote all {len(_PROGRESSION_STAGES)} stages ({path.name})")

    with open(args.output / "progression_meta.json", "w") as f:
        json.dump({"scale": args.scale, "stages": list(_PROGRESSION_STAGES), "columns": columns}, f, indent=2)
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
        "--value-var", choices=["raw", "radiometric"], default="radiometric",
        help="raw = backscatter_raw (no TVG); radiometric = backscatter_intensity (TVG-corrected, "
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
        "--render-level", choices=list(_RENDER_LEVELS), default="radiometric",
        help="Processing level every style renders from this run (applies to both COCO and YOLO): "
             "'raw' (backscatter_raw, no TVG), 'radiometric' (TVG-corrected, the old default), "
             "'geometric' (across/along-track meters -- requires --geometric already run via "
             "sidescan-correct), 'georef' (lat/lon-projected meters -- requires --georef already run). "
             "Under geometric/georef every style's image is a real size_px*resolution_m meters square, "
             "not a pixel crop.",
    )
    p_chip.add_argument("--size-px", type=int, default=400, help="Output size (both axes) for the 'chip' style")
    p_chip.add_argument("--swath-size-px", type=int, default=512, help="Output size (both axes) for the 'single-swath'/'dual-swath' styles")
    p_chip.add_argument("--box-size", type=float, default=40.0, help="COCO/YOLO annotation box size AND (with --with-context) the drawn box size, in pixels")
    p_chip.add_argument(
        "--with-context", action="store_true",
        help="Also save context/context_<id>.png -- the full loaded ping neighborhood, BOTH channels "
             "combined (port+starboard), with a real box drawn at the target's true position, via "
             "sidescanpy.viz.render_target_context. Independent of --styles/--render-level -- always "
             "radiometric-level, matching this feature's existing QA-visualization purpose.",
    )
    p_chip.add_argument("--jitter-px", type=int, default=0)
    p_chip.add_argument("--gamma", type=float, default=common["gamma"])
    p_chip.add_argument("--pct-low", type=float, default=common["pct_low"])
    p_chip.add_argument("--pct-high", type=float, default=common["pct_high"])
    p_chip.add_argument("--normalization-scope", choices=["image", "shared"], default="image")
    p_chip.add_argument("--min-confidence", choices=["unknown", "approximate", "verified"], default="verified")
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
        help="7-target x 5-stage grid (raw / dB / TVG / geometric / georef), one shared jitter draw per target",
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
    p_prog.add_argument("--seed", type=int, default=None)
    p_prog.set_defaults(func=run_progression)

    all_subparsers = (p_wf, p_grid, p_track, p_chip, p_sw, p_prog)
    for p in all_subparsers:
        p.add_argument("--config", type=Path, default=None, help="YAML or JSON file seeding this subcommand's defaults -- explicit flags still override")
        # default=SUPPRESS: see sidescanpy/cli/convert.py's matching
        # block for why a plain default=False here would silently
        # overwrite a real -v already set before the subcommand.
        p.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Also accepted after the subcommand")

    parser.add_argument("-v", "--verbose", action="store_true")

    argv = sys.argv[1:]
    config_path = find_config_path(argv)
    if config_path is not None:
        config = load_config(config_path)
        for p in all_subparsers:
            apply_config_defaults(p, config)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = args.func(args)
    logger.info(f"export {args.subcommand} complete")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
