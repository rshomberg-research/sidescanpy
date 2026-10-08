"""
``sidescan-export``: waterfalls, gridded rasters, survey tracks, and
annotated training datasets from already-corrected ``.ss.h5`` files.

Export subcommands for track maps, waterfalls, grids, and target-centered
dataset chips (``chip``, a thin wrapper over
:func:`sidescanpy.export.ml.build_target_dataset`).

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
  meters-square -- requires that correction already run via
  ``sidescan-correct``). ``--val-fraction``/``--layout ultralytics``
  add a train/val split and an Ultralytics-ready directory layout.
  ``--with-context`` (independent of styles/render-level, always
  radiometric) saves ``context/context_<id>.png`` — the full ping
  neighborhood, both channels combined, with a real box drawn at the
  target's position.
  ``--n-background N`` adds N screened no-target images per style
  (``<style>/bg_<n>.png``, empty labels), rendered the same way (see
  :func:`sidescanpy.export.windows.sample_background_windows`).

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
import dataclasses
import importlib
import json
import logging
import sys
from pathlib import Path

from sidescanpy.cli._common import (
    add_logging_args,
    require_matplotlib,
    resolve_freq_keys,
    setup_logging_from_args,
    uncorrected_alias,
)
from sidescanpy.cli._config import apply_config_to_subparsers, find_config_path, load_config
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.export.ml import DatasetSpec, build_target_dataset
from sidescanpy.export.windows import (
    CHIP_STYLES,
    RENDER_LEVELS,
    with_uncorrected_var,
)
from sidescanpy.grids import (
    ResamplingMethod,
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.io.hdf5 import concat_survey_frequency, read_survey
from sidescanpy.processing.inversion import UNCORRECTED_VAR
from sidescanpy.processing.track import build_survey_track
from sidescanpy.targets.correlate import build_position_index, correlate_catalog
from sidescanpy.targets.io import read_targets_json
from sidescanpy.targets.waterfall import BOX_AXES, targets_by_global_ping, waterfall_boxes

logger = logging.getLogger(__name__)

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

def _export_waterfall_png(ds, output_path: Path, *, x_axis, y_axis, ping_offset, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, dpi, boxes=None, box_size=40.0, value_var="backscatter_intensity"):
    """Render one waterfall window, with a labeled box for each ``(target, global_ping_idx)`` in ``boxes``."""
    from sidescanpy.cli import _figures

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _figures.save_waterfall_png(
        ds, output_path, x_axis=x_axis, y_axis=y_axis, ping_offset=ping_offset,
        cmap=cmap, vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
        gamma=gamma, dpi=dpi, value_var=value_var,
        box_rects=waterfall_boxes(ds, boxes or [], ping_offset, x_axis, y_axis, box_size),
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
        if (x_axis, y_axis) not in BOX_AXES:
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
    # Each target on every frequency of its file: its own mark on the
    # frequency it was picked on, a slant-range projection on the others.
    matched = correlate_catalog(targets, index, args.max_match_dist_m)
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
            ds_all = with_uncorrected_var(ds_all)

        boxes_by_ping = targets_by_global_ping(matched, freq_key, file_start_ping)

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

def _load_ping_ranges(value) -> dict[str, list[tuple[int, int]]] | None:
    """``--background-ping-ranges``: a JSON file path, a JSON string, or an
    already-parsed mapping (from a config file) -> ``{file_name: [(start, stop), ...]}``."""
    if value is None:
        return None
    if isinstance(value, str | Path) and Path(value).is_file():
        value = json.loads(Path(value).read_text())
    elif isinstance(value, str):
        value = json.loads(value)
    return {str(name): [(int(a), int(b)) for a, b in ranges] for name, ranges in value.items()}


def _parse_categories(value) -> list[str] | None:
    """``--categories a,b`` (or a list from a config file) -> ordered names, or ``None``."""
    if value is None:
        return None
    names = [v.strip() for v in value.split(",")] if isinstance(value, str) else [str(v) for v in value]
    names = [n for n in names if n]
    return names or None


def _spec_from_args(args: argparse.Namespace) -> DatasetSpec:
    """``DatasetSpec`` from parsed ``chip`` arguments (absent attributes keep spec defaults)."""
    values = {}
    for f in dataclasses.fields(DatasetSpec):
        if f.name == "categories":
            values[f.name] = _parse_categories(getattr(args, "categories", None))
        elif f.name == "background_ping_ranges":
            values[f.name] = _load_ping_ranges(getattr(args, "background_ping_ranges", None))
        elif hasattr(args, f.name):
            values[f.name] = getattr(args, f.name)
    return DatasetSpec(**values)


def run_chip(args: argparse.Namespace) -> int:
    from sidescanpy.cli import _figures
    # The module, not the same-named function sidescanpy.viz re-exports.
    viz_sheet = importlib.import_module("sidescanpy.viz.contact_sheet")

    def save_contact_sheet(thumbnails, captions, *, boxes, box_size, suptitle, path):
        fig = viz_sheet.contact_sheet(thumbnails, captions, boxes=boxes, box_size=box_size, ncols=6, suptitle=suptitle)
        _figures.save_figure(fig, path, dpi=150)

    def save_context(image, target, *, coordinate_system, resolution_m, box_size, path):
        fig = viz_sheet.render_target_context(
            image, target, box_size=box_size, coordinate_system=coordinate_system, resolution_m=resolution_m,
        )
        _figures.save_figure(fig, path, dpi=150)

    spec = _spec_from_args(args)
    targets = read_targets_json(args.targets_json)
    try:
        result = build_target_dataset(
            args.ss_h5_dir, targets, spec, args.output, contact_sheet=save_contact_sheet, context=save_context,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if result.n_matched == 0:
        logger.error("No targets correlated to any ss_h5 ping -- nothing to export")
        return 1
    if result.n_images == 0:
        logger.error(f"No style produced any window at render-level={spec.render_level!r} -- nothing exported")
        return 1
    per_style = ", ".join(f"{key}: {len(s.images)}" for key, s in result.styles.items())
    logger.info(f"chip: {result.n_matched}/{result.n_targets} target(s) matched; images per style: {per_style}; manifest {result.manifest_path}")
    return 0


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
        "--styles", nargs="+", choices=list(CHIP_STYLES), default=list(CHIP_STYLES),
        help="Which chip style(s) to export as parallel COCO/YOLO datasets under output/<style>/: "
             "'chip' (tight single-channel crop, --size-px), 'single-swath' (one full channel, square), "
             "'dual-swath' (both channels combined, square, port+starboard). Default: all three.",
    )
    p_chip.add_argument(
        "--render-level", choices=list(RENDER_LEVELS), type=uncorrected_alias("raw"), default="radiometric",
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
        "--resolution-m", type=float, nargs="+", default=None,
        help="Real-world meters per output pixel for 'geometric'/'georef' render levels, applied to "
             "every style. When set, used directly instead of the per-target auto-calculated value "
             "(swath width / output size for single-swath/dual-swath, native sample/ping spacing for "
             "chip) -- that auto-calculation has a known failure mode: it derives ping spacing from an "
             "average over a wide ping neighborhood, so a target sitting in a locally slow/dense-ping "
             "region (e.g. where the vessel slowed for a closer look) can get an inflated, too-coarse "
             "resolution that makes the whole crop look zoomed far out. Set this from sidescan-analyze's "
             "findings for the dataset, not by trial and error. No effect (logged, not an error) under "
             "'uncorrected'/'radiometric' render level, which crops in native pixel space and has no resolution_m "
             "concept at all. Several values build one dataset per resolution under res_<m>m/.",
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
    p_chip.add_argument(
        "--n-background", type=int, default=0,
        help="Background (no-target) images per style, written as <style>/bg_<n>.png (bg_<n>_a<k>.png with --n-augment) "
             "with empty labels and rendered exactly like target images. Windows are sampled uniformly by ping over the "
             "frequencies the matched targets use, and rejected if any cataloged target (any confidence) lies inside the "
             "window plus --background-margin-m, or if more than --background-max-nan-frac of it is empty. Only as clean "
             "as the catalog: an uncataloged object can appear in a background image. Default 0 (off).",
    )
    p_chip.add_argument("--background-margin-m", type=float, default=0.0, help="Extra clearance (meters) a background window keeps from every cataloged target")
    p_chip.add_argument("--background-max-nan-frac", type=float, default=0.3, help="Reject background windows with more than this fraction of empty (NaN) cells")
    p_chip.add_argument(
        "--background-ping-ranges", default=None,
        help="Restrict background sampling to these ping ranges: a JSON file or JSON string mapping .ss.h5 file name to "
             "[[start, stop], ...] half-open ranges (e.g. clean segments). Files not listed contribute no background.",
    )
    p_chip.add_argument(
        "--val-fraction", type=float, default=0.0,
        help="Share of targets (by target-id hash) and background ping blocks assigned to the validation split. "
             "0 (default): no split.",
    )
    p_chip.add_argument("--split-block-pings", type=int, default=500, help="Background ping-block size for the split hash")
    p_chip.add_argument(
        "--layout", choices=["flat", "ultralytics"], default="flat",
        help="flat (default): images next to targets.coco.json/targets.yolo/. ultralytics: images/{train,val}/, "
             "labels/{train,val}/ and a data.yaml ready for Ultralytics training.",
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

    all_subparsers = (p_wf, p_grid, p_track, p_chip)
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

    if not require_matplotlib(f"sidescan-export {args.subcommand}"):
        return 1

    result = args.func(args)
    logger.info(f"export {args.subcommand} complete")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
