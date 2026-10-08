"""
Background (no-target) export windows.

A detector trained only on target-centered images never sees an empty
frame. :func:`sample_background_windows` samples window centers across a
set of converted surveys, renders each one through the same window
extractors as target windows (:func:`~sidescanpy.export.windows.pixel.extract_pixel_style_window`,
:func:`~sidescanpy.export.windows.geo.extract_geo_style_window`), and keeps
only windows that

1. contain no cataloged target within the window plus ``margin_m``. Every
   catalog target (any confidence) is correlated to its ping with
   :func:`~sidescanpy.targets.correlate.correlate_target`, projected onto
   every frequency of its file (:func:`~sidescanpy.targets.correlate.place_catalog`),
   and located in the rendered window with
   the same lookup target windows use for co-located targets. A
   conservative geographic guard additionally rejects a window whose
   reach from nadir (plus margin) covers any target's recorded lat/lon --
   this catches targets on another file, another line crossing the same
   ground, or targets that never correlated;
2. are mostly data: at most ``max_nan_frac`` of the rendered cells are
   NaN (outside coverage, nadir gap, cropped far range).

Background windows are only as clean as the target catalog: an object the
catalog misses can appear in one and be learned as background.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.geometric_window import locate_point as locate_point_geometric
from sidescanpy.export.windows.geo import (
    extract_geo_style_window,
    geo_targets_within_window,
    georef_targets_within_window,
)
from sidescanpy.export.windows.levels import (
    WindowSpec,
    mask_nadir_gap_with_uncorrected,
    nominal_sample_interval_m,
    with_db_var,
)
from sidescanpy.export.windows.neighborhood import (
    load_geo_neighborhood,
    neighborhood_ping_spacing_m,
)
from sidescanpy.export.windows.pixel import (
    RenderedWindow,
    WindowExtent,
    extract_pixel_style_window,
    place_targets_in_file,
    targets_within_window,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.targets.correlate import place_catalog

logger = logging.getLogger(__name__)

_GEOD = pyproj.Geod(ellps="WGS84")


@dataclass(frozen=True)
class BackgroundCandidate:
    """A sampled background window center.

    Attributes
    ----------
    path : Path
        Source ``.ss.h5`` file.
    freq_key : str
        Frequency channel.
    ping : int
        File ping index of the center.
    channel : int
        Side of the center sample (0 = port); the channel of a
        single-channel window.
    sample : int
        Center sample on ``channel``.
    """

    path: Path
    freq_key: str
    ping: int
    channel: int
    sample: int


@dataclass(frozen=True)
class WindowFootprint:
    """Real-world extent of a rendered window.

    Attributes
    ----------
    frame : str
        ``"pixel"``: ``rows`` are file pings, ``cols`` the style's column
        space (samples, or combined columns for dual swath).
        ``"geometric"``: ``rows`` along-track and ``cols`` across-track
        meters, in the loaded neighborhood's frame. ``"georef"``: ``rows``
        northing and ``cols`` easting meters in the window's UTM zone.
    rows, cols : tuple of float
        ``(min, max)`` bounds, before ``margin_m``.
    reach_m : float
        Upper bound on the distance from the center ping's nadir to any
        point of the window (used by the geographic guard).
    """

    frame: str
    rows: tuple[float, float]
    cols: tuple[float, float]
    reach_m: float


@dataclass
class BackgroundWindow:
    """One accepted background window, rendered and ready to write.

    Attributes
    ----------
    index : int
        Placement number within this style (0-based, in acceptance order).
    candidate : BackgroundCandidate
    style : str
    draw : int
        Draw index for ``n_draws > 1`` (independent jitter around the same
        center).
    across_center_m : float or None
        Signed across-track meters of the center sample
        (``across_track_m``, else signed slant range).
    footprint : WindowFootprint
    window : RenderedWindow
        ``window.targets`` is empty.
    nan_fraction : float
    """

    index: int
    candidate: BackgroundCandidate
    style: str
    draw: int
    across_center_m: float | None
    footprint: WindowFootprint
    window: RenderedWindow
    nan_fraction: float


def nan_fraction(image: np.ndarray) -> float:
    """Fraction of non-finite cells in ``image``."""
    return float(np.mean(~np.isfinite(image))) if image.size else 1.0


def sample_background_candidates(
    index: Mapping[Path, Mapping[str, Mapping]],
    n: int,
    rng: np.random.Generator,
    *,
    freq_keys: Sequence[str] | None = None,
    ping_ranges: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> list[BackgroundCandidate]:
    """Draw ``n`` window centers, uniformly by ping over every allowed ping.

    Parameters
    ----------
    index : dict
        From :func:`~sidescanpy.targets.correlate.build_position_index`.
    n : int
        Number of candidates.
    rng : np.random.Generator
    freq_keys : sequence of str, optional
        Frequencies to sample (default: every frequency in ``index``).
    ping_ranges : mapping, optional
        ``{file_name: [(start, stop), ...]}`` allowed half-open ping ranges
        per file (keyed by ``Path.name``). Files absent from the mapping
        contribute no pings; ``None`` allows every ping of every file.

    Returns
    -------
    list of BackgroundCandidate
        Empty if no ping is allowed. Channel and center sample are uniform.
    """
    pools = []
    for path, channels in index.items():
        for freq_key, info in channels.items():
            if freq_keys is not None and freq_key not in freq_keys:
                continue
            n_pings = int(info["n_pings"])
            if ping_ranges is None:
                ranges = [(0, n_pings)]
            else:
                ranges = [(max(0, int(a)), min(n_pings, int(b))) for a, b in ping_ranges.get(Path(path).name, [])]
            for a, b in ranges:
                if b > a:
                    pools.append((Path(path), freq_key, a, b, int(info["n_samples"])))
    if not pools:
        logger.warning("no allowed pings to sample background windows from")
        return []
    sizes = np.array([b - a for _p, _f, a, b, _s in pools])
    cumulative = np.cumsum(sizes)
    out = []
    for _ in range(n):
        k = int(rng.integers(0, cumulative[-1]))
        i = int(np.searchsorted(cumulative, k, side="right"))
        path, freq_key, a, _b, n_samples = pools[i]
        ping = a + k - (int(cumulative[i - 1]) if i else 0)
        channel = int(rng.integers(0, 2))
        sample = int(rng.integers(0, n_samples))
        out.append(BackgroundCandidate(path, freq_key, ping, channel, sample))
    return out


def _guard_hits(nadir_lat: float, nadir_lon: float, reach_m: float, catalog_latlon: np.ndarray) -> int:
    """Number of catalog positions within ``reach_m`` of the nadir point."""
    if not catalog_latlon.size or not np.isfinite(nadir_lat) or not np.isfinite(nadir_lon):
        return 0
    n = len(catalog_latlon)
    _az, _baz, dist = _GEOD.inv(np.full(n, nadir_lon), np.full(n, nadir_lat), catalog_latlon[:, 1], catalog_latlon[:, 0])
    return int(np.sum(dist <= reach_m))


def _signed_across(ds: xr.Dataset, local_ping: int, channel: int, sample: int, mps: float) -> float:
    if "across_track_m" in ds:
        value = float(ds["across_track_m"].isel(ping=local_ping, channel=channel, sample=sample).values)
        if np.isfinite(value):
            return value
    return (-1.0 if channel == 0 else 1.0) * sample * mps


def _render_pixel(
    ds_full: xr.Dataset, cand: BackgroundCandidate, spec: WindowSpec, items: dict, rng: np.random.Generator,
    n_draws: int, margin_m: float,
) -> tuple[list[tuple[RenderedWindow, WindowFootprint, list[Target]]], float | None]:
    n_pings, n_samples = ds_full.sizes["ping"], ds_full.sizes["sample"]
    mps = nominal_sample_interval_m(ds_full)
    native_span = {"chip": spec.size_px, "single-swath": n_samples, "dual-swath": 2 * n_samples}[spec.style]
    spacing_lo, spacing_hi = max(0, cand.ping - 500), min(n_pings, cand.ping + 500)
    ping_spacing_m = neighborhood_ping_spacing_m(ds_full, spacing_lo, spacing_hi) or mps
    margin_rows = margin_m / ping_spacing_m
    margin_cols = margin_m / mps
    half_span = native_span // 2 + spec.jitter_px + 50 + int(np.ceil(margin_rows))
    lo, hi = max(0, cand.ping - half_span), min(n_pings, cand.ping + half_span)
    masked = mask_nadir_gap_with_uncorrected(ds_full.isel(ping=slice(lo, hi)).load())
    if spec.needs_db:
        masked = with_db_var(masked)
    local_ping = cand.ping - lo
    across_center = _signed_across(masked, local_ping, cand.channel, cand.sample, mps)
    placed_by_channel, placed_combined = place_targets_in_file(items, n_samples)
    co_source = placed_combined if spec.style == "dual-swath" else placed_by_channel.get(cand.channel, [])

    draws = []
    for _ in range(n_draws):
        rendered = extract_pixel_style_window(
            masked, center_sample=cand.sample, channel_idx=cand.channel, local_ping=local_ping, ping_offset=lo,
            style=spec.style, level=spec.level, size_px=spec.size_px, jitter_px=spec.jitter_px, rng=rng,
            placed_by_channel=placed_by_channel, placed_combined=placed_combined, require_targets=False,
        )
        if rendered is None:
            draws.append(None)
            continue
        ext = rendered.extent
        grown = WindowExtent(
            row_start=int(np.floor(ext.row_start - margin_rows)), col_start=int(np.floor(ext.col_start - margin_cols)),
            n_rows=int(np.ceil(ext.n_rows + 2 * margin_rows)) + 1, n_cols=int(np.ceil(ext.n_cols + 2 * margin_cols)) + 1,
        )
        hits = targets_within_window(co_source, grown)
        rows = (float(ext.row_start), float(ext.row_start + ext.n_rows))
        cols = (float(ext.col_start), float(ext.col_start + ext.n_cols))
        along_reach = max(abs(rows[0] - cand.ping), abs(rows[1] - cand.ping)) * ping_spacing_m
        if spec.style == "dual-swath":
            across_reach = n_samples * mps
        else:
            across_reach = max(abs(cols[0]), abs(cols[1])) * mps
        draws.append((rendered, WindowFootprint("pixel", rows, cols, along_reach + across_reach), hits))
    return draws, across_center


def _render_geo(
    ds_full: xr.Dataset, cand: BackgroundCandidate, spec: WindowSpec, items: dict, rng: np.random.Generator,
    n_draws: int, margin_m: float,
) -> tuple[list[tuple[RenderedWindow, WindowFootprint, list[Target]] | None], float | None]:
    view = load_geo_neighborhood(
        ds_full, None, cand.ping, items, [spec.style], size_px=spec.size_px, jitter_px=spec.jitter_px,
        resolution_m=spec.resolution_m, with_db=spec.needs_db, across_axis=spec.across_axis,
        per_sample_latlon=spec.level == "georef", swath_fill=spec.swath_fill, margin_m=margin_m,
    )
    masked = view.ds
    center_sample = view.to_averaged_sample(cand.sample)
    if not (0 <= center_sample < masked.sizes["sample"]):
        return [None] * n_draws, None
    across_center = _signed_across(masked, view.local_ping, cand.channel, center_sample, view.mps)
    all_items = [item for its in view.items.values() for item in its]
    along_dist = along_track_distance_m(masked, fill_nav_stalls=True)

    draws = []
    for _ in range(n_draws):
        rendered = extract_geo_style_window(
            masked, center_sample=center_sample, channel_idx=cand.channel, local_ping=view.local_ping, style=spec.style,
            level=spec.level, size_px=spec.size_px, jitter_px=spec.jitter_px, rng=rng, mps=view.mps,
            items_by_channel=view.items, lo=view.lo, primary=None, resolution_m_override=spec.resolution_m,
            backscatter_var=spec.backscatter_var, resample_method=spec.resample_method,
            swath_fill=spec.swath_fill and spec.level == "geometric",
        )
        if rendered is None:
            draws.append(None)
            continue
        meta, res = rendered.meta, rendered.resolution_m
        n_rows, n_cols = rendered.grid_shape
        margin_px = margin_m / res
        if spec.level == "geometric":
            hits = geo_targets_within_window(
                all_items, masked, view.lo, along_dist, spec.size_px, meta, locate_point_geometric,
                along_size_px=n_rows, margin_px=margin_px,
            )
            along0, across0 = meta["along_min"], meta["across_min"]
            rows = (along0, along0 + n_rows * meta["along_resolution_m"])
            cols = (across0, across0 + n_cols * meta["across_resolution_m"])
            center_along = float(along_dist[view.local_ping])
            reach = max(abs(rows[0] - center_along), abs(rows[1] - center_along)) + max(abs(cols[0]), abs(cols[1]))
            footprint = WindowFootprint("geometric", rows, cols, reach)
        else:
            hits = georef_targets_within_window(all_items, masked, view.lo, meta, spec.size_px, margin_px=margin_px)
            rows = (meta["northing_min"], meta["northing_min"] + n_rows * res)
            cols = (meta["easting_min"], meta["easting_min"] + n_cols * res)
            to_utm = pyproj.Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
            nadir_e, nadir_n = to_utm.transform(
                float(masked["longitude"].values[view.local_ping]), float(masked["latitude"].values[view.local_ping]),
            )
            reach = max(np.hypot(n - nadir_n, e - nadir_e) for n in rows for e in cols)
            footprint = WindowFootprint("georef", rows, cols, float(reach))
        draws.append((rendered, footprint, hits))
    return draws, across_center


def sample_background_windows(
    index: Mapping[Path, Mapping[str, Mapping]],
    catalog: Sequence[Target],
    spec: WindowSpec,
    n: int,
    rng: np.random.Generator,
    *,
    freq_keys: Sequence[str] | None = None,
    ping_ranges: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    margin_m: float = 0.0,
    max_nan_frac: float = 0.3,
    max_match_dist_m: float = 25.0,
    n_draws: int = 1,
    max_attempts: int | None = None,
    candidates: Iterable[BackgroundCandidate] | None = None,
) -> list[BackgroundWindow]:
    """Sample, render and screen ``n`` background windows for one style.

    Candidates come from :func:`sample_background_candidates` (or from
    ``candidates``), one at a time, until ``n`` are accepted or
    ``max_attempts`` are spent. A
    candidate is rendered ``n_draws`` times (independent jitter around the
    same center, as ``n_augment`` does for targets); it is accepted when
    its first draw passes, and each later draw is kept only if it passes
    too. A draw passes when it has no catalog target inside the window
    plus ``margin_m`` (the frame check and the geographic guard; see the
    module docstring) and at most ``max_nan_frac`` NaN cells.

    Parameters
    ----------
    index : dict
        From :func:`~sidescanpy.targets.correlate.build_position_index`.
    catalog : sequence of Target
        Every cataloged target, regardless of confidence.
    spec : WindowSpec
        Style and rendering settings -- the same ones target windows use.
    n : int
        Accepted windows wanted (each contributing up to ``n_draws`` draws).
    rng : np.random.Generator
        Drives candidate sampling and jitter.
    freq_keys, ping_ranges
        See :func:`sample_background_candidates`.
    margin_m : float, default 0.0
        Extra clearance around the window, meters.
    max_nan_frac : float, default 0.3
    max_match_dist_m : float, default 25.0
        Catalog correlation distance (see :func:`~sidescanpy.targets.correlate.place_catalog`).
    n_draws : int, default 1
    max_attempts : int, optional
        Candidates to try before giving up (default ``50 * n``).
    candidates : iterable of BackgroundCandidate, optional
        Caller-chosen centers to screen, in order, instead of sampling;
        ``freq_keys``/``ping_ranges`` are then not applied.

    Returns
    -------
    list of BackgroundWindow
        In acceptance order; fewer than ``n`` placements (with a warning)
        if the attempts run out.
    """
    if n <= 0:
        return []
    max_attempts = 50 * n if max_attempts is None else max_attempts
    placed = place_catalog(catalog, index, max_match_dist_m)
    catalog_latlon = np.array(
        [(t.latitude, t.longitude) for t in catalog if t.latitude is not None and t.longitude is not None], dtype=float,
    ).reshape(-1, 2)
    datasets: dict[tuple[Path, str], xr.Dataset] = {}
    accepted: list[BackgroundWindow] = []
    n_placements = 0
    rejected = defaultdict(int)
    supplied = None if candidates is None else iter(candidates)
    for _attempt in range(max_attempts):
        if n_placements >= n:
            break
        if supplied is None:
            drawn = sample_background_candidates(index, 1, rng, freq_keys=freq_keys, ping_ranges=ping_ranges)
        else:
            drawn = [c for c in [next(supplied, None)] if c is not None]
        if not drawn:
            break
        cand = drawn[0]
        key = (cand.path, cand.freq_key)
        if key not in datasets:
            datasets[key] = read_survey(cand.path, chunks={"ping": 512}).get_frequency(cand.freq_key).dataset
        items = placed.get(key, {})
        render = _render_pixel if spec.is_pixel else _render_geo
        try:
            draws, across_center = render(datasets[key], cand, spec, items, rng, n_draws, margin_m)
        except KeyError as exc:
            logger.debug(f"background candidate {cand}: missing {exc}, skipping")
            rejected["unavailable"] += 1
            continue
        nadir_lat = float(index[cand.path][cand.freq_key]["lat"][cand.ping])
        nadir_lon = float(index[cand.path][cand.freq_key]["lon"][cand.ping])
        kept = []
        for draw_idx, draw in enumerate(draws):
            if draw is None:
                reason = "unavailable"
            else:
                rendered, footprint, hits = draw
                frac = nan_fraction(rendered.image)
                if hits:
                    reason = "target"
                elif _guard_hits(nadir_lat, nadir_lon, footprint.reach_m + margin_m, catalog_latlon):
                    reason = "geographic_guard"
                elif frac > max_nan_frac:
                    reason = "nan"
                else:
                    reason = None
            if reason is not None:
                rejected[reason] += 1
                if draw_idx == 0:
                    break
                continue
            kept.append(BackgroundWindow(
                index=n_placements, candidate=cand, style=spec.style, draw=draw_idx, across_center_m=across_center,
                footprint=footprint, window=replace(rendered, targets=[]), nan_fraction=frac,
            ))
        if kept:
            accepted.extend(kept)
            n_placements += 1
    logger.debug(f"background {spec.style}/{spec.level}: {n_placements}/{n} accepted, rejected {dict(rejected)}")
    if n_placements < n:
        logger.warning(
            f"background {spec.style}/{spec.level}: only {n_placements}/{n} windows found after {max_attempts} "
            f"candidates (rejected: {dict(rejected)})"
        )
    return accepted
