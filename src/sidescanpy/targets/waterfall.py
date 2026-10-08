"""
Target boxes in waterfall plot coordinates.

A waterfall is drawn either in pixel space (``x_axis="sample"``,
``y_axis="ping"``: port samples negative, starboard positive, the global
ping index up the y-axis) or in meters (``x_axis="across_track"``,
``y_axis="along_track"``). :func:`waterfall_boxes` places each target in
whichever of the two is active and returns plain rectangles for a plotting
layer to draw.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.targets.correlate import CatalogPlacement

logger = logging.getLogger(__name__)

BOX_AXES = (("sample", "ping"), ("across_track", "along_track"))
"""``(x_axis, y_axis)`` pairs :func:`waterfall_boxes` can place targets on."""


class WaterfallBox(NamedTuple):
    """A labeled rectangle in waterfall axis units (lower-left corner + size)."""

    label: str
    x0: float
    y0: float
    width: float
    height: float


def mean_sample_spacing_m(ds: xr.Dataset, local_ping: int, channel_idx: int) -> float:
    """Mean across-track meters per sample over one ping's finite samples.

    Averaged over the whole ping rather than a local difference, so it is
    stable near the water-column boundary. ``1.0`` with fewer than two
    finite samples.
    """
    at = ds["across_track_m"].isel(ping=local_ping, channel=channel_idx).values
    finite = at[np.isfinite(at)]
    if finite.size < 2:
        return 1.0
    return float(np.abs(np.diff(finite)).mean())


def median_ping_spacing_m(along_track: np.ndarray | None) -> float:
    """Median along-track meters per ping (robust to turns); ``1.0`` if undefined."""
    if along_track is None or along_track.size < 2:
        return 1.0
    diffs = np.abs(np.diff(along_track))
    return float(np.median(diffs)) if diffs.size else 1.0


def target_waterfall_position(
    target: Target, ds: xr.Dataset | None, ping_idx: int, ping_offset: int, x_axis: str, y_axis: str,
    along_track: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    """Where ``target`` lands on a waterfall's axes, and the axis scale.

    Parameters
    ----------
    target : Target
        Needs ``sample``; ``channel`` defaults to port.
    ds : xr.Dataset or None
        The plotted window (only read for the meters axes).
    ping_idx : int
        The target's global ping index (the pixel y-axis value).
    ping_offset : int
        Global ping index of ``ds``'s first ping.
    x_axis, y_axis : str
        One of :data:`BOX_AXES`.
    along_track : np.ndarray, optional
        Along-track meters per ``ds`` ping (meters axes; ``None`` falls
        back to the local ping index).

    Returns
    -------
    (x, y, x_scale, y_scale) or None
        ``x_scale``/``y_scale`` convert one pixel (sample, ping) into axis
        units: 1.0 on the pixel axes; meters per sample
        (:func:`mean_sample_spacing_m`) and per ping
        (:func:`median_ping_spacing_m`) on the meters axes. ``None`` when
        the target has no sample, lies outside ``ds``, has no finite
        across-track position, or the axes are unsupported.
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
        x_scale = mean_sample_spacing_m(ds, local_ping, channel_idx)
        y_scale = median_ping_spacing_m(along_track)
        return x_center, y_center, x_scale, y_scale

    return None


def waterfall_boxes(
    ds: xr.Dataset, boxes: Sequence[tuple[Target, int]], ping_offset: int, x_axis: str, y_axis: str, box_size: float,
) -> list[WaterfallBox]:
    """Rectangles for every ``(target, global_ping_idx)`` placeable on these axes.

    ``box_size`` is in pixels (samples x pings) and is scaled into axis
    units per target (see :func:`target_waterfall_position`).
    """
    along_track = along_track_distance_m(ds) if boxes and y_axis == "along_track" else None
    rects = []
    for target, ping_idx in boxes:
        xy = target_waterfall_position(target, ds, ping_idx, ping_offset, x_axis, y_axis, along_track)
        if xy is None:
            continue
        x_center, y_center, x_scale, y_scale = xy
        half_x = (box_size / 2.0) * x_scale
        half_y = (box_size / 2.0) * y_scale
        rects.append(WaterfallBox(str(target.target_id), x_center - half_x, y_center - half_y, 2 * half_x, 2 * half_y))
    return rects


def targets_by_global_ping(
    placements: Sequence[CatalogPlacement], freq_key: str, file_start_ping: dict[Path, int],
) -> dict[int, list[Target]]:
    """Targets on ``freq_key``, keyed by their ping in a concatenated survey.

    Parameters
    ----------
    placements : sequence of CatalogPlacement
        From :func:`~sidescanpy.targets.correlate.correlate_catalog`.
    freq_key : str
    file_start_ping : dict
        ``{path: first global ping}`` of each concatenated file (files
        without ``freq_key`` are absent).
    """
    by_ping: dict[int, list[Target]] = defaultdict(list)
    for placement in placements:
        path = placement.match.path
        if path not in file_start_ping or freq_key not in placement.on_frequency:
            continue
        by_ping[file_start_ping[path] + placement.match.ping_idx].append(placement.on_frequency[freq_key])
    return by_ping
