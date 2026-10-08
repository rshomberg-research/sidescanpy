"""
Display orientation for exported windows, and a display-only range detrend.

Every exported image is oriented to one display convention: pings run
bottom-to-top (the newest ping at the top, matching EdgeTech Discover's
target thumbnails), and a single port channel is mirrored so near range
sits next to nadir on the same side as the starboard view's mirror image.
Each function moves the annotated target positions
(``Target.sample`` = column, ``extra_attributes["window_row"]`` = row)
together with the pixels.

- :func:`orient_for_display` -- one decoded target window, one target.
- :func:`orient_pixel_window` -- a sample/ping-space export window with
  any number of co-located targets.
- :func:`flip_geo_vertical` / :func:`orient_geo_window` -- a resampled
  geometric (along/across meters) window.
- :func:`detrend_by_range` -- display-only per-column median subtraction.

2026 rshomberg@uri.edu
"""

import logging
from dataclasses import replace

import numpy as np

from sidescanpy.core.target import Target

logger = logging.getLogger(__name__)


def detrend_by_range(window: np.ndarray) -> np.ndarray:
    """Subtract each column's median across all pings (display only).

    A flat-field-style correction: removes the typical level at each range
    so only deviations from it (targets, local texture) remain visible.
    It hides a left-to-right brightness gradient, such as a residual left
    by an imperfect TVG model, without correcting it -- not physically
    grounded and not a substitute for radiometric correction.

    Parameters
    ----------
    window : np.ndarray
        ``(n_pings, n_samples)`` array, sample axis last.

    Returns
    -------
    np.ndarray
        Same shape, centered near 0 rather than in the input units; render
        with percentile-based limits, not a fixed absolute scale.
    """
    column_median = np.median(window, axis=0, keepdims=True)
    return window - column_median


def orient_for_display(window: np.ndarray, target: Target) -> tuple[np.ndarray, Target]:
    """Orient a decoded target window to the export display convention.

    Both flips match EdgeTech Discover's own target images (established by
    visual comparison, not from vendor documentation):

    - **Vertical (always):** decoded rows run in ascending ping order; the
      ping axis is reversed so the newest ping is at the top.
    - **Horizontal (port only):** a port window (``target.channel == 0``)
      is mirrored along the sample axis, matching the waterfall sign
      convention (port negative, starboard positive).

    Parameters
    ----------
    window : np.ndarray
        ``(n_pings, n_samples)`` decoded window, sample axis last.
    target : Target
        ``.channel`` controls the horizontal flip (anything but 0 skips it).
        ``extra_attributes["window_row"]`` is remapped if present.

    Returns
    -------
    (window, target)
        Oriented window, and a copy of ``target`` with ``.sample`` and
        ``extra_attributes["window_row"]`` remapped to match.
    """
    oriented_window = window[::-1, :]
    extra_attributes = dict(target.extra_attributes)
    window_row = extra_attributes.get("window_row")
    if window_row is not None:
        extra_attributes["window_row"] = window.shape[0] - 1 - window_row
    oriented_target = replace(target, extra_attributes=extra_attributes)

    if oriented_target.channel == 0:
        n_samples = oriented_window.shape[1]
        oriented_window = oriented_window[:, ::-1]
        mirrored_sample = (
            None if oriented_target.sample is None else (n_samples - 1 - oriented_target.sample)
        )
        oriented_target = replace(oriented_target, sample=mirrored_sample)

    return oriented_window, oriented_target


def orient_pixel_window(
    window: np.ndarray, targets: list[Target], channel_idx: int, style: str,
) -> tuple[np.ndarray, list[Target]]:
    """Orient a sample/ping-space export window and every target in it.

    Same two conventions as :func:`orient_for_display`, applied to a window
    that can carry several co-located targets:

    - **Vertical (always):** row 0 of an extracted window is the earliest
      ping; the ping axis is reversed.
    - **Horizontal (single port channel only):** ``channel_idx == 0`` and
      ``style != "dual-swath"`` mirrors the sample axis. A dual-swath
      window is already port-mirrored by
      :func:`~sidescanpy.export.windows.combine_channels_mirrored`.

    Targets use integer pixel-center coordinates, so a flip maps
    ``x -> n - 1 - x``.

    Parameters
    ----------
    window : np.ndarray
        ``(n_rows, n_cols)`` window, rows = pings.
    targets : list of Target
        Window-local positions (``sample`` = column,
        ``extra_attributes["window_row"]`` = row).
    channel_idx : int
        Physical channel the window was cut from (0 = port).
    style : str
        Export style (``"chip"``, ``"single-swath"`` or ``"dual-swath"``).

    Returns
    -------
    (window, targets)
    """
    n_rows, n_cols = window.shape
    window = window[::-1, :]
    targets = [
        replace(ct, extra_attributes={**ct.extra_attributes, "window_row": n_rows - 1 - ct.extra_attributes["window_row"]})
        for ct in targets
    ]
    if style != "dual-swath" and channel_idx == 0:
        window = window[:, ::-1]
        targets = [replace(ct, sample=n_cols - 1 - ct.sample) for ct in targets]
    return window, targets


def flip_geo_vertical(grid: np.ndarray, targets: list[Target], along_size_px: int) -> tuple[np.ndarray, list[Target]]:
    """Reverse the along-track axis of a geometric window.

    ``extract_geometric_window*`` build row 0 at the minimum along-track
    distance (earliest ping); this applies the same newest-ping-on-top
    convention as :func:`orient_pixel_window`. Target rows are continuous
    pixel-edge coordinates (see
    :func:`~sidescanpy.export.geometric_window.locate_point`), so a row maps
    to ``along_size_px - row``.

    Parameters
    ----------
    grid : np.ndarray
        ``(along_size_px, n_cols)`` resampled window.
    targets : list of Target
        Window-local positions (see :func:`orient_pixel_window`).
    along_size_px : int
        The grid's row count (not its column count -- they differ for a
        non-square grid).

    Returns
    -------
    (grid, targets)
    """
    grid = grid[::-1, :]
    targets = [
        replace(pt, extra_attributes={**pt.extra_attributes, "window_row": along_size_px - pt.extra_attributes["window_row"]})
        for pt in targets
    ]
    return grid, targets


def orient_geo_window(
    grid: np.ndarray, targets: list[Target], size_px: int, *, combine: bool, channel_idx: int | None,
    along_size_px: int | None = None,
) -> tuple[np.ndarray, list[Target]]:
    """Orient a geometric (along/across meters) window and its targets.

    Always applies :func:`flip_geo_vertical`. A single port channel's
    window (``combine=False``, ``channel_idx == 0``) is also mirrored
    horizontally: columns are built in ascending ``across_track_m``, which
    is negative on port, so unmirrored port windows put near range on the
    right while starboard windows put it on the left. A combined
    (dual-swath) window spans both signs and already has nadir in the
    middle, so it is never mirrored.

    The horizontal flip uses ``n - 1 - x`` on the target column, matching
    the long-standing export output.

    Parameters
    ----------
    grid : np.ndarray
        Resampled window, rows = along-track.
    targets : list of Target
        Window-local positions.
    size_px : int
        Column count; also the row count unless ``along_size_px`` is given.
    combine : bool
        Whether the window pools both channels.
    channel_idx : int or None
        Channel of a single-channel window (``None`` when ``combine``).
    along_size_px : int, optional
        Row count, when it differs from ``size_px``.

    Returns
    -------
    (grid, targets)
    """
    grid, targets = flip_geo_vertical(grid, targets, along_size_px if along_size_px is not None else size_px)
    if not combine and channel_idx == 0:
        n_cols = grid.shape[1]
        grid = grid[:, ::-1]
        targets = [replace(pt, sample=n_cols - 1 - pt.sample) for pt in targets]
    return grid, targets
