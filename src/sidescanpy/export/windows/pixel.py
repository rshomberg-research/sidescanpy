"""
Sample/ping pixel-space windows: chips, full-channel views, co-located
targets, and the per-style window used by the ``uncorrected``/``db``/
``radiometric`` render levels.

The low-level functions operate on plain 2D ``(n_pings, n_samples)``
arrays -- a caller has already selected the channel and backscatter
variable (e.g. ``ds["backscatter_intensity"].isel(channel=0).values``).
Row = ping index, column = sample index throughout.
:func:`extract_pixel_style_window` works on a loaded, nadir-masked ping
neighborhood and handles all three styles.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from sidescanpy.core.target import Target
from sidescanpy.export.orientation import orient_pixel_window
from sidescanpy.export.windows.levels import LEVEL_VARS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowExtent:
    """A window's placement in its source array's native (row, col) space."""

    row_start: int
    col_start: int
    n_rows: int
    n_cols: int


def _place_window_1d(length: int, center: float, window_size: int, jitter_px: int, rng: np.random.Generator) -> int:
    """Clamped start index for a ``window_size``-long window in a
    ``length``-long axis, centered on ``center`` (+ uniform jitter),
    shifted inward near an edge rather than zero-padded.

    ``jitter_px=0`` (default) reproduces exact centering.
    """
    if jitter_px:
        center = center + rng.integers(-jitter_px, jitter_px + 1)
    start = int(round(center - window_size / 2))
    return max(0, min(start, max(0, length - window_size)))


def extract_chip(
    array: np.ndarray,
    center_row: float,
    center_col: float,
    size_px: int,
    *,
    jitter_px: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, WindowExtent]:
    """Fixed ``size_px x size_px`` crop around ``(center_row, center_col)``.

    Non-padding placement: the window shifts inward near an array edge
    instead of zero-padding, trading perfect centering for "always full
    real data." ``jitter_px`` (default 0, exact centering) randomizes
    which point the window centers on, so a detector doesn't learn "the
    target is always dead-center" — the true target position is
    unaffected, only where the window lands moves.

    If ``array`` is smaller than ``size_px`` in a given axis, the full
    available extent on that axis is returned instead (non-square
    output) rather than padding — confirmed against real per-target
    survey data where a source catalog's own decoded window can have a
    hard ping-count ceiling well under a requested chip size. Every
    caller downstream (``CocoImageEntry``, ``write_coco``/``write_yolo``)
    already carries per-image ``width``/``height`` explicitly, so a
    dataset with mixed image sizes from this cause is handled correctly,
    not silently mis-sized.

    Returns
    -------
    (window, extent)
    """
    if rng is None:
        rng = np.random.default_rng()
    n_rows, n_cols = array.shape
    row_start = _place_window_1d(n_rows, center_row, size_px, jitter_px, rng)
    col_start = _place_window_1d(n_cols, center_col, size_px, jitter_px, rng)
    window = array[row_start : row_start + size_px, col_start : col_start + size_px]
    extent = WindowExtent(row_start=row_start, col_start=col_start, n_rows=window.shape[0], n_cols=window.shape[1])
    return window, extent


def extract_full_channel(
    array: np.ndarray,
    *,
    center_ping: float | None = None,
    square: bool = True,
    fixed_ping_count: int = 200,
    jitter_px: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, WindowExtent]:
    """Full-sample-axis window spanning a fixed ping span, at native
    resolution — **not resized here**. Resize (via :func:`resize_to`)
    belongs *after* a photometric transform (e.g.
    :func:`sidescanpy.export.image.to_uint8`) has run on the full
    native-resolution values, not before it: computing a percentile
    stretch on an already-downsampled array measures the stretch against
    a blurred, artificially low-variance distribution and washes the
    result out (found the hard way — see this function's own history).
    Every other statistics-computing function in this library (waterfall/
    gridded percentile scaling, ``resample_channel_to_grid``'s cell
    aggregation) works the same way: aggregate/stretch against real
    per-sample values first, resize the *result* for display second.

    ``square=True`` (default): the native ping span equals the channel's
    own sample count, not a fixed ping count against a differing sample
    count — avoids the aspect-ratio stretch a fixed ping count would
    otherwise introduce once the eventual square output is produced.
    ``square=False`` uses ``fixed_ping_count`` pings instead.

    ``center_ping`` defaults to the array's own midpoint (a full-channel
    negative sample has no target to center on); pass a target's ping
    position for a positive sample. ``jitter_px`` behaves as in
    :func:`extract_chip`.

    Known, deliberately deferred edge case: a survey shorter than the
    native ping span raises ``ValueError`` rather than guessing a
    workaround (e.g. padding or falling back to non-square) — revisit if
    a real survey actually hits this.

    Returns
    -------
    (window, extent)
        Native resolution — pass through a photometric transform, then
        :func:`resize_to`, in that order.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_pings, n_samples = array.shape
    native_ping_count = n_samples if square else fixed_ping_count
    if native_ping_count > n_pings:
        raise ValueError(
            f"survey has {n_pings} pings, fewer than the {native_ping_count}-ping "
            "native window this full-channel export needs (square=True uses the "
            "channel's own sample count) — deliberately not handled, see "
            "extract_full_channel's docstring"
        )
    if center_ping is None:
        center_ping = n_pings / 2

    row_start = _place_window_1d(n_pings, center_ping, native_ping_count, jitter_px, rng)
    window = array[row_start : row_start + native_ping_count, :]
    extent = WindowExtent(row_start=row_start, col_start=0, n_rows=window.shape[0], n_cols=window.shape[1])
    return window, extent


def resize_to(image: np.ndarray, output_size_px: int) -> np.ndarray:
    """Resize an already-photometrically-transformed 2D image (e.g. the
    uint8 output of :func:`sidescanpy.export.image.to_uint8`) to a
    square ``output_size_px x output_size_px`` — the deliberately-last
    step after :func:`extract_full_channel` + a percentile/gamma stretch,
    never before it (see that function's docstring for why the order
    matters). Bilinear resampling, matching this project's own established
    resize-after-transform precedent.
    """
    return zoom(image, (output_size_px / image.shape[0], output_size_px / image.shape[1]), order=1)


def sample_negative_window(
    n_rows: int,
    n_cols: int,
    target_positions: list[tuple[float, float]],
    size_px: int,
    min_distance_px: float,
    rng: np.random.Generator,
    *,
    max_attempts: int = 200,
) -> tuple[float, float]:
    """Pick a ``(row, col)`` window center at least ``min_distance_px``
    from every position in ``target_positions`` — for no-target negative
    samples.

    ``target_positions`` is a plain list of ``(row, col)`` tuples, not
    ``Target`` objects, so callers don't need to build one just to check
    distances.

    Raises
    ------
    RuntimeError
        If no valid location is found within ``max_attempts`` — a dense
        target catalog relative to ``n_rows``/``n_cols`` and
        ``min_distance_px`` can make this genuinely infeasible; fail
        loudly rather than silently returning a too-close location.
    """
    half = size_px / 2
    row_bounds = (half, max(half, n_rows - half))
    col_bounds = (half, max(half, n_cols - half))
    for _ in range(max_attempts):
        row = rng.uniform(*row_bounds)
        col = rng.uniform(*col_bounds)
        if all(np.hypot(row - r, col - c) >= min_distance_px for r, c in target_positions):
            return row, col
    raise RuntimeError(
        f"could not find a negative window >= {min_distance_px}px from every cataloged "
        f"target after {max_attempts} attempts"
    )


def targets_within_window(targets: list[Target], extent: WindowExtent) -> list[Target]:
    """Every target from ``targets`` whose position falls inside
    ``extent``, translated to window-local coordinates.

    Reuses the existing ``Target.sample`` (column) /
    ``extra_attributes["window_row"]`` (row) convention used by the chip exporters. Targets missing either
    field are skipped (nothing to place). For multi-target images: every
    included target becomes its own annotation on the same exported
    image, not just the one a chip happened to be centered on.
    """
    included = []
    for t in targets:
        row = t.extra_attributes.get("window_row")
        col = t.sample
        if row is None or col is None:
            continue
        if extent.row_start <= row < extent.row_start + extent.n_rows and (
            extent.col_start <= col < extent.col_start + extent.n_cols
        ):
            local = replace(
                t,
                sample=col - extent.col_start,
                extra_attributes={**t.extra_attributes, "window_row": row - extent.row_start},
            )
            included.append(local)
    return included


def rescale_target_position(
    row: float, col: float, native_shape: tuple[int, int], output_shape: tuple[int, int]
) -> tuple[float, float]:
    """Translate a ``(row, col)`` position from a window's native (pre-
    downsample) space into its resized output space — needed after
    :func:`extract_full_channel`, whose returned window is already
    downsampled while :func:`targets_within_window` reports positions in
    the native extent.
    """
    row_scale = output_shape[0] / native_shape[0]
    col_scale = output_shape[1] / native_shape[1]
    return row * row_scale, col * col_scale


@dataclass
class RenderedWindow:
    """One rendered export window, before the 8/16-bit display transform.

    Attributes
    ----------
    image : np.ndarray
        Display-oriented float window at its output size.
    targets : list of Target
        Targets inside the window, in ``image``'s pixel coordinates
        (``sample`` = column, ``extra_attributes["window_row"]`` = row).
    extent : WindowExtent or None
        Pixel levels: the window's placement in the file's ping axis and
        the style's column space (single-channel samples, or the combined
        columns of :func:`combine_channels_mirrored`), before orientation.
    meta : dict or None
        Geometric/georef levels: the extractor's ``meta`` (grid origin and
        resolution) for locating further points in the window.
    grid_shape : tuple of int or None
        Geometric/georef levels: ``(n_rows, n_cols)`` of the resampled grid.
    resolution_m : float or None
        Geometric/georef levels: meters per output pixel.
    """

    image: np.ndarray
    targets: list[Target] = field(default_factory=list)
    extent: WindowExtent | None = None
    meta: dict | None = None
    grid_shape: tuple[int, int] | None = None
    resolution_m: float | None = None


def combine_channels_mirrored(port_2d: np.ndarray, starboard_2d: np.ndarray) -> np.ndarray:
    """Port (sample axis reversed) and starboard side by side, nadir in the middle.

    A plain concatenation of the two pixel arrays, not a resample into a
    shared real-world frame (that is what the geometric/georef levels do).
    """
    return np.concatenate([port_2d[:, ::-1], starboard_2d], axis=1)


def target_combined_col(sample: int, n_samples_single: int, channel_idx: int) -> int:
    """Column of a single-channel ``sample`` in :func:`combine_channels_mirrored`'s output."""
    if channel_idx == 0:
        return n_samples_single - 1 - sample
    return n_samples_single + sample


def rescale_targets(targets: list[Target], native_shape: tuple[int, int], output_shape: tuple[int, int]) -> list[Target]:
    """Move every target's ``(row, col)`` from a native window into its resized output.

    Matches :func:`resize_to`'s transform; see :func:`rescale_target_position`.
    """
    rescaled = []
    for ct in targets:
        row, col = rescale_target_position(ct.extra_attributes["window_row"], ct.sample, native_shape, output_shape)
        rescaled.append(replace(ct, sample=col, extra_attributes={**ct.extra_attributes, "window_row": row}))
    return rescaled


def with_bbox_size_px(target: Target, scale_x: float, scale_y: float) -> Target:
    """Record ``target.bbox``'s measured ``(w, h)`` in this window's output pixels.

    ``scale_x``/``scale_y`` convert from the bbox's native units (source
    sample/ping pixels) into output pixels -- 1:1 for an unresized pixel
    window; meters-per-sample (or per ping) over ``resolution_m`` for a
    geometric/georef window. The result goes in
    ``extra_attributes["bbox_size_px"]``, which COCO/YOLO writers use
    instead of the shared placeholder box size.

    Returns ``target`` unchanged when it has no ``bbox`` (a center-only
    pick keeps the placeholder).
    """
    if target.bbox is None:
        return target
    _x, _y, w, h = target.bbox
    return replace(target, extra_attributes={**target.extra_attributes, "bbox_size_px": (float(w) * scale_x, float(h) * scale_y)})


def place_targets_in_file(
    items_by_channel: dict[int, list[tuple[Target, int]]], n_samples_single: int,
) -> tuple[dict[int, list[Target]], list[Target]]:
    """File-coordinate target lists for :func:`extract_pixel_style_window`'s co-target lookup.

    Parameters
    ----------
    items_by_channel : dict
        ``{channel_idx: [(target, file_ping_idx), ...]}``.
    n_samples_single : int
        Samples per channel.

    Returns
    -------
    (placed_by_channel, placed_combined)
        Per channel, each target with ``extra_attributes["window_row"]`` =
        its file ping (column = its own sample); and one list over both
        channels in :func:`combine_channels_mirrored`'s column space.
    """
    placed_by_channel = {
        ch: [replace(t, extra_attributes={**t.extra_attributes, "window_row": ping_idx}) for t, ping_idx in its]
        for ch, its in items_by_channel.items()
    }
    placed_combined = [
        replace(t, sample=target_combined_col(t.sample, n_samples_single, ch), extra_attributes={**t.extra_attributes, "window_row": ping_idx})
        for ch, its in items_by_channel.items()
        for t, ping_idx in its
    ]
    return placed_by_channel, placed_combined


def extract_pixel_style_window(
    masked: xr.Dataset,
    *,
    center_sample: int,
    channel_idx: int,
    local_ping: int,
    ping_offset: int,
    style: str,
    level: str,
    size_px: int,
    jitter_px: int,
    rng: np.random.Generator,
    placed_by_channel: dict[int, list[Target]],
    placed_combined: list[Target],
    require_targets: bool = True,
) -> RenderedWindow | None:
    """Sample/ping-space window for one style, with the targets inside it.

    ``chip`` crops ``size_px`` square around ``(local_ping, center_sample)``
    on ``channel_idx``; ``single-swath`` takes that channel's full sample
    axis over a square native ping span; ``dual-swath`` does the same on
    both channels combined (:func:`combine_channels_mirrored`). The window
    is oriented (:func:`~sidescanpy.export.orientation.orient_pixel_window`)
    and resized to ``size_px`` square if needed, with every co-located
    target moved to match and its measured bbox recorded
    (:func:`with_bbox_size_px`).

    Parameters
    ----------
    masked : xr.Dataset
        Nadir-masked ping neighborhood (see
        :func:`~sidescanpy.export.windows.levels.mask_nadir_gap_with_uncorrected`).
    center_sample : int
        Window center on ``channel_idx``'s own sample axis.
    channel_idx : int
        Channel of the window (and of ``center_sample``).
    local_ping : int
        Center ping, indexed into ``masked``.
    ping_offset : int
        File ping index of ``masked``'s first ping (placed targets carry
        file ping indices).
    style, level : str
        See :mod:`sidescanpy.export.windows.levels`.
    size_px : int
        Output size (both axes).
    jitter_px : int
        Uniform center jitter, drawn from ``rng``.
    rng : np.random.Generator
    placed_by_channel, placed_combined
        From :func:`place_targets_in_file`.
    require_targets : bool, default True
        Return ``None`` when no target falls in the window. ``False`` for
        background windows.

    Returns
    -------
    RenderedWindow or None
        ``None`` when ``level``'s variable is absent from ``masked``, the
        neighborhood is too short for a full-swath style's square native
        window, or (``require_targets``) no target is inside.
    """
    value_var = LEVEL_VARS[level]
    if value_var not in masked:
        return None
    n_samples_single = masked.sizes["sample"]

    if style == "dual-swath":
        port_2d = masked[value_var].isel(channel=0).values
        stbd_2d = masked[value_var].isel(channel=1).values
        arr_2d = combine_channels_mirrored(port_2d, stbd_2d)
        center_col = target_combined_col(center_sample, n_samples_single, channel_idx)
        co_source = placed_combined
    else:
        arr_2d = masked[value_var].isel(channel=channel_idx).values
        center_col = center_sample
        co_source = placed_by_channel.get(channel_idx, [])

    try:
        if style == "chip":
            window, extent = extract_chip(arr_2d, center_row=local_ping, center_col=center_col, size_px=size_px, jitter_px=jitter_px, rng=rng)
        else:
            window, extent = extract_full_channel(arr_2d, center_ping=float(local_ping), jitter_px=jitter_px, rng=rng)
    except ValueError:
        return None

    extent_file = WindowExtent(row_start=extent.row_start + ping_offset, col_start=extent.col_start, n_rows=extent.n_rows, n_cols=extent.n_cols)
    co_targets = targets_within_window(co_source, extent_file)
    if require_targets and not co_targets:
        return None
    window, co_targets = orient_pixel_window(window, co_targets, channel_idx, style)

    native_shape = window.shape
    # The window is a crop of source pixels, so t.bbox's sample/ping units
    # map 1:1 until the resize below.
    row_scale, col_scale = 1.0, 1.0
    if native_shape != (size_px, size_px):
        row_scale, col_scale = size_px / native_shape[0], size_px / native_shape[1]
        co_targets = rescale_targets(co_targets, native_shape, (size_px, size_px))
        window = resize_to(window, size_px)
    co_targets = [with_bbox_size_px(ct, scale_x=col_scale, scale_y=row_scale) for ct in co_targets]
    return RenderedWindow(image=window, targets=co_targets, extent=extent_file)
