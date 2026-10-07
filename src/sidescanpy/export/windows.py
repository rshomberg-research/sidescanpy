"""
Window extraction for dataset-export images: chips, full-channel views,
and no-target negative sampling.

Operates on plain 2D ``(n_pings, n_samples)`` arrays — a caller has
already selected which channel/backscatter variable to read from a
corrected survey (e.g. ``ds["backscatter_intensity"].isel(channel=0).values``).
Row = ping index, column = sample index throughout.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import zoom

from sidescanpy.targets.core import Target


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
    :func:`sidescanpy.export.eightbit.to_uint8`) has run on the full
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
    uint8 output of :func:`sidescanpy.export.eightbit.to_uint8`) to a
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
