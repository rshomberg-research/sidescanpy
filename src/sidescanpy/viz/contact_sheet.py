"""
Thumbnail grid and boxed-context QA rendering for sonar targets.

Uses the same percentile-based amplitude
auto-scaling convention as ``viz/waterfall.py`` (default 1st/99th
percentile) so target imagery reads consistently with other sidescanpy
figures.

2026 rshomberg@uri.edu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from sidescanpy.core.target import Target, bbox_from_center
from sidescanpy.viz._cmap import resolve_cmap

logger = logging.getLogger(__name__)

_DEFAULT_CMAP = "copper"
_DEFAULT_VMIN_PCT = 1.0
_DEFAULT_VMAX_PCT = 99.0


def _target_box_size(t: Target, box_size: tuple[float, float]) -> tuple[float, float]:
    """The target's own measured box (``extra_attributes["bbox_size_px"]``,
    set by ``sidescan-export chip`` from a catalog extent, in this
    window's pixels) if it has one, else the shared placeholder -- the
    same choice COCO/YOLO export makes, so QA images draw what the
    annotation files contain."""
    size = t.extra_attributes.get("bbox_size_px")
    return (float(size[0]), float(size[1])) if size is not None else box_size


def _vlimits(image: np.ndarray, vmin_pct: float, vmax_pct: float) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("No finite values in image; cannot auto-scale color")
    return float(np.percentile(finite, vmin_pct)), float(np.percentile(finite, vmax_pct))


def contact_sheet(
    images: list[np.ndarray],
    captions: list[str],
    *,
    boxes: list[list[Target]] | None = None,
    box_size: tuple[float, float] = (40.0, 40.0),
    ncols: int = 6,
    cmap: str = _DEFAULT_CMAP,
    vmin_pct: float = _DEFAULT_VMIN_PCT,
    vmax_pct: float = _DEFAULT_VMAX_PCT,
    cell_size: tuple[float, float] = (2.2, 2.2),
    suptitle: str | None = None,
) -> Figure:
    """Grid of thumbnails, one per image, captioned individually.

    Each image is auto-scaled independently (its own percentile range) —
    appropriate for a review grid of unrelated crops, not for comparing
    absolute intensity across targets.

    Parameters
    ----------
    images : list of 2D arrays
        One per target.
    captions : list of str
        Same length as ``images``.
    boxes : list of list of Target, optional
        Same length as ``images`` if given — every target to draw a box
        for on that thumbnail (``.sample``/``extra_attributes["window_row"]``,
        the same convention ``bbox_from_center``/COCO/YOLO already use).
        An empty list for an entry draws no box on that thumbnail.
        Omit entirely for a plain, unboxed grid (e.g. no-target negative
        samples, which have nothing to box by definition).
    box_size : tuple, default (40.0, 40.0)
        Shared placeholder box size — same meaning as everywhere else in
        this codebase (no measured target extent exists in any source
        catalog seen so far).
    ncols : int, default 6
        Grid columns; rows are computed to fit.
    """
    if len(images) != len(captions):
        raise ValueError(f"images ({len(images)}) and captions ({len(captions)}) must be the same length")
    if boxes is not None and len(boxes) != len(images):
        raise ValueError(f"boxes ({len(boxes)}) must be the same length as images ({len(images)})")
    n = len(images)
    ncols = max(1, min(ncols, n)) if n else 1
    nrows = max(1, -(-n // ncols)) if n else 1

    resolved_cmap = resolve_cmap(cmap)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_size[0] * ncols, cell_size[1] * nrows), squeeze=False
    )
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            image = images[idx]
            vmin, vmax = _vlimits(image, vmin_pct, vmax_pct)
            # aspect="equal" -- "auto" (the prior default) stretches an
            # image to fill whatever box shape the grid layout happens to
            # produce for that cell, which can visibly distort a
            # genuinely square array (found 2026-09-17: every underlying
            # chip file was confirmed square on disk, but contact sheets
            # made some look stretched -- a display bug here, not a data
            # bug in the export itself).
            ax.imshow(image, cmap=resolved_cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(captions[idx], fontsize=8)
            for t in (boxes[idx] if boxes else []):
                row = t.extra_attributes.get("window_row")
                col = t.sample
                if row is None or col is None:
                    continue
                x, y, w, h = bbox_from_center(row, col, _target_box_size(t, box_size))
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.0))
        ax.set_xticks([])
        ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


_CONTEXT_AXIS_LABELS = {
    "pixel": ("Sample", "Ping (window-local)"),
    "geometric": ("Across-track (m, window-local)", "Along-track (m, window-local)"),
    "georef": ("Easting (m, window-local)", "Northing (m, window-local)"),
}


def render_target_context(
    window: np.ndarray,
    target: Target,
    *,
    box_size: tuple[float, float] = (40.0, 40.0),
    cmap: str = _DEFAULT_CMAP,
    vmin_pct: float = _DEFAULT_VMIN_PCT,
    vmax_pct: float = _DEFAULT_VMAX_PCT,
    title: str | None = None,
    coordinate_system: str = "pixel",
    resolution_m: float | None = None,
) -> Figure:
    """Full ping-window image with a box drawn at ``target``'s computed
    position, for visual QA of the localization formula.

    ``box_size`` is a fixed placeholder (width, height in samples/pings
    -- or, under a real-meters ``coordinate_system``, output pixels of
    that resample, i.e. ``box_size * resolution_m`` meters) around the
    computed center — no real measured target dimensions exist in the
    source catalog (``LengthMeters``/``WidthMeters`` unpopulated), so
    this is explicitly not derived from anything measured.

    ``coordinate_system`` -- one of ``"pixel"`` (default, native
    sample/ping space -- ``export.py``'s ``raw``/``radiometric``
    render levels), ``"geometric"`` (``extract_geometric_window``'s
    local along/across-track meters frame), or ``"georef"``
    (``extract_georef_window``'s northing/easting meters frame) --
    controls only the axis labels below; ``window`` itself must already
    be built in that coordinate system by the caller. Found real
    (2026-09-28, user report): this always labeled axes "Sample"/"Ping"
    even when ``--with-context`` rendered a geometric/georef real-meters
    resample (added 2026-09-28 to follow ``--render-level``'s own
    coordinate choice) -- correct-looking image, mislabeled axes.
    """
    x_label, y_label = _CONTEXT_AXIS_LABELS.get(coordinate_system, _CONTEXT_AXIS_LABELS["pixel"])
    # Geometric/georef axes are only in meters when the window's
    # resolution is known (sidescan-export chip passes it); otherwise
    # say what they really count -- output pixels.
    if coordinate_system != "pixel" and resolution_m is None:
        x_label = x_label.replace("(m, ", "(px, ")
        y_label = y_label.replace("(m, ", "(px, ")
    scale = resolution_m if (coordinate_system != "pixel" and resolution_m is not None) else 1.0
    row = target.extra_attributes.get("window_row")
    col = target.sample
    if row is None or col is None:
        raise ValueError(
            f"Target {target.target_id} has no computed window position "
            "(missing extra_attributes['window_row'] or .sample)"
        )

    vmin, vmax = _vlimits(window, vmin_pct, vmax_pct)

    # Sample-index count and ping-index count aren't the same physical
    # unit (sample spacing depends on sound velocity/sample rate, ping
    # spacing on vessel speed/ping rate) — so there's no single "true"
    # aspect ratio to preserve here, aspect="equal" (1 sample-unit == 1
    # ping-unit) is just as arbitrary as any other fixed ratio, and for a
    # full ~4800-sample x ~200-ping window it produces an unusably thin
    # ~24:1 strip. The previous hardcoded figsize=(10, 8) (~1.25:1) was a
    # *different* arbitrary choice that happened to look badly distorted/
    # zoomed-in for a wide window. Matching EdgeTech Discover's own
    # per-target images instead (found 2026-07-30 by comparing output
    # directly) — theirs render the same full sample width at a
    # deliberately compressed, reasonable landscape aspect (~512x200,
    # ~2.56:1) rather than 1:1 or square — so this uses a comparable
    # fixed aspect instead of trying to derive a "correct" one.
    # A geometric/georef window is already square in real meters -- show
    # it undistorted; the fixed landscape aspect is for native pixels only.
    if coordinate_system == "pixel":
        fig, ax = plt.subplots(figsize=(5.0 * 2.56, 5.0))
        ax.imshow(window, cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax, aspect="auto")
    else:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        n_rows, n_cols = window.shape[:2]
        ax.imshow(window, cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax, aspect="equal",
                  extent=(0, n_cols * scale, n_rows * scale, 0))

    x, y, w, h = bbox_from_center(row, col, _target_box_size(target, box_size))
    ax.add_patch(Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False, edgecolor="red", linewidth=1.5))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # A geometric/georef target's .sample is its column in the resampled
    # window, not a sensor sample index -- leave it out of the title there.
    where = f"ping {target.ping}, sample {target.sample}" if coordinate_system == "pixel" else f"ping {target.ping}"
    ax.set_title(title or f"Target {target.target_id}  ({where}, confidence={target.position_confidence})")
    fig.tight_layout()
    return fig
