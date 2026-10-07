"""
Target dataclass — the format-agnostic sonar contact model.

Per ``docs/planning/v0.10.0-target-catalog-and-coco-export.md`` Phase 1.
A target's position may be known in the native address space (ping/
channel/sample, lat/lon), the pixel address space (image id + bbox), both,
or neither, depending on what the source catalog actually gives up.

2026 rshomberg@uri.edu
"""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

PositionConfidence = Literal["unknown", "approximate", "verified"]


@dataclass
class Target:
    """One sonar target/contact, in native and/or pixel address space.

    Parameters
    ----------
    target_id : str
        Identifier from the source catalog (e.g. a filename-derived
        number). Not assumed globally unique across catalogs — see
        ``catalog_source`` for provenance.
    source_format : str
        e.g. ``"jsf_discover"``, ``"sdf_sonarwiz"``.
    catalog_source : str
        Path to the original catalog file this target came from.
    pick_timestamp : float, optional
        Unix epoch seconds, when trustworthy.
    ping, channel, sample : int, optional
        Native (ping/sample) address space. ``channel``: 0=port, 1=starboard.
    latitude, longitude : float, optional
        Degrees, WGS84.
    image_id : str, optional
        Pixel address space: identifies the rendered image this target
        was localized on (populated once a crop/render exists).
    bbox : tuple[float, float, float, float], optional
        ``(x, y, w, h)`` in ``image_id``'s pixel space.
    position_confidence : {"unknown", "approximate", "verified"}
        Whether the position (native and/or pixel) has been visually
        validated. Export code should default to filtering on this
        rather than silently shipping unlocalized or unverified boxes.
    label : str, optional
        Free text, passed through, not interpreted.
    category : str, optional
        Class name for multi-class annotation export (COCO/YOLO). ``None``
        (the default) means "uncategorized": the export writers then use
        their single default category name.
    extra_attributes : dict
        Format-specific fields that don't generalize (e.g. JSF's
        MutableTag text, or notes about data-quality workarounds
        applied while reading this target).
    """

    target_id: str
    source_format: str
    catalog_source: str
    pick_timestamp: float | None = None

    ping: int | None = None
    channel: int | None = None
    sample: int | None = None
    latitude: float | None = None
    longitude: float | None = None

    image_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    position_confidence: PositionConfidence = "unknown"
    label: str | None = None
    category: str | None = None
    extra_attributes: dict[str, Any] = field(default_factory=dict)


CONFIDENCE_RANK = {"unknown": 0, "approximate": 1, "verified": 2}


def clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[tuple[float, float, float, float], bool]:
    """Clip a ``(x, y, w, h)`` pixel box to an image's bounds.

    Shared by ``interchange/coco.py`` and ``interchange/yolo.py`` so both
    formats agree on exactly the same clipped box for a given target.

    Returns
    -------
    (clipped_bbox, was_clipped)
    """
    x, y, w, h = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    clipped = (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))
    return clipped, clipped != bbox


def bbox_from_center(row: float, col: float, box_size: tuple[float, float]) -> tuple[float, float, float, float]:
    """Build a ``(x, y, w, h)`` pixel box of ``box_size`` (width, height)
    centered on ``(col, row)``.

    Shared by box-drawing QA visualization and COCO export so both agree
    on exactly the same box for a given target. ``box_size`` is either a
    catalog's real measured extent already converted to this window's
    pixels (``extra_attributes["bbox_size_px"]``, set by ``sidescan-export
    chip`` when the target carries a ``bbox``) or, for center-only
    catalogs (JSF/Discover picks), a caller-supplied placeholder.
    """
    w, h = box_size
    return (col - w / 2, row - h / 2, w, h)


def project_target_to_frequency(target: Target, source_mps: float, target_mps: float, *, n_samples: int | None = None) -> Target | None:
    """Re-express ``target``'s position on a *different* frequency channel
    of the same survey, using the one thing both channels share exactly:
    the physical world.

    A source catalog (EdgeTech Discover, Klein SonarWiz) only ever marks
    a contact against the one frequency channel it was picked on — there
    is no "also visible on the other channel" field. But ``ping`` (both
    channels are pinged in lockstep — the same physical vessel position,
    confirmed on real data: identical ping *count* per channel within one
    file) and the physical slant range a sample represents
    (``sample * meters_per_sample``) are both real, channel-independent
    quantities. So the target's true position on the untagged channel is
    recoverable: convert its sample index to a physical slant range using
    the *source* channel's own meters/sample, then back to a sample index
    using the *target* channel's — same ping, same side, no catalog entry
    needed for that channel at all.

    ``ping``/``channel``/every other field are carried through unchanged
    (``replace``) — only ``sample`` moves, plus
    ``extra_attributes["projected_from_mps"]`` recording the source
    resolution, so a caller/renderer can tell a real catalog mark apart
    from one recovered this way (e.g. to draw it with a different box
    style) without losing the fact that it's the same real contact.

    Returns ``None`` (not a clamped/guessed value) if the projected
    sample would land at or past ``n_samples`` — a real, physical
    consequence of the two channels recording different maximum ranges,
    not a placement bug: the contact genuinely sits beyond the other
    channel's recorded window and has no honest position to report
    there. Only checked when ``n_samples`` is given.
    """
    if target.sample is None:
        return None
    slant_range_m = target.sample * source_mps
    projected_sample = round(slant_range_m / target_mps)
    if n_samples is not None and not (0 <= projected_sample < n_samples):
        return None
    return replace(
        target,
        sample=projected_sample,
        extra_attributes={**target.extra_attributes, "projected_from_mps": source_mps},
    )


def detrend_by_range(window: np.ndarray) -> np.ndarray:
    """Subtract each column's (sample index's) median value across every
    ping in the window, flattening any left-to-right brightness gradient
    for display.

    Found necessary 2026-07-30: even after real TVG correction
    (``apply_tvg_to_amplitude``), rendered target windows still showed a
    strong range-dependent brightness gradient that EdgeTech Discover's
    own images of the same targets don't show — consistent with
    ``sidescanpy``'s already-documented, unresolved Known Issue #4 (a
    real ~6-8dB range-dependent TVG residual, found on the main survey
    pipeline; ``apply_tvg_to_amplitude`` uses the identical formula, so
    it inherits the same residual). This does **not** fix or claim to
    explain that underlying residual — it's a display-only, per-window
    correction (a flat-field-style correction: subtract the *typical*
    level at each range so only deviations from it — real targets, real
    local texture — remain visible), analogous to how a photograph's
    vignette might be corrected for viewing without claiming to fix the
    lens that caused it.

    Parameters
    ----------
    window : np.ndarray
        ``(n_pings, n_samples)`` array, sample axis last.

    Returns
    -------
    np.ndarray
        Same shape, each column's median subtracted (result is centered
        near 0, not the original units — pair with a percentile-based
        vmin/vmax when rendering, not a fixed absolute scale).
    """
    column_median = np.median(window, axis=0, keepdims=True)
    return window - column_median


def orient_for_display(window: np.ndarray, target: Target) -> tuple[np.ndarray, Target]:
    """Orient a decoded target window to match EdgeTech Discover's own
    display convention, both found empirically by comparing rendered
    output for target 0001 directly against ``Target-0001.jpg``
    (2026-07-30) — neither is derived from a documented Discover
    convention, both from visual comparison:

    - **Horizontal (channel-dependent):** port channels (0) are mirrored
      along the sample axis. Matches the sign convention the rest of
      ``sidescanpy`` already uses for real waterfalls
      (``viz/waterfall.py``'s ``_compute_x_values``: starboard increases
      left-to-right unmirrored, port is negated) — ``decode_target_window()``'s
      raw array has no such orientation of its own (column 0 is always
      the first decoded sample regardless of channel), so anything
      rendering it directly was showing every port-channel target
      mirrored relative to both this project's own convention and
      Discover's.
    - **Vertical (unconditional, every target):** the ping axis is
      reversed. ``decode_target_window()`` orders rows by ascending ping
      sequence number (earliest ping first) and plain ``imshow`` puts row
      0 at the top, so the unflipped output showed earliest-ping-at-top;
      comparing the relative vertical position of target 0001's two
      distinct returns (the one with the larger trailing shadow sits in
      the *upper* half of Discover's image, the *lower* half of ours
      unflipped) confirms Discover shows the opposite ping order.

    Parameters
    ----------
    window : np.ndarray
        ``(n_pings, n_samples)`` decoded window, sample axis last.
    target : Target
        ``.channel`` (0=port, 1=starboard) controls the horizontal flip;
        anything else (including ``None``) skips only that part. The
        vertical flip always applies. Must have ``extra_attributes["window_row"]``
        for that remapping to happen — if absent, the array is still
        flipped but no row remapping is attempted (nothing to remap).

    Returns
    -------
    (window, target)
        Oriented window, and a copy of ``target`` with ``.sample``
        and/or ``extra_attributes["window_row"]`` remapped to match
        whichever flips were applied, so box placement and any
        downstream cropping stay correct.
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
