"""
Display helpers for decoded target windows (range detrending, orientation).

2026 rshomberg@uri.edu
"""

import logging
from dataclasses import replace

import numpy as np

from sidescanpy.core.target import Target

logger = logging.getLogger(__name__)


def detrend_by_range(window: np.ndarray) -> np.ndarray:
    """Subtract each column's (sample index's) median value across every
    ping in the window, flattening any left-to-right brightness gradient
    for display.

    Even after real TVG correction
    (``apply_tvg_to_amplitude``), rendered target windows still showed a
    strong range-dependent brightness gradient that EdgeTech Discover's
    own images of the same targets don't show — consistent with a
    real ~6-8dB range-dependent TVG residual (``apply_tvg_to_amplitude``
    uses the same formula as the main pipeline, so it inherits it). This does **not** fix or claim to
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
    output directly against Discover's own target images — neither is derived from a documented Discover
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
