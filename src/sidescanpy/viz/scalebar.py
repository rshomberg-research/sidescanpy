"""
A simple cartographic scale bar (line + end ticks + a rounded length
label), for any plot whose axes are already in real projected meters --
this library's own default for :func:`sidescanpy.viz.track.plot_track_and_scanlines`
(auto-UTM) and :func:`sidescanpy.viz.gridded.plot_gridded_raster`
(auto-UTM-selected grid).

Deliberately not handling geographic (WGS84 degrees) axes automatically
-- meters-per-degree varies with latitude for the x-axis specifically,
so a caller plotting directly in lat/lon would need to supply the real
conversion factor at that plot's own latitude, not something this
function can infer from the axes alone.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

_LOCATIONS = ("lower right", "lower left", "upper right", "upper left")


def _nice_scale_length(target_m: float) -> float:
    """Round ``target_m`` to the nearest "nice" 1/2/5 x 10^n value --
    the standard cartographic scale-bar convention (never an odd number
    like "37 m"). Returns 1.0 for a non-positive/degenerate target."""
    if not np.isfinite(target_m) or target_m <= 0:
        return 1.0
    exponent = np.floor(np.log10(target_m))
    base = target_m / (10.0 ** exponent)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0 ** exponent)


def _format_scale_label(length_m: float) -> str:
    if length_m >= 1000.0:
        km = length_m / 1000.0
        return f"{km:g} km"
    return f"{length_m:g} m"


def add_scale_bar(
    ax,
    *,
    meters_per_data_unit: float = 1.0,
    location: str = "lower right",
    color: str = "black",
    linewidth: float = 3.0,
    fontsize: int = 9,
    frac_of_width: float = 0.2,
    margin: float = 0.06,
) -> None:
    """Draw a scale bar into ``ax``'s current view.

    Call this AFTER ``ax``'s final data/limits are set (e.g. after
    plotting and any ``set_aspect``/``set_xlim`` calls) -- it reads
    ``ax.get_xlim()`` to size itself to the current view, so calling it
    earlier would size it against limits that haven't settled yet.

    Parameters
    ----------
    meters_per_data_unit : float, default 1.0
        Real-world meters per one unit of ``ax``'s own x-axis data
        coordinates. ``1.0`` is correct for any plot already in
        projected meters (UTM northing/easting, this library's own
        default for the track/gridded plots above) -- pass a real
        computed value for anything else, never guess one.
    location : {"lower right", "lower left", "upper right", "upper left"}
        Corner to draw in, axes-fraction positioned (robust to whatever
        the current data range actually is).
    frac_of_width : float, default 0.2
        Target length as a fraction of the axes' current visible width,
        before rounding to a "nice" value -- the actual drawn length is
        usually close to but not exactly this fraction.
    margin : float, default 0.06
        Axes-fraction margin from the plot edge.

    Silently does nothing (not an error) if the axes' current view is
    degenerate (zero or non-finite width) -- e.g. a single-point plot,
    where no scale bar length would be meaningful anyway.
    """
    if location not in _LOCATIONS:
        raise ValueError(f"Unknown location {location!r}, choose from {_LOCATIONS}")

    xlim = ax.get_xlim()
    span_data = abs(xlim[1] - xlim[0])
    span_m = span_data * meters_per_data_unit
    if not np.isfinite(span_m) or span_m <= 0:
        logger.debug("add_scale_bar: degenerate axes span, skipping")
        return

    bar_length_m = _nice_scale_length(span_m * frac_of_width)
    bar_length_data_frac = (bar_length_m / meters_per_data_unit) / span_data

    x_anchor = margin if "left" in location else 1.0 - margin - bar_length_data_frac
    y_anchor = margin if location.startswith("lower") else 1.0 - margin
    x0, x1 = x_anchor, x_anchor + bar_length_data_frac

    tick_h = 0.015
    label_va = "bottom" if location.startswith("lower") else "top"
    label_y = y_anchor + tick_h if location.startswith("lower") else y_anchor - tick_h

    ax.add_line(Line2D([x0, x1], [y_anchor, y_anchor], transform=ax.transAxes, color=color, linewidth=linewidth, solid_capstyle="butt", zorder=10))
    for x in (x0, x1):
        ax.add_line(Line2D([x, x], [y_anchor - tick_h / 2, y_anchor + tick_h / 2], transform=ax.transAxes, color=color, linewidth=linewidth, zorder=10))
    ax.text(
        (x0 + x1) / 2, label_y, _format_scale_label(bar_length_m),
        transform=ax.transAxes, ha="center", va=label_va, fontsize=fontsize, color=color, zorder=10,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
    )
