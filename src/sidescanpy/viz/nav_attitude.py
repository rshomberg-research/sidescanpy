"""
Plotting for `sidescanpy.analysis.nav_attitude` -- one figure covering
heading, pitch/roll, vertical geometry (altitude/depth/heave), and the
ping-colored position track for a ping range. Consumes
`nav_attitude.nav_attitude_summary`'s `StageResult` for the heading
classification/out-of-range marking, same "pure presentation over
pre-computed analysis output" split as `viz/investigation.py`.

2026 rshomberg@uri.edu
"""

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from sidescanpy.analysis.investigation import StageResult


def plot_nav_attitude(
    ds: xr.Dataset,
    *,
    ping_offset: int = 0,
    result: StageResult | None = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes, Axes]]:
    """Heading (with the [0,360)/[-180,180) convention bands and any
    out-of-both-range points marked), pitch+roll, altitude+depth+heave,
    and a lon/lat position track colored by ping number -- in that
    order, sharing one ping-number x-axis across the first three panels.

    `ping_offset` matches `viz/waterfall.py: plot_waterfall`'s own
    parameter -- pass the ping range's true global start so the x-axis
    reads real survey ping numbers, not a 0-based index local to
    whatever slice was loaded.
    """
    n_pings = ds.sizes["ping"]
    ping_axis = ping_offset + np.arange(n_pings)

    fig, axd = plt.subplot_mosaic(
        [["heading", "track"], ["attitude", "track"], ["vertical", "track"]],
        figsize=(13, 9), width_ratios=[2.0, 1.2],
    )
    ax_heading, ax_attitude, ax_vertical, ax_track = axd["heading"], axd["attitude"], axd["vertical"], axd["track"]
    ax_attitude.sharex(ax_heading)
    ax_vertical.sharex(ax_heading)

    if "heading_deg" in ds:
        heading = np.asarray(ds["heading_deg"].values, dtype=np.float64)
        ax_heading.axhspan(0, 360, color="tab:green", alpha=0.05, zorder=0)
        ax_heading.axhspan(-180, 180, color="tab:blue", alpha=0.05, zorder=0)
        for y in (0, 360, -180, 180):
            ax_heading.axhline(y, color="0.75", lw=0.6, ls=":")
        ax_heading.plot(ping_axis, heading, color="0.2", lw=0.8)
        if result is not None and result.metrics.get("heading_frac_out_of_range", 0.0) > 0.0:
            out_mask = ~(((heading >= 0.0) & (heading < 360.0)) | ((heading >= -180.0) & (heading < 180.0)))
            ax_heading.scatter(
                ping_axis[out_mask], heading[out_mask], color="tab:red", s=10, zorder=5,
                label="outside both conventions",
            )
            ax_heading.legend(loc="upper right", fontsize=8)
        title = "heading_deg"
        if result is not None and result.classification:
            title += f"  (classification: {result.classification})"
        ax_heading.set_title(title)
        ax_heading.set_ylabel("degrees")

    for var, color in (("pitch_deg", "tab:orange"), ("roll_deg", "tab:purple")):
        if var in ds:
            ax_attitude.plot(ping_axis, ds[var].values, color=color, lw=0.8, label=var)
    ax_attitude.axhline(0, color="0.8", lw=0.6)
    ax_attitude.set_ylabel("degrees")
    ax_attitude.set_title("pitch / roll")
    if ax_attitude.get_legend_handles_labels()[0]:
        ax_attitude.legend(loc="upper right", fontsize=8)

    for var, color in (("altitude_m", "tab:green"), ("depth_m", "tab:brown"), ("heave_m", "tab:gray")):
        if var in ds:
            ax_vertical.plot(ping_axis, ds[var].values, color=color, lw=0.8, label=var)
    ax_vertical.set_ylabel("meters")
    ax_vertical.set_xlabel("ping number")
    ax_vertical.set_title("altitude / depth / heave")
    if ax_vertical.get_legend_handles_labels()[0]:
        ax_vertical.legend(loc="upper right", fontsize=8)

    if "latitude" in ds and "longitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        ax_track.plot(lon, lat, color="0.6", lw=0.4, zorder=0)
        sc = ax_track.scatter(lon, lat, c=ping_axis, cmap="viridis", s=6, zorder=1)
        ax_track.set_aspect("equal", adjustable="datalim")
        ax_track.set_xlabel("longitude")
        ax_track.set_ylabel("latitude")
        ax_track.set_title("track (color = ping number)")
        fig.colorbar(sc, ax=ax_track, label="ping number", fraction=0.046, pad=0.04)
    else:
        ax_track.set_axis_off()
        ax_track.text(0.5, 0.5, "no latitude/longitude", ha="center", va="center", transform=ax_track.transAxes)

    fig.suptitle(f"pings {ping_axis[0]}:{ping_axis[-1] + 1}" if n_pings else "pings (empty)")
    fig.tight_layout()
    return fig, (ax_heading, ax_attitude, ax_vertical, ax_track)
