"""
Waterfall plotting for SideScan sonar.

Plots a port+starboard combined waterfall image with configurable x-axis
(sample, ground_range, across_track) and y-axis (ping, time, along_track)
modes. Returns the matplotlib Axes for further customization.

2026 rshomberg@uri.edu
"""

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

logger = logging.getLogger(__name__)


XAxisMode = Literal["sample", "ground_range", "across_track"]
YAxisMode = Literal["ping", "time", "along_track"]


_DEFAULT_CMAP = "copper"
_DEFAULT_VMIN_PCT = 1.0
_DEFAULT_VMAX_PCT = 99.0


def plot_waterfall(
    ds: xr.Dataset,
    *,
    start: int = 0,
    stop: int | None = None,
    x_axis: XAxisMode = "sample",
    y_axis: YAxisMode = "ping",
    cmap: str | Colormap = _DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = _DEFAULT_VMIN_PCT,
    vmax_pct: float = _DEFAULT_VMAX_PCT,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """
    Plot a port+starboard combined waterfall.

    Parameters
    ----------
    ds : xr.Dataset
        Validated sidescan dataset.
    start : int, default 0
        First ping index to include.
    stop : int, optional
        One past last ping index. Defaults to all pings.
    x_axis : {"sample", "ground_range", "across_track"}
        How to interpret the x-axis.

        - ``"sample"``: sample number, port mirrored (-N to +N).
          Always works.
        - ``"ground_range"``: across-bottom distance, signed by channel
          (port = negative). Requires ``ground_range_m`` from
          ``compute_geometric_ranges``.
        - ``"across_track"``: signed across-track in platform frame
          (already negative for port). Requires ``across_track_m``.

    y_axis : {"ping", "time", "along_track"}
        How to interpret the y-axis.

        - ``"ping"``: ping number.
        - ``"time"``: ping timestamp.
        - ``"along_track"``: cumulative platform distance from first ping.
          Computed via ``pyproj.Geod.inv`` from per-ping latitude/
          longitude.

    cmap : str or Colormap, default "copper"
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color scale bounds. If not given, computed from finite-value
        percentiles (``vmin_pct`` and ``vmax_pct``).
    vmin_pct, vmax_pct : float, default 1.0 and 99.0
        Percentiles for auto color scaling. Ignored if ``vmin`` or
        ``vmax`` is given.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not given, creates a new figure with a
        single Axes.
    title : str, optional
        Plot title. Default: ping range and processing level.

    Returns
    -------
    Axes
        The matplotlib Axes containing the plot.

    Raises
    ------
    ValueError
        If a required variable is missing for the chosen axis mode.
    """
    # Slice the dataset by ping range
        # Compute full-survey y values, then slice
    y_full = _compute_y_full(ds, y_axis)
    y_sliced = y_full[start:stop]

    # Slice dataset for backscatter and x-axis
    sliced = ds.isel(ping=slice(start, stop))
    n_pings_slice = sliced.sizes["ping"]
    n_samples = sliced.sizes["sample"]

    bs = sliced["backscatter_intensity"].values

    # Compute x for the slice
    x_port, x_stbd = _compute_x_values(sliced, x_axis)

    # Broadcast y to (n_pings_slice, n_samples)
    y_2d = np.broadcast_to(y_sliced[:, None], (n_pings_slice, n_samples)).copy()
    y_port = y_2d
    y_stbd = y_2d


    # ----- color scale -----
    if vmin is None or vmax is None:
        finite = bs[np.isfinite(bs)]
        if finite.size == 0:
            raise ValueError("No finite values in backscatter; cannot auto-scale color")
        if vmin is None:
            vmin = float(np.percentile(finite, vmin_pct))
        if vmax is None:
            vmax = float(np.percentile(finite, vmax_pct))
        logger.debug(f"Auto color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # # ----- y-axis values -----
    # y_port, y_stbd = _compute_y_values(sliced, y_axis, start)

    # # ----- x-axis values -----
    # x_port, x_stbd = _compute_x_values(sliced, x_axis)

    # ----- plot -----
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    

    # Plot port (will be on the left side via negative x or signed values)
    x_port, y_port_clean, c_port = _mask_invalid_coord(x_port, y_port, bs[:, 0, :])
    x_stbd, y_stbd_clean, c_stbd = _mask_invalid_coord(x_stbd, y_stbd, bs[:, 1, :])

    ax.pcolormesh(
        x_port, y_port_clean, c_port,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
        rasterized=True,
    )
    ax.pcolormesh(
        x_stbd, y_stbd_clean, c_stbd,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
        rasterized=True,
    )
    # ----- axis labels and limits -----
    ax.set_xlabel(_x_axis_label(x_axis))
    ax.set_ylabel(_y_axis_label(y_axis))

    # Vertical line at center (port|starboard divide for sample/ground_range)
    if x_axis in ("sample", "ground_range", "across_track"):
        ax.axvline(x=0, color="white", linestyle="-",
                   linewidth=0.5, alpha=0.5)

    # Y inverted so first ping is at top (waterfall convention)
    ax.invert_yaxis()

    # Title
    if title is None:
        proc_level = ds.attrs.get("processing_level", "raw")
        title = f"Waterfall  pings {start}–{stop}  ({proc_level})"
    ax.set_title(title)

    return ax


# ============================================================
# X-AXIS COMPUTATION
# ============================================================

def _compute_x_values(
    ds: xr.Dataset,
    x_axis: XAxisMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-channel x-axis arrays of shape (n_pings, n_samples).

    Returns (x_port, x_stbd). For "sample" mode, x_port is negative-mirrored.
    """
    n_pings = ds.sizes["ping"]
    n_samples = ds.sizes["sample"]

    if x_axis == "sample":
        # Port: -N to 0, Starboard: 0 to +N. Both broadcast to (pings, samples).
        sample_idx = np.arange(n_samples, dtype=np.float64)
        x_stbd = np.broadcast_to(sample_idx[None, :], (n_pings, n_samples)).copy()
        x_port = -x_stbd
        return x_port, x_stbd

    if x_axis == "ground_range":
        if "ground_range_m" not in ds.data_vars:
            raise ValueError(
                "x_axis='ground_range' requires 'ground_range_m'. "
                "Run compute_geometric_ranges first."
            )
        gr = ds["ground_range_m"].values  # (pings, channel, sample)
        # Port: negative; Starboard: positive
        x_port = -gr[:, 0, :]
        x_stbd = +gr[:, 1, :]
        return x_port, x_stbd

    if x_axis == "across_track":
        if "across_track_m" not in ds.data_vars:
            raise ValueError(
                "x_axis='across_track' requires 'across_track_m'. "
                "Run compute_geometric_ranges first."
            )
        at = ds["across_track_m"].values  # (pings, channel, sample), already signed
        x_port = at[:, 0, :]   # port is already negative
        x_stbd = at[:, 1, :]   # starboard is already positive
        return x_port, x_stbd

    raise ValueError(f"Unknown x_axis: {x_axis}")


# ============================================================
# Y-AXIS COMPUTATION
# ============================================================

def _compute_y_values(
    ds: xr.Dataset,
    y_axis: YAxisMode,
    start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-channel y-axis arrays of shape (n_pings, n_samples).

    Returns (y_port, y_stbd). Both channels share the same y values
    per ping (since y is a per-ping property).
    """
    n_pings = ds.sizes["ping"]
    n_samples = ds.sizes["sample"]

    if y_axis == "ping":
        # Use absolute ping index from the original dataset
        y = np.arange(start, start + n_pings, dtype=np.float64)
    elif y_axis == "time":
        # ping timestamps as datetime64 — matplotlib handles these natively
        y = ds["time"].values.astype("datetime64[ns]")
    elif y_axis == "along_track":
        y = _along_track_distance(ds)
    else:
        raise ValueError(f"Unknown y_axis: {y_axis}")

    # Broadcast to (pings, samples) for pcolormesh
    y_2d = np.broadcast_to(y[:, None], (n_pings, n_samples)).copy()
    return y_2d, y_2d

def _compute_y_full(ds: xr.Dataset, y_axis: YAxisMode) -> np.ndarray:
    """Compute 1D y values for all pings in ds. Length = ds.sizes['ping']."""
    if y_axis == "ping":
        return np.arange(ds.sizes["ping"], dtype=np.float64)
    if y_axis == "time":
        return ds["time"].values.astype("datetime64[ns]")
    if y_axis == "along_track":
        return _along_track_distance(ds)
    raise ValueError(f"Unknown y_axis: {y_axis}")

def _along_track_distance(ds: xr.Dataset) -> np.ndarray:
    """Compute cumulative along-track distance from per-ping lat/lon."""
    import pyproj

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    geod = pyproj.Geod(ellps="WGS84")

    if len(lats) < 2:
        return np.array([0.0])

    # Forward azimuths and distances between consecutive pings
    _, _, segment_distances = geod.inv(
        lons[:-1], lats[:-1], lons[1:], lats[1:]
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_distances)])
    return cumulative.astype(np.float64)

# ==========================================================
# Masking
# ==========================================================

def _mask_invalid_coord(x, y, c):
    """Build a masked array for c where x has non-finite values.
    y is always finite in supported modes.
    """
    nan_mask = ~np.isfinite(x)
    if not nan_mask.any():
        return x, y, c
    # Replace NaN in x with a finite placeholder
    x_safe = np.where(np.isnan(x), 0.0, x)
    c_masked = np.ma.masked_array(c, mask=nan_mask)
    return x_safe, y, c_masked

# ============================================================
# AXIS LABELS
# ============================================================

def _x_axis_label(x_axis: XAxisMode) -> str:
    if x_axis == "sample":
        return "Sample (port ← | → starboard)"
    if x_axis == "ground_range":
        return "Ground range (m)  (port ← | → starboard)"
    if x_axis == "across_track":
        return "Across-track distance (m)  (port ← | → starboard)"
    return ""


def _y_axis_label(y_axis: YAxisMode) -> str:
    if y_axis == "ping":
        return "Ping number"
    if y_axis == "time":
        return "Time"
    if y_axis == "along_track":
        return "Along-track distance (m)"
    return ""
