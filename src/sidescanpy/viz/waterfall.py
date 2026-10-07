"""
Waterfall plotting for SideScan sonar.

Plots a port+starboard combined waterfall image with configurable x-axis
(sample, ground_range, across_track, slant_range) and y-axis (ping, time, along_track)
modes. Returns the matplotlib Axes for further customization.

2026 rshomberg@uri.edu
"""

import logging
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, PowerNorm
from matplotlib.ticker import FuncFormatter

from sidescanpy.core.units import mask_floor_clip
from sidescanpy.processing.navigation import along_track_distance_m as _along_track_distance_m
from sidescanpy.processing.sample_axis import sample_index
from sidescanpy.viz._cmap import resolve_cmap

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    # Deprecated shim, kept for one release.
    if name == "along_track_distance_m":
        warnings.warn(
            "sidescanpy.viz.waterfall.along_track_distance_m has moved to "
            "sidescanpy.processing.navigation",
            DeprecationWarning,
            stacklevel=2,
        )
        return _along_track_distance_m
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


XAxisMode = Literal["sample", "ground_range", "across_track", "slant_range"]
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
    ping_offset: int = 0,
    cmap: str | Colormap = _DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = _DEFAULT_VMIN_PCT,
    vmax_pct: float = _DEFAULT_VMAX_PCT,
    gamma: float = 1.0,
    show_colorbar: bool = False,
    ax: Axes | None = None,
    title: str | None = None,
    value_var: str = "backscatter_intensity",
) -> Axes:
    """
    Plot a port+starboard combined waterfall.

    Parameters
    ----------
    ds : xr.Dataset
        Validated sidescan dataset.
    value_var : str, default "backscatter_intensity"
        Which backscatter variable to plot -- the TVG-corrected default,
        or ``"backscatter_raw"`` for an uncorrected view. Always present
        regardless of what correction stages have run (raw is immutable).
    start : int, default 0
        First ping index to include.
    stop : int, optional
        One past last ping index. Defaults to all pings.
    x_axis : {"sample", "ground_range", "across_track", "slant_range"}
        How to interpret the x-axis.

        - ``"sample"``: sample number, port mirrored (-N to +N).
          Always works.
        - ``"ground_range"``: across-bottom distance, signed by channel
          (port = negative). Requires ``ground_range_m`` from
          ``compute_geometric_ranges``.
        - ``"across_track"``: signed across-track in platform frame
          (already negative for port). Requires ``across_track_m``.
        - ``"slant_range"``: attitude-invariant time-of-flight range,
          signed by channel (port = negative) the same way as
          ``"ground_range"``. Unlike ground range, always finite (never
          NaN in the water column) — useful when the water column
          itself needs to stay visible/maskable by a caller rather than
          being NaN'd out by the range variable's own geometry.
          Requires ``slant_range_m``.

    y_axis : {"ping", "time", "along_track"}
        How to interpret the y-axis.

        - ``"ping"``: ping number.
        - ``"time"``: ping timestamp.
        - ``"along_track"``: cumulative platform distance from first ping.
          Computed via ``pyproj.Geod.inv`` from per-ping latitude/
          longitude.

    ping_offset : int, default 0
        Added to ``y_axis="ping"`` values (and to the title's ping
        range). Use when ``ds`` is itself an already-extracted window
        of a larger survey (e.g. one segment of a multi-file
        concatenated channel) so the y-axis reports the segment's true
        position in the full survey instead of 0-based local indices.
        Ignored for ``y_axis="time"``/``"along_track"`` — those are
        derived from ``ds``'s own per-ping fields and need no external
        offset.
    cmap : str or Colormap, default "copper"
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color scale bounds. If not given, computed from finite-value
        percentiles (``vmin_pct`` and ``vmax_pct``).
    vmin_pct, vmax_pct : float, default 1.0 and 99.0
        Percentiles for auto color scaling. Ignored if ``vmin`` or
        ``vmax`` is given.
    gamma : float, default 1.0
        Display-only power-law contrast curve, applied via
        ``matplotlib.colors.PowerNorm(gamma, vmin, vmax)`` instead of a
        plain linear ``Normalize``. The plotted array itself is never
        modified — only the value-to-color mapping is warped — so the
        backscatter stays real dB and (if ``show_colorbar=True``) the
        colorbar keeps honest dB tick values, just non-uniformly
        spaced. ``gamma=1.0`` (default) is a linear scale, identical to
        this function's behavior before this parameter existed.
    show_colorbar : bool, default False
        If True, add a colorbar labeled "Backscatter (dB)".
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
    y_full = _compute_y_full(ds, y_axis, ping_offset=ping_offset)
    y_sliced = y_full[start:stop]

    # Slice dataset for backscatter and x-axis
    sliced = ds.isel(ping=slice(start, stop))
    n_pings_slice = sliced.sizes["ping"]
    n_samples = sliced.sizes["sample"]

    bs = mask_floor_clip(sliced[value_var].values)

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

    # gamma != 1.0: PowerNorm warps the color mapping only, so the plotted
    # array and colorbar stay real dB (never renormalized to [0, 1]).
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if gamma != 1.0 else None
    scale_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    resolved_cmap = resolve_cmap(cmap)

    ax.pcolormesh(
        x_port, y_port_clean, c_port,
        cmap=resolved_cmap, shading="nearest", rasterized=True, **scale_kwargs,
    )
    mesh_stbd = ax.pcolormesh(
        x_stbd, y_stbd_clean, c_stbd,
        cmap=resolved_cmap, shading="nearest", rasterized=True, **scale_kwargs,
    )

    if show_colorbar:
        fig.colorbar(mesh_stbd, ax=ax, label="Backscatter (dB)")
    # ----- axis labels and limits -----
    ax.set_xlabel(_x_axis_label(x_axis))
    ax.set_ylabel(_y_axis_label(y_axis))

    # Vertical line at center (port|starboard divide for sample/ground_range)
    if x_axis in ("sample", "ground_range", "across_track", "slant_range"):
        ax.axvline(x=0, color="white", linestyle="-",
                   linewidth=0.5, alpha=0.5)

    # Y inverted so first ping is at top (waterfall convention)
    ax.invert_yaxis()

    # Title
    if title is None:
        proc_level = ds.attrs.get("processing_level", "raw")
        title_start = start + ping_offset
        title_stop = (stop if stop is not None else ds.sizes["ping"]) + ping_offset
        title = f"Waterfall  pings {title_start}–{title_stop}  ({proc_level})"
    ax.set_title(title)

    return ax


def resolve_waterfall_axes(steps: frozenset[str], cli_x_axis: str, cli_y_axis: str) -> tuple[str, str]:
    """Pick the waterfall x/y axis mode for a given correction's step set.

    Corrections that include the "geometric" step always render in
    real-world distance (across_track x meters-in / along_track meters)
    rather than the caller's chosen x/y axis — otherwise the geometric
    correction has no visible effect (sample-index and raw ping-count
    axes look identical whether or not ground/across-track ranges were
    computed, since only the color values change from TVG, not the
    pixel geometry). The caller's axes remain in effect for corrections
    without "geometric" (raw, radiometric), where sample/ping/time are
    the only meaningful choices anyway.

        """
    if "geometric" in steps:
        return "across_track", "along_track"
    return cli_x_axis, cli_y_axis


def plot_survey_overview(
    ds: xr.Dataset,
    *,
    ping_stride: int = 20,
    break_ping_indices: list[int] | None = None,
    x_axis: XAxisMode = "sample",
    cmap: str | Colormap = _DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = _DEFAULT_VMIN_PCT,
    vmax_pct: float = _DEFAULT_VMAX_PCT,
    gamma: float = 1.0,
    show_colorbar: bool = False,
    ax: Axes | None = None,
    title: str | None = None,
    value_var: str = "backscatter_intensity",
) -> Axes:
    """
    Compressed, whole-survey waterfall overview.

    Strides the ping axis by ``ping_stride`` before rendering (via
    ``plot_waterfall``), so a very long survey renders as one
    reasonably-sized image instead of one row of pixels per ping.
    Optionally draws a horizontal marker line + ping-number label at each
    index in ``break_ping_indices`` — useful for e.g. file boundaries in
    a multi-file concatenated survey. Nothing in ``SideScanSurvey``/
    ``.ss.h5`` currently records per-ping source-file identity, so if you
    need break markers, compute the boundary ping indices yourself (e.g.
    while concatenating per-file datasets) and pass them in here.

    Parameters
    ----------
    ds : xr.Dataset
        Validated sidescan dataset (typically a whole survey/channel, not
        a pre-sliced segment).
    ping_stride : int, default 20
        Vertical compression factor — every ``ping_stride``-th ping is
        kept. Higher = more compressed. Must be >= 1.
    break_ping_indices : list[int], optional
        Ping indices (in ``ds``'s own, un-strided index space) at which
        to draw a break marker — e.g. where one source file ends and the
        next begins in a concatenated survey.
    x_axis, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, show_colorbar, ax, title
        Same meaning as in ``plot_waterfall``. ``title`` defaults to a
        summary of the total ping count and compression factor.

    Returns
    -------
    Axes
        The matplotlib Axes containing the plot. The y-axis is labeled
        with true ping numbers (not the strided/local index), even
        though the underlying mesh is built from the strided data.

    Raises
    ------
    ValueError
        If ``ping_stride`` is not a positive integer, or (via
        ``plot_waterfall``) if a required variable is missing for the
        chosen ``x_axis`` mode.
    """
    if ping_stride < 1:
        raise ValueError(f"ping_stride must be >= 1, got {ping_stride}")

    n_pings = ds.sizes["ping"]
    strided = ds.isel(ping=slice(0, None, ping_stride))

    if title is None:
        proc_level = ds.attrs.get("processing_level", "raw")
        title = f"Survey overview  {n_pings} pings, 1:{ping_stride} compressed  ({proc_level})"

    ax = plot_waterfall(
        strided,
        x_axis=x_axis,
        y_axis="ping",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmin_pct=vmin_pct,
        vmax_pct=vmax_pct,
        gamma=gamma,
        show_colorbar=show_colorbar,
        ax=ax,
        title=title,
        value_var=value_var,
    )

    # plot_waterfall labeled the y-axis with the *strided* dataset's own
    # local 0..N-1 ping index — it has no notion of the stride applied
    # above. Relabel tick text to true ping numbers without touching the
    # already-plotted mesh's coordinates (which must stay in local/strided
    # index space to match break-marker placement below).
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _pos: f"{val * ping_stride:.0f}")
    )

    if break_ping_indices:
        xlim = ax.get_xlim()
        for break_idx in break_ping_indices:
            local_y = break_idx / ping_stride
            ax.axhline(y=local_y, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.text(
                xlim[1], local_y, f" ping {break_idx}",
                color="red", fontsize=7, va="center", ha="left", clip_on=False,
            )

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
        sample_idx = sample_index(ds)
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

    if x_axis == "slant_range":
        if "slant_range_m" not in ds.data_vars:
            raise ValueError(
                "x_axis='slant_range' requires 'slant_range_m'. "
                "Run compute_geometric_ranges first."
            )
        sr = ds["slant_range_m"].values  # (pings, channel, sample), always unsigned/finite
        x_port = -sr[:, 0, :]
        x_stbd = +sr[:, 1, :]
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
        y = _along_track_distance_m(ds)
    else:
        raise ValueError(f"Unknown y_axis: {y_axis}")

    # Broadcast to (pings, samples) for pcolormesh
    y_2d = np.broadcast_to(y[:, None], (n_pings, n_samples)).copy()
    return y_2d, y_2d

def _compute_y_full(ds: xr.Dataset, y_axis: YAxisMode, ping_offset: int = 0) -> np.ndarray:
    """Compute 1D y values for all pings in ds. Length = ds.sizes['ping']."""
    if y_axis == "ping":
        return np.arange(ds.sizes["ping"], dtype=np.float64) + ping_offset
    if y_axis == "time":
        return ds["time"].values.astype("datetime64[ns]")
    if y_axis == "along_track":
        return _along_track_distance_m(ds)
    raise ValueError(f"Unknown y_axis: {y_axis}")

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
    if x_axis == "slant_range":
        return "Slant range (m)  (port ← | → starboard)"
    return ""


def _y_axis_label(y_axis: YAxisMode) -> str:
    if y_axis == "ping":
        return "Ping number"
    if y_axis == "time":
        return "Time"
    if y_axis == "along_track":
        return "Along-track distance (m)"
    return ""
