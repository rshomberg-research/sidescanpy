"""
Survey track + scan-lines with target catalog positions overlaid.

Builds on ``viz/track.py``'s ``plot_track_and_scanlines``/
``build_survey_track``, adding a scatter of target positions (a list of
``sidescanpy.targets.core.Target``) projected into the same CRS as the
track. Kept in its own module rather than folded into ``viz/track.py``
because it needs ``targets.core.Target`` — matches the existing
``viz/contact_sheet.py`` precedent of keeping that dependency isolated
from the modules that don't need it.


2026 rshomberg@uri.edu
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer

from sidescanpy.targets.core import Target
from sidescanpy.viz.track import plot_track_and_scanlines

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)


def _project_lonlat(lat, lon, crs: CRS) -> Tuple[np.ndarray, np.ndarray]:
    """WGS84 (lat, lon) -> (x, y) in crs. Mirrors viz/track.py's own
    (private) projection so target markers land in the same coordinate
    system as the track/scanlines drawn by plot_track_and_scanlines."""
    if crs.is_geographic:
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    transformer = Transformer.from_crs(_WGS84, crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def plot_target_position_scatter(targets: list[Target], *, title: Optional[str] = None) -> plt.Figure:
    """Plain lat/lon scatter of every target with a resolved position —
    no track/scanlines context, just a standalone QA plot. Returns
    ``None`` if no target has both ``latitude``/``longitude`` set.

    Promoted from the near-identical scatter block duplicated in both
    format's own target-catalog-reading step (JSF's Discover
    catalog already carries position; Klein's ``.tgt`` needs it resolved
    externally first — see ``sidescanpy.io.sdf.targets``).
    """
    lat = [t.latitude for t in targets if t.latitude is not None and t.longitude is not None]
    lon = [t.longitude for t in targets if t.latitude is not None and t.longitude is not None]
    if not lat:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lon, lat, s=50, c="#c1440e", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Target positions (n={len(lat)})")
    ax.ticklabel_format(useOffset=False)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def export_target_position_scatter_png(targets: list[Target], output_path: str, *, title: Optional[str] = None, dpi: int = 150) -> bool:
    """Plot and save :func:`plot_target_position_scatter`'s output.
    Returns False (no file written) if no target had a resolvable
    position, True otherwise."""
    fig = plot_target_position_scatter(targets, title=title)
    if fig is None:
        return False
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.debug(f"Exported target position scatter to {output_path}")
    return True


def plot_track_with_targets(
    track: dict,
    targets: list[Target],
    *,
    crs: Optional[CRS] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    show_land: bool = False,
    show_locator: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a survey track + scan-lines (see ``plot_track_and_scanlines``)
    with target catalog positions scattered on top, each annotated with
    its ``target_id``.

    Parameters
    ----------
    track : dict
        As returned by ``sidescanpy.viz.track.build_survey_track``.
    targets : list[Target]
        Targets with ``.latitude``/``.longitude`` set; targets missing
        either are silently excluded from the scatter (matches the
        source script's behavior — a target with only native ping/
        sample position and no georeferenced lat/lon can't be plotted
        here).
    crs : pyproj.CRS, optional
        Defaults to an auto-selected UTM zone from the track's centroid.
    fig, ax : optional
        Existing figure/axes to draw into.
    show_land, show_locator : bool, default False
        See ``plot_track_and_scanlines``. Note: ``show_land`` fixes the
        view to the track/scan-line extent (plus a margin) — a target
        cataloged well outside that extent would be plotted but fall
        outside the visible view. Not expected in practice (a cataloged
        target is normally inside the swath that found it), not
        specially handled here.

    Returns
    -------
    fig, ax
    """
    if crs is None:
        from sidescanpy.grids.definition import utm_crs_for_lonlat
        crs = utm_crs_for_lonlat(
            float(np.mean(track["track_lon"])), float(np.mean(track["track_lat"]))
        )

    fig, ax = plot_track_and_scanlines(
        track["track_lat"], track["track_lon"],
        track["scan_port_lat"], track["scan_port_lon"],
        track["scan_stbd_lat"], track["scan_stbd_lon"],
        crs=crs, fig=fig, ax=ax,
        show_land=show_land, show_locator=show_locator,
    )

    t_lat = [t.latitude for t in targets if t.latitude is not None and t.longitude is not None]
    t_lon = [t.longitude for t in targets if t.latitude is not None and t.longitude is not None]
    t_id = [t.target_id for t in targets if t.latitude is not None and t.longitude is not None]
    tx, ty = _project_lonlat(t_lat, t_lon, crs)

    ax.scatter(
        tx, ty, s=70, marker="o", facecolor="#e0501c", edgecolor="black",
        linewidth=0.8, zorder=5, label=f"Targets (n={len(tx)})",
    )
    for x, y, tid in zip(tx, ty, t_id):
        ax.annotate(tid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#e0501c")
    ax.legend(loc="best")
    ax.set_title(ax.get_title() + "\nwith target catalog overlay")

    return fig, ax


def export_track_with_targets_png(
    track: dict,
    targets: list[Target],
    output_path: str,
    *,
    crs: Optional[CRS] = None,
    dpi: int = 150,
    show_land: bool = False,
    show_locator: bool = False,
) -> None:
    """Plot and save a survey track + scan-lines + target overlay PNG.
    See ``plot_track_with_targets``.

    Saves at the figure's own full, fixed size, no ``bbox_inches="tight"``
    -- see ``export_track_and_scanlines_png``'s docstring (``viz/track.py``)
    for why: that flag crops the saved PNG to the drawn content's own
    bounding box, making saved dimensions vary with a survey's real
    aspect ratio despite the figure always being created at one fixed
    size. This is the function `sidescan-export track` actually calls
    whenever `--targets-json` is given -- i.e. every survey with a
    target catalog.
    """
    fig, _ax = plot_track_with_targets(track, targets, crs=crs, show_land=show_land, show_locator=show_locator)
    fig.savefig(output_path, dpi=dpi)
    logger.debug(f"Exported track + targets to {output_path}")
    plt.close(fig)
