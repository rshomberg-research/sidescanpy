"""
Training-time augmentation for exported windows: random flips and jitter.

Every function takes an explicit ``numpy.random.Generator`` so a seeded
export is reproducible, and every flip moves the annotated target
positions (``Target.sample`` = column, ``extra_attributes["window_row"]``
= row) together with the pixels.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pyproj

from sidescanpy.core.target import Target
from sidescanpy.grids.definition import utm_crs_for_lonlat

logger = logging.getLogger(__name__)


def random_flip(
    window: np.ndarray, targets: list[Target], rng: np.random.Generator, *, edge_coords: bool,
) -> tuple[np.ndarray, list[Target], bool, bool]:
    """Flip ``window`` horizontally and vertically, each with probability 0.5.

    Applied after display orientation; every target is remapped to match.
    Two draws are always taken from ``rng`` (horizontal, then vertical), so
    the generator advances identically whether or not anything flips.

    Parameters
    ----------
    window : np.ndarray
        ``(n_rows, n_cols)`` image.
    targets : list of Target
        Window-local positions.
    rng : np.random.Generator
    edge_coords : bool
        ``True`` for continuous pixel-edge coordinates (geometric/georef
        windows: pixel ``i`` spans ``[i, i+1)``), mirrored as ``n - x``;
        ``False`` for integer pixel-center indices (sample/ping windows),
        mirrored as ``n - 1 - x``.

    Returns
    -------
    (window, targets, flipped_h, flipped_v)
        ``window`` is C-contiguous.
    """
    flip_h, flip_v = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    n_rows, n_cols = window.shape
    off = 0 if edge_coords else 1
    if flip_h:
        window = window[:, ::-1]
        targets = [replace(ct, sample=n_cols - off - ct.sample) for ct in targets]
    if flip_v:
        window = window[::-1, :]
        targets = [
            replace(ct, extra_attributes={**ct.extra_attributes, "window_row": n_rows - off - ct.extra_attributes["window_row"]})
            for ct in targets
        ]
    return np.ascontiguousarray(window), targets, flip_h, flip_v


def jitter_offset_m(jitter_px: int, resolution_m: float, rng: np.random.Generator) -> float:
    """Uniform integer pixel jitter in ``[-jitter_px, jitter_px]``, in meters.

    Same convention as :func:`~sidescanpy.export.windows.extract_chip`'s
    pixel jitter, scaled by the window's ``resolution_m``. ``jitter_px == 0``
    returns ``0.0`` without drawing from ``rng``.
    """
    if not jitter_px:
        return 0.0
    return float(rng.integers(-jitter_px, jitter_px + 1)) * resolution_m


def jitter_latlon(lat: float, lon: float, north_jitter_m: float, east_jitter_m: float) -> tuple[float, float]:
    """Offset a WGS84 point by exact north/east meters via its UTM zone.

    The round trip through the auto-selected UTM zone avoids a small-angle
    degrees approximation. A zero offset returns the input unchanged.

    Returns
    -------
    (lat, lon)
    """
    if north_jitter_m == 0.0 and east_jitter_m == 0.0:
        return lat, lon
    crs = utm_crs_for_lonlat(lon, lat)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_latlon = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    easting, northing = to_utm.transform(lon, lat)
    new_lon, new_lat = to_latlon.transform(easting + east_jitter_m, northing + north_jitter_m)
    return new_lat, new_lon


def augmented_file_name(stem: str, aug_idx: int, n_augment: int) -> str:
    """``<stem>.png`` for a single draw, ``<stem>_a<aug_idx>.png`` when ``n_augment > 1``."""
    return f"{stem}.png" if n_augment <= 1 else f"{stem}_a{aug_idx}.png"
