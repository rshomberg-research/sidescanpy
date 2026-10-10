"""
Object proposals from acoustic shadows in native slant-range data.

An object standing proud of the seabed returns a highlight on its
near-range face and leaves a shadow (little or no return) behind it,
further out along the same pings. :func:`detect_shadow_proposals` finds
those shadows as regions well below the local seabed level, pairs each
with the highlight just in front of it, and returns one proposal box per
object in native ``(ping, sample, channel)`` coordinates, with features
(sizes, contrast, an estimated height) to rank or filter them.

The data are averaged into cells about ``cell_m`` across track (one ping
along track) before thresholding, because single samples carry several
dB of speckle. Proposals are meant to pre-fill boxes for a person to
correct, so the defaults favour recall over precision.

2026 rshomberg@uri.edu
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from sidescanpy.analysis._result import ResultMixin
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.processing.sample_axis import (
    invalid_sound_vel_warning_suppressed,
    sample_interval_m,
)

logger = logging.getLogger(__name__)

PROPOSAL_COLUMNS = [
    "channel", "ping_start", "ping_stop", "sample_start", "sample_stop",
    "shadow_sample_start", "n_pings", "shadow_length_m", "highlight_length_m",
    "length_along_m", "length_across_m", "shadow_contrast_db", "highlight_contrast_db",
    "highlight_fraction", "height_m", "score",
]


@dataclass
class ShadowProposals(ResultMixin):
    """Proposals found by :func:`detect_shadow_proposals`.

    ``proposals`` has one row per proposal (columns
    :data:`PROPOSAL_COLUMNS`). Extents are half-open native positions
    along the dataset's ``ping`` and ``sample`` axes, like
    :class:`~sidescanpy.export.windows.native.NativeWindow`; the box spans
    the highlight (when found) and the shadow. ``height_m`` is NaN where
    altitude is unknown. ``score`` is in ``[0, 1]``: higher for a deeper,
    longer shadow with a highlight in front of it.
    """

    proposals: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PROPOSAL_COLUMNS))
    m_per_ping: float | None = None
    m_per_sample: float | None = None
    cell_samples: int | None = None
    params: dict = field(default_factory=dict)
    n_shadow_regions: int = 0
    n_rejected: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _m_per_ping(ds: xr.Dataset) -> float:
    along = along_track_distance_m(ds)
    ok = np.isfinite(along)
    if ok.sum() < 2:
        raise ValueError("detect_shadow_proposals: needs m_per_ping or at least two pings with a position")
    idx = np.flatnonzero(ok)
    total = float(np.nanmax(along[ok]) - np.nanmin(along[ok]))
    if total <= 0:
        raise ValueError("detect_shadow_proposals: the track does not move; pass m_per_ping")
    return total / (idx[-1] - idx[0])


def _cells(values: np.ndarray, k: int) -> np.ndarray:
    """Mean over consecutive blocks of ``k`` samples (NaN-aware), shape (ping, n_cells)."""
    n = values.shape[1] // k
    blocks = values[:, : n * k].reshape(values.shape[0], n, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        return np.nanmean(blocks, axis=2)


def _background(cells: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """NaN-aware running median, ``size`` = (pings, cells)."""
    finite = np.isfinite(cells)
    if not finite.any():
        return np.full_like(cells, np.nan)
    filled = np.where(finite, cells, np.nanmedian(cells))
    bg = ndimage.median_filter(filled, size=size, mode="nearest")
    # Cells near blanked areas: the fill value biases the median, so use the
    # median of the valid neighbours where at least half of the window is valid.
    frac = ndimage.uniform_filter(finite.astype(np.float32), size=size, mode="nearest")
    bg[frac < 0.5] = np.nan
    return bg


def detect_shadow_proposals(
    ds: xr.Dataset,
    *,
    value_var: str = "backscatter_intensity",
    channels: tuple[int, ...] = (0, 1),
    cell_m: float = 0.06,
    background_m: tuple[float, float] = (3.0, 6.0),
    smooth_cells: tuple[int, int] = (1, 3),
    shadow_db: float = 6.0,
    highlight_db: float = 3.0,
    min_shadow_m: float = 0.2,
    min_pings: int = 2,
    max_gap_m: float = 0.6,
    highlight_search_m: float = 2.0,
    min_height_m: float = 0.0,
    max_height_m: float = 3.0,
    max_length_m: float = 8.0,
    m_per_ping: float | None = None,
) -> ShadowProposals:
    """Find objects by their acoustic shadow.

    For each channel:

    1. Average ``value_var`` (dB, after the ``tvg`` step) into cells of
       about ``cell_m`` along range, one ping each, then smooth them with a
       ``smooth_cells`` (pings, cells) running mean.
    2. Background: a running median over ``background_m`` (across, along
       track, metres).
    3. Shadow cells are at least ``shadow_db`` below the background; they
       are grouped into connected regions (8-connected).
    4. A region is kept if it covers at least ``min_pings`` pings and its
       longest run along range is at least ``min_shadow_m``.
    5. Its highlight is the run of cells at least ``highlight_db`` above the
       background ending within ``max_gap_m`` before the shadow, searched
       up to ``highlight_search_m`` towards nadir, on each of its pings.
    6. Height from flat-seabed geometry, ``h = A * Ls / (R + Ls)`` (altitude
       ``A``, slant range ``R`` to the shadow start, shadow slant length
       ``Ls``), per ping, median over the region. Proposals outside
       ``[min_height_m, max_height_m]`` (when the height is known) or longer
       than ``max_length_m`` in either direction are dropped.

    NaN cells (blanked water column, hull echo, masked pings) are never
    shadow, and a region whose near-range side lies within 0.3 m of NaN
    cells on most of its pings is dropped: that is the edge of the blanked
    area, not an object.

    Parameters
    ----------
    ds : xr.Dataset
        One frequency channel (slice pings beforehand to limit memory).
    value_var : str, default "backscatter_intensity"
        Values in dB.
    channels : tuple of int, default (0, 1)
    cell_m : float, default 0.06
        Across-track cell size; rounded to a whole number of samples.
    background_m : (float, float), default (3.0, 6.0)
        Background window, across track and along track (m).
    smooth_cells : (int, int), default (1, 3)
        Running-mean size (pings, cells) applied before thresholding.
    shadow_db, highlight_db : float, default 6.0, 3.0
        Thresholds below / above the background (dB).
    min_shadow_m : float, default 0.2
    min_pings : int, default 2
    max_gap_m : float, default 0.6
    highlight_search_m : float, default 2.0
    min_height_m, max_height_m : float, default 0.0, 3.0
    max_length_m : float, default 8.0
    m_per_ping : float, optional
        Along-track spacing; default from the per-ping positions.

    Returns
    -------
    ShadowProposals
    """
    with invalid_sound_vel_warning_suppressed():
        m_s = float(np.nanmedian(sample_interval_m(ds)))
    if m_per_ping is None:
        m_per_ping = _m_per_ping(ds)
    k = max(1, int(round(cell_m / m_s)))
    cm = k * m_s
    bg_size = (max(3, int(round(background_m[1] / m_per_ping)) | 1), max(3, int(round(background_m[0] / cm)) | 1))
    min_shadow_cells = max(1, int(np.ceil(min_shadow_m / cm)))
    gap_cells = max(0, int(round(max_gap_m / cm)))
    search_cells = max(1, int(round(highlight_search_m / cm)))
    edge_cells = max(1, int(round(0.3 / cm)))
    params = {"cell_m": cm, "background_m": list(background_m), "smooth_cells": list(smooth_cells),
              "shadow_db": shadow_db, "highlight_db": highlight_db, "min_shadow_m": min_shadow_m,
              "min_pings": min_pings, "max_gap_m": max_gap_m, "highlight_search_m": highlight_search_m,
              "min_height_m": min_height_m, "max_height_m": max_height_m, "max_length_m": max_length_m}

    altitude = ds["altitude_m"].values.astype(float) if "altitude_m" in ds else np.full(ds.sizes["ping"], np.nan)
    sample0 = int(ds["sample"].values[0]) if "sample" in ds.coords else 0
    rows, rejected, n_regions = [], {"small": 0, "edge": 0, "length": 0, "height": 0}, 0
    for ch in channels:
        raw = ds[value_var].isel(channel=ch).values.astype(np.float32)
        cells = _cells(raw, k)
        finite = np.isfinite(cells)
        if smooth_cells != (1, 1):
            w = ndimage.uniform_filter(finite.astype(np.float32), size=smooth_cells, mode="nearest")
            s = ndimage.uniform_filter(np.where(finite, cells, 0.0).astype(np.float32), size=smooth_cells, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                cells = np.where(finite, s / w, np.nan)
        bg = _background(cells, bg_size)
        resid = cells - bg
        shadow = np.nan_to_num(resid, nan=0.0) <= -shadow_db
        bright = np.nan_to_num(resid, nan=0.0) >= highlight_db
        labels, n = ndimage.label(shadow, structure=np.ones((3, 3), bool))
        n_regions += n
        for idx, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            region = labels[sl] == idx
            p0, c0 = sl[0].start, sl[1].start
            per_ping = region.any(axis=1)
            if per_ping.sum() < min_pings:
                rejected["small"] += 1
                continue
            run = _longest_run_per_row(region)
            if run.max() < min_shadow_cells:
                rejected["small"] += 1
                continue
            # A dark band along the edge of blanked data (water column, hull
            # echo) is the background median failing there, not a shadow.
            first = [c0 + np.flatnonzero(region[r])[0] for r in np.flatnonzero(per_ping)]
            rows_p = p0 + np.flatnonzero(per_ping)
            near_nan = [not finite[p, max(0, f - edge_cells):f].all() for p, f in zip(rows_p, first)]
            if np.mean(near_nan) > 0.5:
                rejected["edge"] += 1
                continue
            hl_start, hl_hits, hl_contrast = [], 0, []
            heights = []
            for r in np.flatnonzero(per_ping):
                cols = np.flatnonzero(region[r])
                s_first = c0 + cols[0]
                p = p0 + r
                lo = max(0, s_first - gap_cells - search_cells)
                hits = np.flatnonzero(bright[p, lo:s_first])
                if hits.size and lo + hits[-1] >= s_first - gap_cells - 1:
                    end = lo + int(hits[-1])
                    start = end
                    while start - 1 >= lo and bright[p, start - 1]:
                        start -= 1
                    hl_start.append(start)
                    hl_hits += 1
                    hl_contrast.append(float(np.nanmean(resid[p, start:end + 1])))
                ls = (cols[-1] - cols[0] + 1) * cm
                rng = (sample0 + s_first * k) * m_s
                a = altitude[p]
                if np.isfinite(a) and rng + ls > 0:
                    heights.append(a * ls / (rng + ls))
            n_p = int(per_ping.sum())
            first_cols = np.array([c0 + np.flatnonzero(region[r])[0] for r in np.flatnonzero(per_ping)])
            last_cols = np.array([c0 + np.flatnonzero(region[r])[-1] for r in np.flatnonzero(per_ping)])
            shadow_start = int(first_cols.min())
            box_start = int(min(hl_start)) if hl_start else shadow_start
            box_stop = int(last_cols.max()) + 1
            ping_start, ping_stop = p0 + int(np.flatnonzero(per_ping)[0]), p0 + int(np.flatnonzero(per_ping)[-1]) + 1
            along = (ping_stop - ping_start) * m_per_ping
            across = (box_stop - box_start) * cm
            if along > max_length_m or across > max_length_m:
                rejected["length"] += 1
                continue
            height = float(np.median(heights)) if heights else float("nan")
            if np.isfinite(height) and not (min_height_m <= height <= max_height_m):
                rejected["height"] += 1
                continue
            contrast = float(-np.nanmean(resid[sl][region]))
            hl_frac = hl_hits / n_p
            shadow_len = float(run.max() * cm)
            score = (np.clip((contrast - shadow_db) / (2 * shadow_db), 0, 1) * 0.4
                     + np.clip(shadow_len / 1.0, 0, 1) * 0.3 + hl_frac * 0.3)
            rows.append({
                "channel": int(ch), "ping_start": ping_start, "ping_stop": ping_stop,
                "sample_start": box_start * k, "sample_stop": box_stop * k,
                "shadow_sample_start": shadow_start * k, "n_pings": n_p,
                "shadow_length_m": round(shadow_len, 3),
                "highlight_length_m": round(float(np.median([shadow_start - s for s in hl_start]) * cm), 3) if hl_start else 0.0,
                "length_along_m": round(along, 3), "length_across_m": round(across, 3),
                "shadow_contrast_db": round(contrast, 2),
                "highlight_contrast_db": round(float(np.mean(hl_contrast)), 2) if hl_contrast else float("nan"),
                "highlight_fraction": round(hl_frac, 3), "height_m": round(height, 3) if np.isfinite(height) else height,
                "score": round(float(score), 3),
            })
    df = pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)
    logger.debug(f"detect_shadow_proposals: {len(df)} proposal(s) from {n_regions} shadow region(s), rejected {rejected}")
    return ShadowProposals(proposals=df, m_per_ping=float(m_per_ping), m_per_sample=m_s, cell_samples=k,
                           params=params, n_shadow_regions=n_regions, n_rejected=rejected)


def _longest_run_per_row(mask: np.ndarray) -> np.ndarray:
    """Length of the longest run of ``True`` in each row."""
    out = np.zeros(mask.shape[0], dtype=int)
    for i, row in enumerate(mask):
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        out[i] = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
    return out
