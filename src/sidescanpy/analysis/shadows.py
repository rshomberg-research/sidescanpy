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
dB of speckle. A second, weaker threshold catches the short, shallow
shadows of objects close to nadir, where the grazing angle is steep; those
regions are kept only with a highlight in front of them. Proposals are
meant to pre-fill boxes for a person to correct, so the defaults favour
recall over precision.

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
    "highlight_fraction", "height_m", "weak", "score",
]


@dataclass
class ShadowProposals(ResultMixin):
    """Proposals found by :func:`detect_shadow_proposals`.

    ``proposals`` has one row per proposal (columns
    :data:`PROPOSAL_COLUMNS`). Extents are half-open native positions
    along the dataset's ``ping`` and ``sample`` axes, like
    :class:`~sidescanpy.export.windows.native.NativeWindow`; the box spans
    the highlight (when found) and the shadow. ``height_m`` is NaN where
    altitude is unknown. ``weak`` marks proposals from the weak-shadow pass.
    ``score`` is in ``[0, 1]``: higher for a deeper shadow, a taller object
    (a longer shadow where altitude is unknown) and a highlight in front.
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
    weak_shadow_db: float | None = 4.0,
    weak_min_shadow_m: float = 0.1,
    weak_min_highlight: float = 0.5,
    score_height_m: float = 0.3,
    m_per_ping: float | None = None,
    mask_water_column: bool = True,
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
    7. Weak pass (unless ``weak_shadow_db`` is None): regions at least
       ``weak_shadow_db`` below the background that contain no cell of a
       proposal from steps 3-6, with a run of at least ``weak_min_shadow_m``,
       go through steps 4-6 and are kept when a highlight is found on at
       least ``weak_min_highlight`` of their pings. They get ``weak=True``.

    ``score`` = 0.4 x shadow depth (``shadow_db`` -> 0, ``3 * shadow_db`` -> 1)
    + 0.3 x height / ``score_height_m`` (capped at 1; shadow length / 1 m
    where the height is unknown) + 0.3 x the fraction of pings with a
    highlight. Height rather than shadow length, because the shadow of the
    same object is much shorter near nadir.

    The water column is excluded: with ``mask_water_column`` (default),
    samples at a slant range up to the ping's altitude (``sample *
    m_per_sample <= altitude_m``, the same flat-seabed geometry as the
    height estimate) are treated as NaN, because they come before the first
    bottom return and their low level reads as many weak false shadows.
    Pings with NaN or non-positive ``altitude_m`` (or a dataset without it)
    are not masked; fill missing altitude first with the ``bottom_track``
    correction step. ``ds`` is not modified.

    NaN cells (blanked or masked water column, hull echo, masked pings) are
    never shadow, and a region whose near-range side lies within 0.3 m of NaN
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
    weak_shadow_db : float or None, default 4.0
        Threshold of the weak pass (dB below background); None skips it.
    weak_min_shadow_m : float, default 0.1
    weak_min_highlight : float, default 0.5
        Fraction of a weak region's pings that must have a highlight.
    score_height_m : float, default 0.3
        Height that earns the full height part of the score.
    m_per_ping : float, optional
        Along-track spacing; default from the per-ping positions.
    mask_water_column : bool, default True
        Treat samples with slant range <= ``altitude_m`` as NaN (see above).
        False uses every sample as given.

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
              "min_height_m": min_height_m, "max_height_m": max_height_m, "max_length_m": max_length_m,
              "weak_shadow_db": weak_shadow_db, "weak_min_shadow_m": weak_min_shadow_m,
              "weak_min_highlight": weak_min_highlight, "score_height_m": score_height_m,
              "mask_water_column": mask_water_column}

    altitude = ds["altitude_m"].values.astype(float) if "altitude_m" in ds else np.full(ds.sizes["ping"], np.nan)
    sample0 = int(ds["sample"].values[0]) if "sample" in ds.coords else 0
    sample_pos = ds["sample"].values if "sample" in ds.coords else np.arange(ds.sizes["sample"])
    water = None
    if mask_water_column:
        known = np.isfinite(altitude) & (altitude > 0)
        # Slant range of each sample, as in the height estimate: sample position x m_per_sample.
        water = known[:, None] & (sample_pos[None, :] * m_s <= np.where(known, altitude, 0.0)[:, None])
        logger.debug("detect_shadow_proposals: water column masked on %d of %d ping(s) (%d samples)",
                     int(known.sum()), altitude.size, int(water.sum()))
    rows, rejected, n_regions = [], {"small": 0, "edge": 0, "length": 0, "height": 0, "weak_no_highlight": 0}, 0
    for ch in channels:
        raw = ds[value_var].isel(channel=ch).values.astype(np.float32)  # a copy
        if water is not None:
            raw[water] = np.nan
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
        geo = _Geometry(k=k, cm=cm, m_s=m_s, m_per_ping=m_per_ping, sample0=sample0, altitude=altitude,
                        gap_cells=gap_cells, search_cells=search_cells, edge_cells=edge_cells)
        labels, n = ndimage.label(shadow, structure=np.ones((3, 3), bool))
        n_regions += n
        kept = np.zeros(n + 1, bool)
        for idx, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            row = _region_proposal(labels[sl] == idx, sl, resid, bright, finite, geo, min_pings=min_pings,
                                   min_run=min_shadow_cells, rejected=rejected)
            if row is not None and _accept(row, rejected, min_height_m, max_height_m, max_length_m):
                rows.append(_finish(row, ch, False, shadow_db, score_height_m))
                kept[idx] = True
        if weak_shadow_db is None:
            continue
        weak = np.nan_to_num(resid, nan=0.0) <= -weak_shadow_db
        wlabels, wn = ndimage.label(weak, structure=np.ones((3, 3), bool))
        n_regions += wn
        weak_min_cells = max(1, int(np.ceil(weak_min_shadow_m / cm)))
        for idx, sl in enumerate(ndimage.find_objects(wlabels), 1):
            if sl is None:
                continue
            region = wlabels[sl] == idx
            if kept[labels[sl][region]].any():       # already a proposal from the main pass
                continue
            row = _region_proposal(region, sl, resid, bright, finite, geo, min_pings=min_pings,
                                   min_run=weak_min_cells, rejected=rejected)
            if row is None:
                continue
            if row["highlight_fraction"] < weak_min_highlight:
                rejected["weak_no_highlight"] += 1
                continue
            if _accept(row, rejected, min_height_m, max_height_m, max_length_m):
                rows.append(_finish(row, ch, True, shadow_db, score_height_m))
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


@dataclass
class _Geometry:
    k: int
    cm: float
    m_s: float
    m_per_ping: float
    sample0: int
    altitude: np.ndarray
    gap_cells: int
    search_cells: int
    edge_cells: int


def _region_proposal(region, sl, resid, bright, finite, g, *, min_pings, min_run, rejected):
    """Box, highlight and features of one shadow region, or None (counted in ``rejected``)."""
    p0, c0 = sl[0].start, sl[1].start
    per_ping = region.any(axis=1)
    if per_ping.sum() < min_pings:
        rejected["small"] += 1
        return None
    run = _longest_run_per_row(region)
    if run.max() < min_run:
        rejected["small"] += 1
        return None
    rows_r = np.flatnonzero(per_ping)
    first_cols = np.array([c0 + np.flatnonzero(region[r])[0] for r in rows_r])
    last_cols = np.array([c0 + np.flatnonzero(region[r])[-1] for r in rows_r])
    # A dark band along the edge of blanked data (water column, hull echo) is
    # the background median failing there, not a shadow.
    near_nan = [not finite[p0 + r, max(0, f - g.edge_cells):f].all() for r, f in zip(rows_r, first_cols)]
    if np.mean(near_nan) > 0.5:
        rejected["edge"] += 1
        return None
    hl_start, hl_contrast, heights = [], [], []
    for r, s_first, s_last in zip(rows_r, first_cols, last_cols):
        p = p0 + r
        lo = max(0, s_first - g.gap_cells - g.search_cells)
        hits = np.flatnonzero(bright[p, lo:s_first])
        if hits.size and lo + hits[-1] >= s_first - g.gap_cells - 1:
            end = lo + int(hits[-1])
            start = end
            while start - 1 >= lo and bright[p, start - 1]:
                start -= 1
            hl_start.append(start)
            hl_contrast.append(float(np.nanmean(resid[p, start:end + 1])))
        ls = (s_last - s_first + 1) * g.cm
        rng = (g.sample0 + s_first * g.k) * g.m_s
        a = g.altitude[p]
        if np.isfinite(a) and rng + ls > 0:
            heights.append(a * ls / (rng + ls))
    shadow_start = int(first_cols.min())
    box_start = int(min(hl_start)) if hl_start else shadow_start
    box_stop = int(last_cols.max()) + 1
    ping_start, ping_stop = p0 + int(rows_r[0]), p0 + int(rows_r[-1]) + 1
    return {
        "ping_start": ping_start, "ping_stop": ping_stop,
        "sample_start": box_start * g.k, "sample_stop": box_stop * g.k,
        "shadow_sample_start": shadow_start * g.k, "n_pings": int(per_ping.sum()),
        "shadow_length_m": float(run.max() * g.cm),
        "highlight_length_m": float(np.median([shadow_start - s for s in hl_start]) * g.cm) if hl_start else 0.0,
        "length_along_m": (ping_stop - ping_start) * g.m_per_ping,
        "length_across_m": (box_stop - box_start) * g.cm,
        "shadow_contrast_db": float(-np.nanmean(resid[sl][region])),
        "highlight_contrast_db": float(np.mean(hl_contrast)) if hl_contrast else float("nan"),
        "highlight_fraction": len(hl_start) / int(per_ping.sum()),
        "height_m": float(np.median(heights)) if heights else float("nan"),
    }


def _accept(row, rejected, min_height_m, max_height_m, max_length_m):
    if row["length_along_m"] > max_length_m or row["length_across_m"] > max_length_m:
        rejected["length"] += 1
        return False
    h = row["height_m"]
    if np.isfinite(h) and not (min_height_m <= h <= max_height_m):
        rejected["height"] += 1
        return False
    return True


def _finish(row, channel, weak, shadow_db, score_height_m):
    h = row["height_m"]
    size = h / score_height_m if np.isfinite(h) else row["shadow_length_m"] / 1.0
    score = (np.clip((row["shadow_contrast_db"] - shadow_db) / (2 * shadow_db), 0, 1) * 0.4
             + np.clip(size, 0, 1) * 0.3 + row["highlight_fraction"] * 0.3)
    out = {"channel": int(channel), **row, "weak": bool(weak), "score": round(float(score), 3)}
    for key in ("shadow_length_m", "highlight_length_m", "length_along_m", "length_across_m", "highlight_fraction",
                "height_m"):
        out[key] = round(out[key], 3) if np.isfinite(out[key]) else out[key]
    for key in ("shadow_contrast_db", "highlight_contrast_db"):
        out[key] = round(out[key], 2) if np.isfinite(out[key]) else out[key]
    return out
