"""
Does a seabed object look the same from different passes? Slant-range and
ground-range image chips of the same objects, seen from different
directions and ranges, side by side and scored.

The question: an object detector trained on sidescan images sees each
object from many directions and ranges. In slant range the across-track
axis is compressed near nadir and stretched far out, so the same object
changes shape with range; in ground range (``|across_track_m|`` from the
geometric step) it keeps its size. If that matters, chips of the same
object should agree better with each other in ground range.

Given a list of object positions (any source: a target catalog,
detections, observations), the script:

1. merges positions within ``--cluster-m`` into one object (several passes
   may have reported it);
2. finds every pass that ensonified each object: the georeferenced sample
   (``latitude_per_sample`` / ``longitude_per_sample``) nearest to it,
   within ``--max-dist-m``, one look per pass (a new pass after a gap of
   ``--pass-gap-s``);
3. renders, for every look, four chips of ``--chip-m`` metres around the
   object: slant and ground range, each in the track frame (rows along
   track, columns away from the sonar) and rotated north-up using the
   heading and side (the rotation fixed, so every look shares one map
   frame);
4. scores every pair of looks of the same object with masked normalised
   cross-correlation (best within ``--max-shift-m``, to absorb navigation
   error), in the track frame, north-up, and north-up with the best
   rotation (a rotation-invariant upper bound); and the same for pairs of
   looks of *different* objects as the baseline;
5. summarises: same-object similarity, its separation from the
   different-object baseline (AUC), and how it falls with the difference
   in range and in look direction between the two looks.

Needs files that went through the ``geometric`` and ``georef`` steps.
Uses only the sidescanpy API of 0.7.x, numpy, scipy, pandas and
matplotlib. Memory: files are read in ping blocks (``--block-pings``) and
chips from short ping windows, about 1 GB on 900-ping files; the chips of
every look are kept until the end (~1 MB per look at the default size), so
``--max-objects`` bounds the rest.

Writes, under --out:
  objects.csv, looks.csv, pairs.csv      every object, look and scored pair
  summary.json, summary.md               the numbers per geometry x frame
  similarity_summary.png                 distributions, AUC, similarity vs
                                         range / direction difference
  gallery.png                            a few objects, every look, 4 chips each
  objects/object_<id>.png                one sheet per object with >= 2 looks

Usage:
    # objects from sidescanpy's shadow detector, nothing else needed (recommended)
    python look_consistency.py SS_H5_DIR --proposals --frequency hz_900000 --out OUT [--min-length-m 2.5]
    # objects from your own detections (lat, lon, file, ping, channel, sample, time[, conf, length_along_m])
    python look_consistency.py SS_H5_DIR --detections dets.csv --frequency hz_900000 --out OUT
    # objects at known positions (looks found by position: sensitive to navigation error between passes)
    python look_consistency.py SS_H5_DIR --points points.csv --frequency hz_900000 --out OUT
        [--chip-m 12] [--cluster-m 3] [--max-dist-m 1] [--min-looks 2] [--max-objects 400]

points.csv needs columns ``lat`` and ``lon``; ``id`` and ``source`` are
kept when present.

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from sidescanpy.export.windows.levels import nominal_sample_interval_m
from sidescanpy.export.windows.native import along_track_spacing_m
from sidescanpy.io.hdf5 import load_channel_dataset
from sidescanpy.log import configure_logging

logger = logging.getLogger("look_consistency")
M_PER_DEG = 111_320.0
GEOMETRIES = ("slant", "ground")
FRAMES = ("track", "north_up", "best_rotation")


# ---------------------------------------------------------------------------
# positions


class LocalFrame:
    """Flat east/north metres around a reference point (fine over a survey)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.k = math.cos(math.radians(lat0))

    def xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * M_PER_DEG * self.k, (np.asarray(lat) - self.lat0) * M_PER_DEG


def load_objects(path: Path, cluster_m: float) -> tuple[pd.DataFrame, LocalFrame]:
    pts = pd.read_csv(path)
    pts = pts[np.isfinite(pts.lat) & np.isfinite(pts.lon)].reset_index(drop=True)
    if "id" not in pts:
        pts["id"] = np.arange(1, len(pts) + 1)
    if "source" not in pts:
        pts["source"] = path.stem
    frame = LocalFrame(float(pts.lat.mean()), float(pts.lon.mean()))
    x, y = frame.xy(pts.lat, pts.lon)
    # connected components of points closer than cluster_m
    parent = np.arange(len(pts))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(np.c_[x, y]).query_pairs(cluster_m):
        parent[root(i)] = root(j)
    pts["cluster"] = [root(i) for i in range(len(pts))]
    pts["x"], pts["y"] = x, y
    objs = pts.groupby("cluster").agg(x=("x", "mean"), y=("y", "mean"), lat=("lat", "mean"), lon=("lon", "mean"),
                                      n_points=("id", "size"), point_ids=("id", lambda s: ";".join(map(str, s))),
                                      sources=("source", lambda s: ";".join(sorted(set(map(str, s))))))
    objs = objs.reset_index(drop=True)
    objs.insert(0, "object_id", np.arange(1, len(objs) + 1))
    return objs, frame


def proposal_detections(files, freq_key, out_csv: Path, block_pings=1000, min_score=0.5,
                        min_length_m=0.0, **proposal_kw) -> Path:
    """Run sidescanpy's shadow-proposal detector over every file (in ping
    blocks, so memory stays small) and write a detections CSV for
    ``--detections``: lat, lon, file, ping, channel, sample, time, conf,
    length_along_m, length_across_m. ``min_length_m`` keeps only objects at
    least that long along or across track (e.g. 2.5 to skip trap-sized ones).
    Reused when ``out_csv`` exists."""
    from sidescanpy.analysis import detect_shadow_proposals

    if out_csv.exists():
        return out_csv
    rows = []
    for k, path in enumerate(files):
        ds = load_channel_dataset(path, freq_key, {"ping": block_pings})
        if "latitude_per_sample" not in ds:
            continue
        n = ds.sizes["ping"]
        try:
            mpp = along_track_spacing_m(ds)
        except ValueError:
            continue
        keep = [v for v in ("backscatter_intensity", "latitude_per_sample", "longitude_per_sample", "altitude_m",
                            "latitude", "longitude", "heading_deg", "sound_vel_ms") if v in ds]
        for p0 in range(0, n, block_pings):
            sub = ds[keep].isel(ping=slice(p0, min(n, p0 + block_pings))).load()  # only what is used: memory
            props = detect_shadow_proposals(sub, m_per_ping=mpp, **proposal_kw).proposals
            props = props[(props.score >= min_score)
                          & ((props.length_along_m >= min_length_m) | (props.length_across_m >= min_length_m))]
            lat, lon, t = sub["latitude_per_sample"].values, sub["longitude_per_sample"].values, sub["time"].values
            for r in props.itertuples():
                pi = int((r.ping_start + r.ping_stop) / 2)
                si = int((r.sample_start + r.sample_stop) / 2)
                ch = int(r.channel)
                if not np.isfinite(lat[pi, ch, si]):
                    continue
                rows.append({"lat": lat[pi, ch, si], "lon": lon[pi, ch, si], "file": path.name, "ping": p0 + pi,
                             "channel": ch, "sample": si, "time": t[pi], "conf": float(r.score),
                             "length_along_m": float(r.length_along_m), "length_across_m": float(r.length_across_m)})
        logger.info("[%d/%d] %s: %d proposals so far", k + 1, len(files), path.name, len(rows))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def isolated_objects(path: Path, link_m: float, iso_m: float, pass_gap_s: float, min_conf: float,
                     size_tol: float | None = None):
    """Objects from detections that are unambiguous across passes.

    ``path``: detections from every pass (columns lat, lon, file, ping,
    channel, sample, time; conf optional). A detection is isolated when no
    other detection of its pass lies within ``iso_m``. Isolated detections of
    different passes within ``link_m`` are linked into one object, kept when
    each pass contributes exactly one detection and no other detection of
    those passes lies within ``iso_m`` of any member: then the detection is
    the object in every pass, and each look is centred on its own detection,
    so navigation error between passes (often several metres) does not
    matter. With ``size_tol`` and a ``length_along_m`` column, an object is
    also dropped when its looks disagree on along-track length by more than
    that fraction (along-track length does not depend on the projection).
    Returns (objects, looks, frame).
    """
    det = pd.read_csv(path)
    det = det[np.isfinite(det.lat) & np.isfinite(det.lon)]
    if "conf" in det:
        det = det[det.conf >= min_conf]
    det = det.assign(time=pd.to_datetime(det.time)).sort_values("time").reset_index(drop=True)
    det["pass_no"] = (det.time.diff().dt.total_seconds().fillna(np.inf) > pass_gap_s).cumsum()
    frame = LocalFrame(float(det.lat.mean()), float(det.lon.mean()))
    det["x"], det["y"] = frame.xy(det.lat, det.lon)
    xy = det[["x", "y"]].values
    tree = cKDTree(xy)
    passes = det.pass_no.values
    neigh = tree.query_ball_point(xy, iso_m)
    isolated = np.array([not any(passes[j] == passes[i] for j in nb if j != i) for i, nb in enumerate(neigh)])
    parent = np.arange(len(det))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(link_m):
        if isolated[i] and isolated[j] and passes[i] != passes[j]:
            parent[root(i)] = root(j)
    groups = {}
    for i in np.flatnonzero(isolated):
        groups.setdefault(root(i), []).append(i)
    obj_rows, look_rows = [], []
    for members in groups.values():
        mp = passes[members]
        if len(members) < 2 or len(set(mp)) != len(mp):
            continue
        member_set, member_passes = set(members), set(mp)
        if any(j not in member_set and passes[j] in member_passes for i in members for j in neigh[i]):
            continue  # another detection of one of these passes is close: ambiguous
        if size_tol is not None and "length_along_m" in det:
            sz = det.loc[members, "length_along_m"]
            if sz.min() <= 0 or sz.max() / sz.min() > 1 + size_tol:
                continue  # looks disagree on the object's size: probably not the same object
        oid = len(obj_rows) + 1
        g = det.loc[members]
        obj_rows.append({"object_id": oid, "x": g.x.mean(), "y": g.y.mean(), "lat": g.lat.mean(), "lon": g.lon.mean(),
                         "n_points": len(g), "point_ids": "", "sources": "detections",
                         "spread_m": float(np.hypot(g.x - g.x.mean(), g.y - g.y.mean()).max())})
        look_rows += [{"object_id": oid, "file": r.file, "ping": int(r.ping), "channel": int(r.channel),
                       "sample": int(r.sample), "dist_m": 0.0} for r in g.itertuples()]
    return pd.DataFrame(obj_rows), pd.DataFrame(look_rows), frame


def find_looks(files, freq_key, objs, frame, max_dist_m, block_pings=1000, sample_step=4) -> pd.DataFrame:
    """The georeferenced sample nearest to each object, per file and ping block, within max_dist_m."""
    oxy = np.c_[objs.x, objs.y]
    hits = []
    for k, path in enumerate(files):
        ds = load_channel_dataset(path, freq_key, {"ping": block_pings})
        if "latitude_per_sample" not in ds or "across_track_m" not in ds:
            logger.warning("%s: not georeferenced / geometric-corrected, skipped", path.name)
            continue
        n = ds.sizes["ping"]
        for p0 in range(0, n, block_pings):
            sl = {"ping": slice(p0, min(p0 + block_pings, n)), "sample": slice(None, None, sample_step)}
            lat = ds["latitude_per_sample"].isel(sl).values
            lon = ds["longitude_per_sample"].isel(sl).values
            ok = np.isfinite(lat) & np.isfinite(lon)
            if not ok.any():
                continue
            x, y = frame.xy(lat[ok], lon[ok])
            near = ((oxy[:, 0] > x.min() - max_dist_m) & (oxy[:, 0] < x.max() + max_dist_m)
                    & (oxy[:, 1] > y.min() - max_dist_m) & (oxy[:, 1] < y.max() + max_dist_m))
            if not near.any():
                continue
            d, idx = cKDTree(np.c_[x, y]).query(oxy[near], distance_upper_bound=max_dist_m)
            flat = np.flatnonzero(ok)
            for oi, dd, ii in zip(np.flatnonzero(near), d, idx):
                if not np.isfinite(dd):
                    continue
                pi, ch, si = np.unravel_index(flat[ii], lat.shape)
                hits.append({"object_id": int(objs.object_id.iloc[oi]), "file": path.name, "ping": int(p0 + pi),
                             "channel": int(ch), "sample": int(si * sample_step), "dist_m": float(dd)})
        logger.info("[%d/%d] %s: %d look(s) so far", k + 1, len(files), path.name, len(hits))
    return pd.DataFrame(hits)


def describe_looks(looks: pd.DataFrame, ss_dir: Path, freq_key: str, pass_gap_s: float) -> pd.DataFrame:
    """Time, heading, ranges and grazing angle of each look; one look per pass (closest)."""
    rows = []
    for fname, g in looks.groupby("file"):
        ds = load_channel_dataset(ss_dir / fname, freq_key, {"ping": 2048})
        m_s = nominal_sample_interval_m(ds)
        t = ds["time"].values
        head = ds["heading_deg"].values
        alt = ds["altitude_m"].values if "altitude_m" in ds else np.full(ds.sizes["ping"], np.nan)
        for r in g.to_dict("records"):
            p, ch, s = r["ping"], r["channel"], r["sample"]
            ground = float(abs(ds["across_track_m"].isel(ping=p, channel=ch, sample=s).values))
            slant = s * m_s
            rows.append({**r, "time": t[p], "heading_deg": float(head[p]), "side": "port" if ch == 0 else "starboard",
                         "slant_range_m": round(slant, 2), "ground_range_m": round(ground, 2),
                         "altitude_m": round(float(alt[p]), 2),
                         "grazing_deg": round(math.degrees(math.atan2(float(alt[p]), ground)), 1)
                         if np.isfinite(alt[p]) and ground > 0 else np.nan,
                         "look_bearing_deg": round((float(head[p]) + (90 if ch == 1 else -90)) % 360, 1)})
    out = pd.DataFrame(rows).sort_values(["object_id", "time"])
    keep = []
    for _, g in out.groupby("object_id"):
        gap = g.time.diff().dt.total_seconds().fillna(np.inf) > pass_gap_s
        g = g.assign(pass_no=np.cumsum(gap.values))
        keep.append(g.loc[g.groupby("pass_no").dist_m.idxmin()])
    out = pd.concat(keep).reset_index(drop=True)
    out.insert(0, "look_id", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# chips


def sample_window(vals, coord, p_c, mpp, along, out, c_c):
    """Values at (along-track offset, outward offset) pairs, metres from the look.

    ``vals``/``coord``: (pings, samples) of one side, ``coord`` the outward
    position of each sample (slant or ground metres); ``p_c`` the look's row,
    ``c_c`` its outward coordinate. Linear in both directions; NaN outside.
    """
    shape = along.shape
    along, out = along.ravel(), out.ravel()
    r = p_c + along / mpp
    target = c_c + out
    res = np.full(along.shape, np.nan, np.float64)
    r0 = np.floor(r).astype(int)
    w = r - r0
    acc = np.zeros(along.shape)
    wsum = np.zeros(along.shape)
    for shift, weight in ((0, 1 - w), (1, w)):
        rows = r0 + shift
        valid = (rows >= 0) & (rows < vals.shape[0])
        for row in np.unique(rows[valid]):
            m = valid & (rows == row)
            cr, vr = coord[row], vals[row]
            ok = np.isfinite(cr)
            if ok.sum() < 2:
                continue
            v = np.interp(target[m], cr[ok], vr[ok], left=np.nan, right=np.nan)
            good = np.isfinite(v)
            idx = np.flatnonzero(m)[good]
            acc[idx] += weight[m][good] * v[good]
            wsum[idx] += weight[m][good]
    filled = wsum > 0.5
    res[filled] = acc[filled] / wsum[filled]
    return res.reshape(shape)


def look_chips(ds, look, chip_m, chip_px, m_s, offset_en=(0.0, 0.0), geometries=GEOMETRIES,
               frames=("track", "north_up")):
    """Chips of one look: {(geometry, frame): float image}.

    ``offset_en`` (east, north metres) moves the chip centre from the look's
    position, e.g. onto the object after registration.
    """
    p, ch = int(look["ping"]), int(look["channel"])
    h = math.radians(look["heading_deg"])
    de, dn = offset_en
    along_off = de * math.sin(h) + dn * math.cos(h)
    right_off = de * math.cos(h) - dn * math.sin(h)  # positive to starboard
    out_off = right_off if ch == 1 else -right_off
    half = chip_m * 0.75 + math.hypot(de, dn)  # north-up needs the corners: sqrt(2)/2 * chip_m
    lo, hi = max(0, p - 600), min(ds.sizes["ping"], p + 600)
    mpp = along_track_spacing_m(ds.isel(ping=slice(lo, hi)))
    n = int(math.ceil(half / mpp)) + 2
    w0, w1 = max(0, p - n), min(ds.sizes["ping"], p + n + 1)
    win = ds.isel(ping=slice(w0, w1), channel=ch)
    vals = win["backscatter_intensity"].values.astype(np.float64)
    ground = np.abs(win["across_track_m"].values.astype(np.float64))
    slant = np.broadcast_to(np.arange(vals.shape[1]) * m_s, vals.shape)
    coords = {"slant": slant, "ground": ground}
    centre = {"slant": look["sample"] * m_s,
              "ground": float(ground[p - w0, look["sample"]])}
    px = chip_m / chip_px
    g = (np.arange(chip_px) + 0.5 - chip_px / 2) * px
    out_track, along_track = np.meshgrid(g, g)  # rows: along track (forward down), columns: away from the sonar
    e, nn = np.meshgrid(g, -g)  # north-up: rows north (up) to south, columns west to east
    along_nu = e * math.sin(h) + nn * math.cos(h)
    right = e * math.cos(h) - nn * math.sin(h)
    out_nu = right if ch == 1 else -right
    grids = {"track": (along_track + along_off, out_track + out_off), "north_up": (along_nu + along_off, out_nu + out_off)}
    chips = {}
    for geom in geometries:
        if not np.isfinite(centre[geom]):
            continue
        for frame in frames:
            chips[(geom, frame)] = sample_window(vals, coords[geom], p - w0, mpp, *grids[frame],
                                                 centre[geom]).astype(np.float32)
    return chips


def look_extents(ds, look, m_s, search_m=2.0, **proposal_kw):
    """Across-track size of the object at a look, in slant and in ground metres.

    Runs the shadow-proposal detector on +-15 m of pings around the look and
    takes the proposal on the look's side nearest the look (within
    ``search_m``). The highlight runs from the box's near edge to the shadow
    start, the shadow from there to the far edge. Along-track length is the
    same in both projections.
    """
    from sidescanpy.analysis import detect_shadow_proposals

    p, ch = int(look["ping"]), int(look["channel"])
    lo, hi = max(0, p - 600), min(ds.sizes["ping"], p + 600)
    mpp = along_track_spacing_m(ds.isel(ping=slice(lo, hi)))
    n = int(math.ceil(15.0 / mpp))
    w0, w1 = max(0, p - n), min(ds.sizes["ping"], p + n + 1)
    sub = ds.isel(ping=slice(w0, w1)).load()
    props = detect_shadow_proposals(sub, m_per_ping=mpp, channels=(ch,), **proposal_kw).proposals
    if props.empty:
        return None
    cp = (props.ping_start + props.ping_stop) / 2 + w0
    cs = (props.sample_start + props.sample_stop) / 2
    dist = np.hypot((cp - p) * mpp, (cs - look["sample"]) * m_s)
    k = int(np.argmin(dist.values))
    if dist.iloc[k] > search_m:
        return None
    r = props.iloc[k]
    row = int(round((r.ping_start + r.ping_stop) / 2))
    g = np.abs(sub["across_track_m"].isel(ping=min(row, sub.sizes["ping"] - 1), channel=ch).values)

    def gr(sample):
        s = int(np.clip(round(sample), 0, len(g) - 1))
        return float(g[s])

    s0, s1, s2 = float(r.sample_start), float(r.shadow_sample_start), float(r.sample_stop)
    return {"highlight_slant_m": (s1 - s0) * m_s, "highlight_ground_m": gr(s1) - gr(s0),
            "shadow_slant_m": (s2 - s1) * m_s, "shadow_ground_m": gr(s2) - gr(s1),
            "object_slant_m": (s2 - s0) * m_s, "object_ground_m": gr(s2) - gr(s0),
            "along_m": float((r.ping_stop - r.ping_start) * mpp), "proposal_score": float(r.score),
            "height_m": float(r.height_m) if np.isfinite(r.height_m) else np.nan}


def extents_figure(looks, path):
    """Across-track size vs ground range in both projections, and its spread per object."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[1].sharex(axes[0])  # highlight and shadow on one scale, so they compare directly
    axes[1].sharey(axes[0])
    colours = {"slant": "#d95f02", "ground": "#1b9e77"}
    for ax, part in zip(axes[:2], ("highlight", "shadow")):
        for geom in GEOMETRIES:
            ax.scatter(looks.ground_range_m, looks[f"{part}_{geom}_m"], s=10, alpha=0.6, color=colours[geom],
                       label=f"{geom} range")
        ax.set(xlabel="ground range of the look (m)", ylabel=f"{part} across-track length (m)",
               title=f"{part}: same objects, all looks")
        ax.legend()
        ax.grid(alpha=0.3)
    ax = axes[2]
    cv = []
    for part in ("highlight", "shadow", "object"):
        for geom in GEOMETRIES:
            c = looks.groupby("object_id")[f"{part}_{geom}_m"].agg(lambda v: v.std() / v.mean() if len(v.dropna()) >= 2
                                                                    and v.mean() > 0 else np.nan)
            cv.append((part, geom, float(c.median())))
    x = np.arange(3)
    for k, geom in enumerate(GEOMETRIES):
        ax.bar(x + (k - 0.5) * 0.38, [v for pt, g, v in cv if g == geom], 0.36, color=colours[geom], label=geom)
    ax.set_xticks(x)
    ax.set_xticklabels(["highlight", "shadow", "highlight + shadow"])
    ax.set(ylabel="median coefficient of variation across looks",
           title="how much an object's across-track size changes\nbetween looks (lower = more consistent)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {f"{pt}_{g}": round(v, 3) for pt, g, v in cv}


# ---------------------------------------------------------------------------
# similarity


def downsample(img, n):
    k = img.shape[0] // n
    a = img[: n * k, : n * k].reshape(n, k, n, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(a, axis=3), axis=1) if np.isfinite(a).any() else np.full((n, n), np.nan)


def disk_mask(n, frac=0.45):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    return np.hypot(x, y) <= frac * n


def masked_ncc(a, b, max_shift, return_shift=False):
    """Best normalised cross-correlation of a and b over shifts up to max_shift
    pixels, NaN-aware (masked NCC, Padfield 2012), using only overlaps of at
    least a third of the pixels. With ``return_shift``, also the shift
    ``(dy, dx)`` at the peak: the content of ``b`` at ``q`` matches ``a`` at
    ``q + (dy, dx)``."""
    ma, mb = np.isfinite(a), np.isfinite(b)
    if ma.sum() < 20 or mb.sum() < 20:
        return (np.nan, (0, 0)) if return_shift else np.nan
    A, B = np.where(ma, a, 0.0), np.where(mb, b, 0.0)
    ma, mb = ma.astype(float), mb.astype(float)
    Br, mbr = B[::-1, ::-1], mb[::-1, ::-1]

    def corr(x, y):
        return fftconvolve(x, y, mode="full")

    n = corr(ma, mbr)
    sa, sb = corr(A, mbr), corr(ma, Br)
    saa, sbb = corr(A * A, mbr), corr(ma, Br * Br)
    sab = corr(A, Br)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = sab - sa * sb / n
        den = np.sqrt(np.maximum(saa - sa * sa / n, 0) * np.maximum(sbb - sb * sb / n, 0))
        ncc = num / den
    c0, c1 = a.shape[0] - 1, a.shape[1] - 1
    win = ncc[c0 - max_shift:c0 + max_shift + 1, c1 - max_shift:c1 + max_shift + 1]
    nwin = n[c0 - max_shift:c0 + max_shift + 1, c1 - max_shift:c1 + max_shift + 1]
    win = np.where(nwin >= max(20, min(ma.sum(), mb.sum()) / 3), win, np.nan)
    if not np.isfinite(win).any():
        return (np.nan, (0, 0)) if return_shift else np.nan
    if not return_shift:
        return float(np.nanmax(win))
    iy, ix = np.unravel_index(np.nanargmax(win), win.shape)
    return float(win[iy, ix]), (int(iy - max_shift), int(ix - max_shift))


def rotate_nan(img, angle):
    m = np.isfinite(img)
    r = ndimage.rotate(np.where(m, img, 0.0), angle, reshape=False, order=1)
    rm = ndimage.rotate(m.astype(float), angle, reshape=False, order=1)
    r[rm < 0.99] = np.nan
    return r


def prepare(img, n, mask):
    d = downsample(img, n)
    d[~mask] = np.nan
    return d


def pair_scores(ca, cb, n, max_shift, rot_step):
    out = {}
    mask = disk_mask(n)
    for geom in GEOMETRIES:
        if (geom, "track") not in ca or (geom, "track") not in cb:
            continue
        out[(geom, "track")] = masked_ncc(prepare(ca[(geom, "track")], n, mask),
                                          prepare(cb[(geom, "track")], n, mask), max_shift)
        a = prepare(ca[(geom, "north_up")], n, mask)
        b_full = cb[(geom, "north_up")]
        out[(geom, "north_up")] = masked_ncc(a, prepare(b_full, n, mask), max_shift)
        best = out[(geom, "north_up")]
        for ang in range(rot_step, 360, rot_step):
            s = masked_ncc(a, prepare(rotate_nan(b_full, ang), n, mask), max_shift)
            if np.isfinite(s) and (not np.isfinite(best) or s > best):
                best = s
        out[(geom, "best_rotation")] = best
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def smooth_nan(img, sigma_px):
    """Gaussian low-pass that ignores NaN (normalised convolution); NaN stays NaN."""
    if sigma_px <= 0:
        return img
    m = np.isfinite(img)
    num = ndimage.gaussian_filter(np.where(m, img, 0.0), sigma_px)
    den = ndimage.gaussian_filter(m.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out


def register_object(looks, reg_chips, px, max_shift, min_ncc, smooth_px=0.0):
    """Offsets (east, north m) that move each look's centre onto the object,
    relative to a reference look (the one with the most valid pixels).
    ``smooth_px``: low-pass before matching, so large seabed structure rather
    than look-dependent shadows drives the match."""
    reg_chips = {lk: smooth_nan(reg_chips[lk], smooth_px) for lk in looks.look_id}
    valid = {lk: float(np.isfinite(reg_chips[lk]).mean()) for lk in looks.look_id}
    ref = max(valid, key=valid.get)
    out = {ref: {"de_m": 0.0, "dn_m": 0.0, "reg_ncc": 1.0, "reference": True}}
    for lk in looks.look_id:
        if lk == ref:
            continue
        v, (sy, sx) = masked_ncc(reg_chips[ref], reg_chips[lk], max_shift, return_shift=True)
        out[lk] = {"de_m": -sx * px, "dn_m": sy * px, "reg_ncc": v, "reference": False,
                   "registered": bool(np.isfinite(v) and v >= min_ncc)}
    out[ref]["registered"] = True
    return ref, out


def register_by_points(looks, pts, radius_m, max_m, tol_m, min_matches):
    """Offsets from the pattern of detected objects around each object.

    ``pts``: other detections (x, y in the local frame, file, time). For each
    look, the detections of the same pass within ``radius_m`` of the object
    are matched against the reference look's (the look with the most): every
    difference vector votes, the densest ``tol_m`` neighbourhood wins, and
    the offset is the mean of its votes. Independent of look direction (an
    object is detected wherever its shadow points), unlike image
    correlation. Returns {look_id: {...}}.
    """
    out = {}
    for _, g in looks.groupby("object_id"):
        sets = {}
        for lk in g.itertuples():
            t = pd.Timestamp(lk.time)
            m = (pts.file == lk.file) & ((pts.time - t).abs() < pd.Timedelta(seconds=120))
            near = pts[m & (np.hypot(pts.x - lk.ox, pts.y - lk.oy) < radius_m)]
            sets[lk.look_id] = near[["x", "y"]].values
        ref = max(sets, key=lambda k: len(sets[k]))
        out[ref] = {"de_m": 0.0, "dn_m": 0.0, "reg_matches": len(sets[ref]), "reference": True, "registered": True}
        for lk, P in sets.items():
            if lk == ref:
                continue
            Q = sets[ref]
            if len(P) < min_matches or len(Q) < min_matches:
                out[lk] = {"reg_matches": 0, "reference": False, "registered": False}
                continue
            d = (P[:, None, :] - Q[None, :, :]).reshape(-1, 2)
            d = d[np.hypot(d[:, 0], d[:, 1]) <= max_m]
            if not len(d):
                out[lk] = {"reg_matches": 0, "reference": False, "registered": False}
                continue
            counts = np.array([len(n) for n in cKDTree(d).query_ball_point(d, tol_m)])
            best = d[np.hypot(*(d - d[counts.argmax()]).T) <= tol_m]
            de, dn = best.mean(0)
            n = len(best)
            out[lk] = {"de_m": float(de), "dn_m": float(dn), "reg_matches": n, "reference": False,
                       "registered": bool(n >= min_matches)}
    return out


def fit_nav_offset(looks: pd.DataFrame) -> dict:
    """Least-squares constant position error in the vessel frame, from the
    registration offsets: d = E_look - E_ref with E = a * forward + c * starboard
    (a: along-track error, e.g. latency x speed or layback; c: across-track,
    e.g. a lateral sensor offset)."""
    rows, rhs = [], []
    by_obj = {o: g.set_index("look_id") for o, g in looks.groupby("object_id")}
    for _, g in by_obj.items():
        ref = g[g.reference]
        if ref.empty:
            continue
        hr = math.radians(ref.heading_deg.iloc[0])
        for _, r in g[~g.reference & g.registered].iterrows():
            h = math.radians(r.heading_deg)
            fwd = (math.sin(h) - math.sin(hr), math.cos(h) - math.cos(hr))
            stb = (math.cos(h) - math.cos(hr), -math.sin(h) + math.sin(hr))
            rows += [[fwd[0], stb[0]], [fwd[1], stb[1]]]
            rhs += [r.de_m, r.dn_m]
    if len(rows) < 6:
        return {"pairs": len(rows) // 2}
    A, b = np.array(rows), np.array(rhs)
    (a, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ np.array([a, c])
    return {"pairs": len(rows) // 2, "along_track_error_m": round(float(a), 2),
            "starboard_error_m": round(float(c), 2),
            "offset_rms_m": round(float(np.sqrt(np.mean(b ** 2))), 2),
            "residual_rms_m": round(float(np.sqrt(np.mean(resid ** 2))), 2)}


def offsets_figure(looks, fit, path):
    reg = looks[~looks.reference & looks.registered]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.scatter(reg.de_m, reg.dn_m, s=10, alpha=0.6)
    ax.set(aspect="equal", xlabel="east offset (m)", ylabel="north offset (m)",
           title="registration offset of each look\n(relative to its object's reference look)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ref_head = looks[looks.reference].set_index("object_id").heading_deg
    dh = (reg.heading_deg - reg.object_id.map(ref_head) + 180) % 360 - 180
    ax = axes[1]
    ax.scatter(dh, np.hypot(reg.de_m, reg.dn_m), s=10, alpha=0.6)
    ax.set(xlabel="heading difference to the reference look (deg)", ylabel="offset (m)",
           title="offsets grow with heading difference\nif the error is fixed in the vessel frame")
    ax.grid(alpha=0.3)
    ax = axes[2]
    q = "reg_matches" if looks.reg_matches.notna().any() else "reg_ncc"
    ax.hist(looks[~looks.reference][q].dropna(), bins=30)
    ax.set(xlabel=q.replace("_", " "), ylabel="looks", title="registration quality")
    if "along_track_error_m" in fit:
        fig.suptitle(f"fitted vessel-frame position error: along track {fit['along_track_error_m']} m, starboard "
                     f"{fit['starboard_error_m']} m; offset RMS {fit['offset_rms_m']} m -> residual "
                     f"{fit['residual_rms_m']} m over {fit['pairs']} looks", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figures


def stretch(imgs):
    v = np.concatenate([i[np.isfinite(i)] for i in imgs if np.isfinite(i).any()] or [np.zeros(1)])
    return np.percentile(v, 1), np.percentile(v, 99.5)


def object_sheet(obj, looks, chips, path, chip_m):
    cols = [("slant", "track"), ("ground", "track"), ("slant", "north_up"), ("ground", "north_up")]
    n = len(looks)
    fig, axes = plt.subplots(n, 4, figsize=(9.6, 2.55 * n + 0.6), squeeze=False)
    lims = {g: stretch([chips[lk.look_id][(g, f)] for lk in looks.itertuples() for f in ("track", "north_up")
                        if (g, f) in chips[lk.look_id]]) for g in GEOMETRIES}
    for i, lk in enumerate(looks.itertuples()):
        for j, key in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            img = chips[lk.look_id].get(key)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=lims[key[0]][0], vmax=lims[key[0]][1],
                          extent=[-chip_m / 2, chip_m / 2, -chip_m / 2, chip_m / 2])
            if i == 0:
                ax.set_title(f"{key[0]} range\n{'track frame' if key[1] == 'track' else 'north up'}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"look {lk.look_id}\n{lk.side[:4]}, hdg {lk.heading_deg:.0f}\n"
                              f"slant {lk.slant_range_m:.0f} m\nground {lk.ground_range_m:.0f} m\n"
                              f"graze {lk.grazing_deg:.0f}°", fontsize=7)
    fig.suptitle(f"object {obj.object_id} ({obj.sources}), {n} looks; chips {chip_m:g} m; track frame: rows along "
                 "track, sonar on the left; north up: rotated by heading and side", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_figure(pairs, path):
    same, diff = pairs[pairs.same], pairs[~pairs.same]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colours = {"slant": "#d95f02", "ground": "#1b9e77"}
    ax = axes[0, 0]
    pos = 0
    ticks, labels = [], []
    for frame in FRAMES:
        for geom in GEOMETRIES:
            col = f"{geom}_{frame}"
            for k, (sub, hatch) in enumerate(((same, None), (diff, "//"))):
                v = sub[col].dropna().values
                if len(v):
                    bp = ax.boxplot(v, positions=[pos + 0.4 * k], widths=0.35, patch_artist=True, showfliers=False)
                    bp["boxes"][0].set(facecolor=colours[geom], alpha=0.8 if k == 0 else 0.3, hatch=hatch)
            ticks.append(pos + 0.2)
            labels.append(f"{geom}\n{frame.replace('_', ' ')}")
            pos += 1.2
        pos += 0.4
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("NCC")
    ax.set_title("similarity: same object (solid) vs different objects (hatched)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(FRAMES))
    for k, geom in enumerate(GEOMETRIES):
        a = [auc(same[f"{geom}_{f}"], diff[f"{geom}_{f}"]) for f in FRAMES]
        ax.bar(x + (k - 0.5) * 0.38, a, 0.36, color=colours[geom], label=geom)
        for xi, ai in zip(x, a):
            ax.text(xi + (k - 0.5) * 0.38, ai + 0.01, f"{ai:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ") for f in FRAMES])
    ax.set(ylim=(0.4, 1.0), ylabel="AUC same vs different",
           title="how well similarity tells the same object\nfrom a different one (0.5 = not at all)")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.legend()

    ax = axes[0, 2]
    d = same.dropna(subset=["slant_north_up", "ground_north_up"])
    ax.scatter(d.slant_north_up, d.ground_north_up, s=8, alpha=0.5, c="#555")
    lim = [np.nanmin([d.slant_north_up.min(), d.ground_north_up.min(), 0]), 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlabel="slant NCC (north up)", ylabel="ground NCC (north up)",
           title=f"same-object pairs: ground above the line in {np.mean(d.ground_north_up > d.slant_north_up):.0%}")

    for ax, (dcol, xlabel) in zip(axes[1, :2], (("d_ground_range_m", "difference in ground range between looks (m)"),
                                                ("d_look_bearing_deg", "difference in look direction (deg)"))):
        bins = np.quantile(same[dcol].dropna(), np.linspace(0, 1, 6)) if len(same) > 10 else np.array([0, 1])
        bins = np.unique(bins)
        mids = (bins[:-1] + bins[1:]) / 2
        for geom in GEOMETRIES:
            for frame, ls in (("track", ":"), ("north_up", "-")):
                col = f"{geom}_{frame}"
                med = [same[(same[dcol] >= a) & (same[dcol] <= b)][col].median() for a, b in zip(bins[:-1], bins[1:])]
                ax.plot(mids, med, ls, marker="o", color=colours[geom], label=f"{geom}, {frame.replace('_', ' ')}")
        ax.set(xlabel=xlabel, ylabel="median NCC, same object")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes[1, 2]
    for geom in GEOMETRIES:
        ax.hist(same[f"{geom}_north_up"].dropna() - diff[f"{geom}_north_up"].dropna().median(), bins=30, alpha=0.5,
                color=colours[geom], label=f"{geom}")
    ax.set(xlabel="same-object NCC minus the different-object median (north up)", ylabel="pairs",
           title="margin over the baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gallery(objs, looks, chips, path, chip_m, n_objects=6, max_looks=6):
    pick = (looks.groupby("object_id").agg(n=("look_id", "size"), spread=("look_bearing_deg", lambda s: np.ptp(s)))
            .query("n >= 2").sort_values(["spread", "n"], ascending=False).head(n_objects).index)
    if not len(pick):
        return
    fig, axes = plt.subplots(2 * len(pick), max_looks, figsize=(1.9 * max_looks, 2.0 * 2 * len(pick)), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for i, oid in enumerate(pick):
        lk = looks[looks.object_id == oid].sort_values("look_bearing_deg").head(max_looks)
        for g, geom in enumerate(GEOMETRIES):
            imgs = [chips[x][(geom, "north_up")] for x in lk.look_id if (geom, "north_up") in chips[x]]
            lo, hi = stretch(imgs)
            for j, x in enumerate(lk.itertuples()):
                if (geom, "north_up") not in chips[x.look_id]:
                    continue
                ax = axes[2 * i + g, j]
                ax.imshow(chips[x.look_id][(geom, "north_up")], cmap="gray", vmin=lo, vmax=hi)
                ax.set_title(f"{geom} | look {x.look_bearing_deg:.0f}° | gr {x.ground_range_m:.0f} m", fontsize=6)
    fig.suptitle(f"same objects, north up, {chip_m:g} m chips: slant (odd rows) vs ground (even rows), "
                 "sorted by look direction", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ss_h5_dir", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--detections", type=Path,
                     help="detections from every pass (lat, lon, file, ping, channel, sample, time[, conf]): "
                          "objects = isolated detections seen on several passes, each look centred on its own "
                          "detection (recommended: immune to navigation error between passes)")
    src.add_argument("--points", type=Path, help="CSV with lat, lon (id, source optional): looks found by position")
    src.add_argument("--proposals", action="store_true",
                     help="make the detections with sidescanpy's shadow-proposal detector (no other input needed)")
    ap.add_argument("--block-pings", type=int, default=1000,
                    help="--proposals: pings per detector block (lower it if memory is tight)")
    ap.add_argument("--min-length-m", type=float, default=0.0,
                    help="--proposals: keep objects at least this long along or across track (2.5: skip traps)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="--detections: minimum conf")
    ap.add_argument("--link-m", type=float, default=8.0,
                    help="--detections: largest position difference of one object between passes (m)")
    ap.add_argument("--size-tol", type=float, default=None,
                    help="--detections with a length_along_m column: drop objects whose looks disagree on "
                         "along-track length by more than this fraction (e.g. 0.5)")
    ap.add_argument("--isolation-m", type=float, default=8.0,
                    help="--detections: no other detection of the same pass this close (m)")
    ap.add_argument("--frequency", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chip-m", type=float, default=12.0)
    ap.add_argument("--chip-px", type=int, default=256)
    ap.add_argument("--cluster-m", type=float, default=3.0)
    ap.add_argument("--max-dist-m", type=float, default=1.0)
    ap.add_argument("--pass-gap-s", type=float, default=60.0)
    ap.add_argument("--min-looks", type=int, default=2)
    ap.add_argument("--max-objects", type=int, default=400, help="most-looked-at objects kept")
    ap.add_argument("--max-shift-m", type=float, default=0.75,
                    help="residual misalignment absorbed by the similarity NCC (after registration)")
    ap.add_argument("--min-ground-m", type=float, default=3.0, help="looks closer to nadir are dropped")
    ap.add_argument("--register-points", type=Path, default=None,
                    help="CSV of detected objects (lat, lon, file, time) from every pass: align the looks of each "
                         "object by the pattern of detections around it (recommended)")
    ap.add_argument("--register-image", action="store_true",
                    help="align looks by image correlation instead (unreliable across look directions)")
    ap.add_argument("--reg-radius-m", type=float, default=30.0, help="detections used around each object (m)")
    ap.add_argument("--reg-tol-m", type=float, default=1.0, help="vote tolerance for point registration (m)")
    ap.add_argument("--reg-min-matches", type=int, default=6, help="looks with fewer matched detections are dropped")
    ap.add_argument("--reg-m", type=float, default=24.0, help="registration chip size (m)")
    ap.add_argument("--reg-px", type=int, default=160)
    ap.add_argument("--reg-max-m", type=float, default=20.0, help="largest registration offset searched (m)")
    ap.add_argument("--reg-min-ncc", type=float, default=0.3, help="looks registering worse are dropped")
    ap.add_argument("--reg-smooth-m", type=float, default=0.0,
                    help="Gaussian low-pass (sigma, m) before registration: matches large structure, not shadows")
    ap.add_argument("--score-px", type=int, default=64, help="chips are averaged to this size for scoring")
    ap.add_argument("--rotation-step", type=int, default=15, help="degrees, for the best-rotation score")
    ap.add_argument("--sheets", type=int, default=40, help="per-object sheets written (most looks first)")
    ap.add_argument("--no-extents", action="store_true", help="skip measuring object sizes per look")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not logging.getLogger().handlers:  # a calling script may have set up logging already
        configure_logging("INFO")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "objects").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.proposals:
        a.detections = proposal_detections(sorted(a.ss_h5_dir.glob("*.ss.h5")), a.frequency,
                                           a.out / "proposal_detections.csv", block_pings=a.block_pings,
                                           min_score=a.min_conf, min_length_m=a.min_length_m)
    if a.detections:
        objs, raw, frame = isolated_objects(a.detections, a.link_m, a.isolation_m, a.pass_gap_s, a.min_conf,
                                            a.size_tol)
        logger.info("%d isolated objects detected on >= 2 passes (link %.1f m, isolation %.1f m)", len(objs),
                    a.link_m, a.isolation_m)
        if raw.empty:
            raise SystemExit("no isolated object is detected on two or more passes")
    else:
        objs, frame = load_objects(a.points, a.cluster_m)
        logger.info("%d points -> %d objects (cluster %.1f m)", objs.n_points.sum(), len(objs), a.cluster_m)
        files = sorted(a.ss_h5_dir.glob("*.ss.h5"))
        raw = find_looks(files, a.frequency, objs, frame, a.max_dist_m)
        if raw.empty:
            raise SystemExit("no object lies within --max-dist-m of a georeferenced sample")
    looks = describe_looks(raw, a.ss_h5_dir, a.frequency, a.pass_gap_s)
    counts = looks.groupby("object_id").size()
    multi = counts[counts >= a.min_looks].sort_values(ascending=False).head(a.max_objects).index
    looks = looks[looks.object_id.isin(multi)].reset_index(drop=True)
    logger.info("%d objects seen on >= %d passes, %d looks", len(multi), a.min_looks, len(looks))

    looks = looks[looks.ground_range_m >= a.min_ground_m].reset_index(drop=True)

    # registration: the same lat/lon can land on different seabed in different
    # passes (navigation, layback, latency); align every look of an object to
    # one reference look on a wide north-up ground-range chip
    reg_px_m = a.reg_m / a.reg_px
    looks["de_m"], looks["dn_m"], looks["reg_ncc"], looks["reg_matches"] = 0.0, 0.0, np.nan, np.nan
    looks["reference"], looks["registered"] = False, True
    if a.register_points:
        pts = pd.read_csv(a.register_points)
        pts = pts[np.isfinite(pts.lat) & np.isfinite(pts.lon)].copy()
        pts["x"], pts["y"] = frame.xy(pts.lat, pts.lon)
        pts["time"] = pd.to_datetime(pts.time)
        oxy = objs.set_index("object_id")[["x", "y"]]
        looks["ox"], looks["oy"] = looks.object_id.map(oxy.x), looks.object_id.map(oxy.y)
        res = register_by_points(looks, pts, a.reg_radius_m, a.reg_max_m, a.reg_tol_m, a.reg_min_matches)
        for lk, r in res.items():
            i = looks.index[looks.look_id == lk][0]
            for k, v in r.items():
                looks.at[i, k] = v
        logger.info("registered %d of %d looks by detected-object patterns (>= %d matches)",
                    int(looks.registered.sum()), len(looks), a.reg_min_matches)
        looks = looks[looks.registered].reset_index(drop=True)
        counts = looks.groupby("object_id").size()
        looks = looks[looks.object_id.isin(counts[counts >= a.min_looks].index)].reset_index(drop=True)
    elif a.register_image:
        reg_chips = {}
        for fname, g in looks.groupby("file"):
            ds = load_channel_dataset(a.ss_h5_dir / fname, a.frequency, {"ping": 2048})
            m_s = nominal_sample_interval_m(ds)
            for lk in g.to_dict("records"):
                reg_chips[lk["look_id"]] = look_chips(ds, lk, a.reg_m, a.reg_px, m_s, geometries=("ground",),
                                                      frames=("north_up",)).get(("ground", "north_up"),
                                                                                 np.full((a.reg_px, a.reg_px), np.nan))
        max_reg = int(round(a.reg_max_m / reg_px_m))
        for _oid, g in looks.groupby("object_id"):
            _, res = register_object(g, reg_chips, reg_px_m, max_reg, a.reg_min_ncc, a.reg_smooth_m / reg_px_m)
            for lk, r in res.items():
                i = looks.index[looks.look_id == lk][0]
                for k, v in r.items():
                    looks.at[i, k] = v
        logger.info("registered %d of %d looks (NCC >= %.2f)", int(looks.registered.sum()), len(looks),
                    a.reg_min_ncc)
        looks = looks[looks.registered].reset_index(drop=True)
        counts = looks.groupby("object_id").size()
        looks = looks[looks.object_id.isin(counts[counts >= a.min_looks].index)].reset_index(drop=True)

    chips, extents = {}, {}
    for fname, g in looks.groupby("file"):
        ds = load_channel_dataset(a.ss_h5_dir / fname, a.frequency, {"ping": 2048})
        m_s = nominal_sample_interval_m(ds)
        for lk in g.to_dict("records"):
            chips[lk["look_id"]] = look_chips(ds, lk, a.chip_m, a.chip_px, m_s, offset_en=(lk["de_m"], lk["dn_m"]))
            if not a.no_extents:
                extents[lk["look_id"]] = look_extents(ds, lk, m_s)
    # drop looks whose chip is mostly empty (water column, swath edge)
    valid = {k: float(np.isfinite(v.get(("ground", "track"), np.full(1, np.nan))).mean()) for k, v in chips.items()}
    looks["valid_fraction"] = looks.look_id.map(valid)
    looks = looks[looks.valid_fraction >= 0.5]
    counts = looks.groupby("object_id").size()
    looks = looks[looks.object_id.isin(counts[counts >= a.min_looks].index)].reset_index(drop=True)

    if extents:
        ext = pd.DataFrame([{"look_id": k, **v} for k, v in extents.items() if v])
        looks = looks.merge(ext, on="look_id", how="left")
    px_score = a.chip_m / a.score_px
    max_shift = max(1, int(round(a.max_shift_m / px_score)))
    pairs = []
    by_obj = {o: g for o, g in looks.groupby("object_id")}
    all_ids = looks.look_id.values
    obj_xy = objs.set_index("object_id")[["x", "y"]]
    for oid, g in by_obj.items():
        lk = g.to_dict("records")
        for i in range(len(lk)):
            for j in range(i + 1, len(lk)):
                A, B = lk[i], lk[j]
                base = {"object_a": oid, "look_a": A["look_id"], "look_b": B["look_id"],
                        "d_ground_range_m": abs(A["ground_range_m"] - B["ground_range_m"]),
                        "d_slant_range_m": abs(A["slant_range_m"] - B["slant_range_m"]),
                        "d_look_bearing_deg": abs((A["look_bearing_deg"] - B["look_bearing_deg"] + 180) % 360 - 180),
                        "same_side_of_object": abs((A["look_bearing_deg"] - B["look_bearing_deg"] + 180) % 360 - 180)
                        < 90}
                s = pair_scores(chips[A["look_id"]], chips[B["look_id"]], a.score_px, max_shift, a.rotation_step)
                pairs.append({**base, "object_b": oid, "same": True,
                              **{f"{k[0]}_{k[1]}": v for k, v in s.items()}})
                # baseline: look A against a look of another object at least 20 m away
                for _ in range(50):
                    other = looks[looks.look_id == rng.choice(all_ids)].iloc[0]
                    if other.object_id != oid and np.hypot(*(obj_xy.loc[oid] - obj_xy.loc[other.object_id])) > 20:
                        break
                else:
                    continue  # no other object far enough: no baseline pair
                s = pair_scores(chips[A["look_id"]], chips[int(other.look_id)], a.score_px, max_shift,
                                a.rotation_step)
                pairs.append({**base, "look_b": int(other.look_id), "object_b": int(other.object_id), "same": False,
                              "d_ground_range_m": abs(A["ground_range_m"] - other.ground_range_m),
                              "d_slant_range_m": abs(A["slant_range_m"] - other.slant_range_m),
                              **{f"{k[0]}_{k[1]}": v for k, v in s.items()}})
        logger.info("object %d: %d looks scored", oid, len(lk))
    pairs = pd.DataFrame(pairs)

    objs.to_csv(a.out / "objects.csv", index=False)
    looks.to_csv(a.out / "looks.csv", index=False)
    pairs.to_csv(a.out / "pairs.csv", index=False)
    same, diff = pairs[pairs.same], pairs[~pairs.same]
    summary = {"objects": int(looks.object_id.nunique()), "looks": len(looks), "same_pairs": len(same),
               "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "scores": {}}
    for geom in GEOMETRIES:
        for f in FRAMES:
            c = f"{geom}_{f}"
            if c in pairs:
                summary["scores"][c] = {"same_median": round(float(same[c].median()), 3),
                                        "different_median": round(float(diff[c].median()), 3),
                                        "auc_same_vs_different": round(auc(same[c], diff[c]), 3)}
    for f in FRAMES:
        if f"ground_{f}" in same and f"slant_{f}" in same:
            d = same.dropna(subset=[f"ground_{f}", f"slant_{f}"])
            dd = d[f"ground_{f}"] - d[f"slant_{f}"]
            summary["scores"][f"ground_minus_slant_{f}"] = {
                "median": round(float(dd.median()), 3), "ground_higher_fraction": round(float((dd > 0).mean()), 3),
                "pairs": len(d)}
    if extents and "highlight_slant_m" in looks:
        summary["extent_cv_median"] = extents_figure(looks, a.out / "extent_consistency.png")
        ok = looks.dropna(subset=["highlight_slant_m"])
        summary["extent_looks"] = len(ok)
        for part in ("highlight", "object"):
            r = (ok[f"{part}_ground_m"] / ok[f"{part}_slant_m"]).replace([np.inf, -np.inf], np.nan)
            summary[f"{part}_ground_over_slant_vs_range"] = {
                f"{lo}-{hi} m": round(float(r[(ok.ground_range_m >= lo) & (ok.ground_range_m < hi)].median()), 3)
                for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 60))}
    registering = bool(a.register_points or a.register_image)
    fit = fit_nav_offset(looks) if registering else {}
    summary["registration"] = {"looks_registered": int(looks.registered.sum()), "nav_offset_fit": fit}
    if registering:
        offsets_figure(looks, fit, a.out / "registration_offsets.png")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Look consistency: slant vs ground range", "",
             f"{summary['objects']} objects, {summary['looks']} looks, {summary['same_pairs']} same-object pairs.",
             "", "| score | same median | different median | AUC |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['same_median']} | {v['different_median']} | {v['auc_same_vs_different']} |"
              for k, v in summary["scores"].items() if "same_median" in v]
    if fit.get("along_track_error_m") is not None:
        lines += ["", f"Registration: fitted vessel-frame position error along track {fit['along_track_error_m']} m, "
                      f"starboard {fit['starboard_error_m']} m (offset RMS {fit['offset_rms_m']} m -> residual "
                      f"{fit['residual_rms_m']} m, {fit['pairs']} looks)."]
    if "extent_cv_median" in summary:
        cvs = summary["extent_cv_median"]
        lines += ["", f"Across-track size of the same object between looks ({summary['extent_looks']} looks measured), "
                      "median coefficient of variation (lower = more consistent):", "",
                  "| part | slant | ground |", "|---|---|---|"]
        lines += [f"| {pt} | {cvs[pt + '_slant']} | {cvs[pt + '_ground']} |" for pt in ("highlight", "shadow", "object")]
    lines += ["", "| ground minus slant (same-object pairs) | median | ground higher |", "|---|---|---|"]
    lines += [f"| {k[len('ground_minus_slant_'):]} | {v['median']} | {v['ground_higher_fraction']:.0%} |"
              for k, v in summary["scores"].items() if k.startswith("ground_minus")]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    summary_figure(pairs, a.out / "similarity_summary.png")
    gallery(objs, looks, chips, a.out / "gallery.png", a.chip_m)
    order = looks.groupby("object_id").size().sort_values(ascending=False).head(a.sheets).index
    for oid in order:
        object_sheet(objs.set_index("object_id").loc[oid].rename(oid).to_frame().T.assign(object_id=oid).iloc[0],
                     looks[looks.object_id == oid].sort_values("look_bearing_deg"), chips,
                     a.out / "objects" / f"object_{oid:05d}.png", a.chip_m)
    logger.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
