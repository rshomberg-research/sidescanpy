"""
Swath tiles for annotation and detection, and boxes drawn on them back to targets.

The round trip a detector workflow needs:

1. :func:`write_swath_tiles` cuts a channel into dual-swath tiles
   (:func:`~sidescanpy.export.windows.native.swath_tiles`), writes each as
   an 8-bit PNG and records where every tile sits in the sonar data in an
   index file (``tiles.json``).
2. People annotate the tiles, or a detector predicts boxes on them, in any
   tool that exports COCO (FiftyOne, CVAT, Label Studio, ...).
   :func:`read_coco_boxes` reads the boxes per image.
3. :func:`boxes_to_targets` turns the boxes on one tile into
   :class:`~sidescanpy.core.target.Target` records: file, ping, side,
   sample and, on a georeferenced channel, latitude and longitude. They
   can be rendered as chips with
   :func:`~sidescanpy.export.ml.build_target_dataset` or scored against a
   catalog with :func:`sidescanpy.targets.score.match_targets`.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.image import save_grayscale_png, to_uint8
from sidescanpy.export.windows.levels import nominal_sample_interval_m
from sidescanpy.export.windows.native import NativeWindow, swath_tiles

logger = logging.getLogger(__name__)

TILE_INDEX_NAME = "tiles.json"


@dataclass(frozen=True)
class TileBox:
    """One box on a tile image, from an annotation or a detector.

    Attributes
    ----------
    x, y, w, h : float
        Box in image pixels: top-left corner and size.
    category : str, optional
        Class name.
    score : float, optional
        Detector confidence; ``None`` for a human annotation.
    """

    x: float
    y: float
    w: float
    h: float
    category: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class TileRecord:
    """Where one tile image sits in the sonar data (an entry of ``tiles.json``).

    Attributes
    ----------
    file_name : str
        Image file, relative to the index.
    ss_h5 : str
        Stem of the ``.ss.h5`` file the tile was cut from.
    freq_key : str
        Frequency channel, e.g. ``"hz_400000"``.
    window : NativeWindow
    """

    file_name: str
    ss_h5: str
    freq_key: str
    window: NativeWindow


def write_swath_tiles(
    ds: xr.Dataset,
    out_dir: str | Path,
    *,
    ss_h5: str,
    freq_key: str,
    tile_m: float,
    px_m: float,
    overlap_m: float = 0.0,
    value_var: str = "backscatter_intensity",
    pct_low: float = 1.0,
    pct_high: float = 99.0,
    gamma: float = 1.0,
) -> list[TileRecord]:
    """Write a channel as dual-swath PNG tiles plus an index for mapping boxes back.

    Parameters
    ----------
    ds : xr.Dataset
        One frequency channel, usually corrected (TVG, so the tiles are even
        across range).
    out_dir : path
        Created if needed. Tiles are written as
        ``<ss_h5>_<freq_key>_<n>.png``; the index ``tiles.json`` there is
        extended, so several channels and files can share one folder.
    ss_h5, freq_key : str
        Where ``ds`` came from: the ``.ss.h5`` file stem and frequency key.
        Recorded in the index so boxes can be traced back to the data.
    tile_m, px_m, overlap_m : float
        Tile length along track, pixel size and along-track overlap, in
        metres (see :func:`~sidescanpy.export.windows.native.swath_tiles`).
    value_var : str, default "backscatter_intensity"
    pct_low, pct_high, gamma : float
        8-bit stretch (see :func:`~sidescanpy.export.image.to_uint8`),
        applied per tile.

    Returns
    -------
    list of TileRecord
        The tiles written by this call.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (image, window) in enumerate(
        swath_tiles(ds, tile_m=tile_m, px_m=px_m, overlap_m=overlap_m, value_var=value_var)
    ):
        name = f"{ss_h5}_{freq_key}_{n:04d}.png"
        pixels, _ = to_uint8(image, pct_low=pct_low, pct_high=pct_high, gamma=gamma)
        save_grayscale_png(pixels, out_dir / name)
        records.append(TileRecord(name, ss_h5, freq_key, window))
    index_path = out_dir / TILE_INDEX_NAME
    index = _read_index_json(index_path) if index_path.exists() else {}
    for r in records:
        index[r.file_name] = {"ss_h5": r.ss_h5, "freq_key": r.freq_key, "window": r.window.to_dict()}
    index_path.write_text(json.dumps(index, indent=1))
    logger.debug("wrote %d tile(s) of %s %s to %s", len(records), ss_h5, freq_key, out_dir)
    return records


def _read_index_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def read_tile_index(path: str | Path) -> dict[str, TileRecord]:
    """Read a ``tiles.json`` written by :func:`write_swath_tiles`.

    Parameters
    ----------
    path : path
        The index file, or the folder holding it.

    Returns
    -------
    dict
        ``{file_name: TileRecord}``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TILE_INDEX_NAME
    return {
        name: TileRecord(name, entry["ss_h5"], entry["freq_key"], NativeWindow.from_dict(entry["window"]))
        for name, entry in _read_index_json(path).items()
    }


def read_coco_boxes(path: str | Path) -> dict[str, list[TileBox]]:
    """Read the boxes of a COCO file, per image file name.

    Works for annotation exports (FiftyOne, CVAT, Label Studio) and for
    detector results written in COCO form: an annotation's ``score`` is
    kept when present.

    Parameters
    ----------
    path : path
        COCO JSON with ``images``, ``annotations`` and ``categories``.

    Returns
    -------
    dict
        ``{file_name: [TileBox, ...]}`` keyed by the image's file name
        without folders. Images with no boxes map to an empty list.
    """
    coco = json.loads(Path(path).read_text())
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    files = {im["id"]: Path(im["file_name"]).name for im in coco.get("images", [])}
    boxes: dict[str, list[TileBox]] = {name: [] for name in files.values()}
    for ann in coco.get("annotations", []):
        x, y, w, h = (float(v) for v in ann["bbox"])
        score = ann.get("score")
        boxes[files[ann["image_id"]]].append(TileBox(
            x, y, w, h,
            category=names.get(ann.get("category_id")),
            score=None if score is None else float(score),
        ))
    return boxes


def boxes_to_targets(
    ds: xr.Dataset,
    record: TileRecord,
    boxes: Iterable[TileBox],
    *,
    source_format: str = "tile_box",
    catalog_source: str = "",
) -> list[Target]:
    """Turn boxes drawn on one tile into targets in the sonar data.

    Each box becomes a :class:`~sidescanpy.core.target.Target` at the box
    centre: the side under the centre (a dual-swath box crossing nadir is
    clipped to that side), the ping and sample there, and, when ``ds`` is
    georeferenced (``latitude_per_sample``), the position of that sample.

    Parameters
    ----------
    ds : xr.Dataset
        The channel the tile was cut from (``record.ss_h5``,
        ``record.freq_key``), with the same pings and samples.
    record : TileRecord
        The tile (from :func:`write_swath_tiles` or :func:`read_tile_index`).
    boxes : iterable of TileBox
    source_format, catalog_source : str
        Recorded on each target (e.g. ``"detector"`` and the model path).

    Returns
    -------
    list of Target
        ``ping`` and ``sample`` are the dataset's ``ping`` and ``sample``
        coordinates at the box centre (the axes a catalog's ping and sample
        refer to). ``extra_attributes`` holds
        ``ss_h5`` and ``ping_index`` (so
        :func:`~sidescanpy.targets.correlate.correlate_target` places the
        target on that exact ping), ``ping_meters_per_sample``,
        ``score``, ``tile`` and the box size in metres (``length_m`` along
        track, ``width_m`` across). ``position_confidence`` is
        ``"verified"`` for a box without a score (a person drew it) and
        ``"approximate"`` for a detection.
    """
    window = record.window
    n_pings, n_samples = ds.sizes["ping"], ds.sizes["sample"]
    m_per_sample = nominal_sample_interval_m(ds)
    has_georef = "latitude_per_sample" in ds and "longitude_per_sample" in ds
    if not has_georef:
        logger.debug("boxes_to_targets: %s has no latitude_per_sample; targets get no position",
                     record.file_name)
    targets = []
    for k, box in enumerate(boxes):
        native = window.box_to_native(box.x, box.y, box.w, box.h)
        channel = native["channel"]
        p = int(np.clip(np.floor((native["ping_start"] + native["ping_stop"]) / 2), 0, n_pings - 1))
        s = int(np.clip(np.floor((native["sample_start"] + native["sample_stop"]) / 2), 0, n_samples - 1))
        lat = lon = None
        if has_georef:
            la = float(ds["latitude_per_sample"].values[p, channel, s])
            lo = float(ds["longitude_per_sample"].values[p, channel, s])
            if np.isfinite(la) and np.isfinite(lo):
                lat, lon = la, lo
        targets.append(Target(
            target_id=f"{Path(record.file_name).stem}_{k:03d}",
            source_format=source_format,
            catalog_source=catalog_source,
            ping=int(ds["ping"].values[p]),
            channel=channel,
            sample=int(ds["sample"].values[s]),
            latitude=lat,
            longitude=lon,
            image_id=record.file_name,
            bbox=(box.x, box.y, box.w, box.h),
            position_confidence="verified" if box.score is None else "approximate",
            category=box.category,
            extra_attributes={
                "ss_h5": record.ss_h5,
                "freq_key": record.freq_key,
                "ping_index": p,
                "ping_meters_per_sample": m_per_sample,
                "score": box.score,
                "tile": record.file_name,
                "length_m": (native["ping_stop"] - native["ping_start"]) * window.m_per_ping,
                "width_m": (native["sample_stop"] - native["sample_start"]) * window.m_per_sample,
            },
        ))
    return targets
