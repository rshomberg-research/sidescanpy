"""
Annotated training datasets of target-centered (and background) images.

:func:`build_target_dataset` turns converted ``.ss.h5`` surveys plus a
target catalog into one COCO + YOLO dataset per export style, rendered at
one render level and configured by a :class:`DatasetSpec`:

1. Targets at or above ``min_confidence`` are correlated to their ping
   (:func:`~sidescanpy.targets.correlate.correlate_target`) and grouped by
   file and frequency.
2. For each target a bounded ping neighborhood is loaded once and every
   style is cut from it ``n_augment`` times (independent jitter, optional
   random flip). Every catalog target inside a window becomes an
   annotation.
3. Optionally, ``n_background`` screened no-target windows per style
   (:func:`~sidescanpy.export.windows.sample_background_windows`).
4. Images are written as single-channel 8- or 16-bit PNGs, with
   ``targets.coco.json``, a YOLO label set, ``augmentations.json`` (when
   draws vary) and a top-level ``manifest.json``.

Plotting (contact sheets, context views) is optional and supplied by the
caller as callables, so this module never imports matplotlib.

Output layout per style directory (``layout="flat"``, the default)::

    <style>/target_<id>[_a<k>].png, bg_<n>[_a<k>].png
    <style>/targets.coco.json
    <style>/targets.yolo/{data.yaml,labels/<stem>.txt}
    <style>/augmentations.json       (n_augment > 1, random_flip or backgrounds)
    <style>/splits/{train,val}.txt   (val_fraction > 0)

``layout="ultralytics"``::

    <style>/images/{train,val}/<file>.png
    <style>/labels/{train,val}/<stem>.txt
    <style>/{train,val}.coco.json
    <style>/data.yaml                (Ultralytics dataset file)

With several resolutions each gets its own ``res_<m>m/`` directory.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from sidescanpy._version import __version__
from sidescanpy.core.target import CONFIDENCE_RANK, Target
from sidescanpy.export.augment import augmented_file_name, random_flip
from sidescanpy.export.image import (
    NORMALIZATION_SCOPES,
    reference_stats_for_scope,
    save_grayscale_png,
    to_uint8,
    to_uint16,
)
from sidescanpy.export.ml.coco import CocoImageEntry, write_coco
from sidescanpy.export.ml.yolo import write_yolo
from sidescanpy.export.orientation import orient_pixel_window
from sidescanpy.export.windows import (
    CHIP_STYLES,
    LEVEL_VARS,
    PIXEL_LEVELS,
    RENDER_LEVELS,
    RESAMPLE_METHODS,
    WindowSpec,
    combine_channels_mirrored,
    extract_geo_style_window,
    extract_pixel_style_window,
    load_geo_neighborhood,
    mask_nadir_gap_with_uncorrected,
    place_targets_in_file,
    sample_background_windows,
    target_combined_col,
    with_bbox_size_px,
    with_db_var,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.targets.correlate import build_position_index, correlate_target

logger = logging.getLogger(__name__)

LAYOUTS = ("flat", "ultralytics")
"""Dataset directory layouts (see the module docstring)."""

ContactSheetFn = Callable[..., None]
"""``contact_sheet(thumbnails, captions, *, boxes, box_size, suptitle, path)``: draw and save a sheet."""

ContextFn = Callable[..., None]
"""``context(image, target, *, coordinate_system, resolution_m, box_size, path)``: draw and save a context view."""


# ============================================================
# Spec
# ============================================================

@dataclass
class DatasetSpec:
    """Everything that shapes a target dataset (defaults match ``sidescan-export chip``).

    Attributes
    ----------
    styles : tuple of str
        Subset of :data:`~sidescanpy.export.windows.CHIP_STYLES`.
    render_level : str
        One of :data:`~sidescanpy.export.windows.RENDER_LEVELS`.
    size_px, swath_size_px : int
        Output size of ``chip`` / the swath styles.
    resolution_m : float, list of float, or None
        Geometric/georef meters per pixel; ``None`` chooses per window; a
        list builds one dataset per resolution under ``res_<m>m/``.
    value_var : str
        Geometric/georef (and pixel context) values: ``uncorrected``,
        ``db`` or ``radiometric``.
    resample_method, across_axis, swath_fill
        See :func:`~sidescanpy.export.windows.extract_geo_style_window`.
    box_size : float
        Placeholder annotation box (pixels) for targets with no measured bbox.
    box_size_m : float, optional
        Geometric/georef placeholder box in meters instead.
    with_context : bool
        Write ``context/context_<id>.png`` views (needs a ``context`` callable).
    jitter_px, n_augment, random_flip
        Augmentation: center jitter, draws per target, random flips.
    gamma, pct_low, pct_high, normalization_scope, bit_depth
        Display transform (see :func:`~sidescanpy.export.image.to_uint8`).
    min_confidence : str
        Targets below this are neither rendered nor annotated (backgrounds
        still avoid them).
    categories : list of str, optional
        Fixed class order (see :func:`~sidescanpy.export.ml.coco.resolve_categories`).
    max_match_dist_m : float
        Target-to-ping correlation distance.
    seed : int, optional
    n_background, background_margin_m, background_max_nan_frac, background_ping_ranges
        Background windows (see :func:`~sidescanpy.export.windows.sample_background_windows`).
    val_fraction : float
        Share of targets (by id hash) and background ping blocks put in the
        validation split; 0 disables splitting.
    split_block_pings : int
        Background ping-block size for the split hash, so overlapping
        background windows share a split.
    layout : str
        One of :data:`LAYOUTS`.
    frequencies : list of str, optional
        Frequency keys (e.g. ``["hz_900000"]``) that target and background
        windows may come from. Default: every frequency in the files. Use it
        to keep a dead or unwanted channel out of the dataset; a target is
        otherwise matched to the frequency whose sample spacing is closest
        to its catalog's, which can be ambiguous when frequencies share a
        sample spacing.
    """

    styles: tuple[str, ...] = CHIP_STYLES
    render_level: str = "radiometric"
    size_px: int = 400
    swath_size_px: int = 512
    resolution_m: float | list[float] | None = None
    value_var: str = "radiometric"
    resample_method: str = "griddata"
    across_axis: str = "ground"
    swath_fill: bool = False
    box_size: float = 40.0
    box_size_m: float | None = None
    with_context: bool = False
    jitter_px: int = 0
    n_augment: int = 1
    random_flip: bool = False
    gamma: float = 1.0
    pct_low: float = 1.0
    pct_high: float = 99.0
    normalization_scope: str = "image"
    bit_depth: int = 8
    min_confidence: str = "verified"
    categories: list[str] | None = None
    max_match_dist_m: float = 25.0
    seed: int | None = None
    n_background: int = 0
    background_margin_m: float = 0.0
    background_max_nan_frac: float = 0.3
    background_ping_ranges: dict[str, list[tuple[int, int]]] | None = None
    val_fraction: float = 0.0
    split_block_pings: int = 500
    layout: str = "flat"
    frequencies: list[str] | None = None

    def __post_init__(self):
        self.styles = tuple(self.styles)
        self.n_augment = max(1, int(self.n_augment or 1))
        self.n_background = max(0, int(self.n_background or 0))
        self.background_margin_m = float(self.background_margin_m or 0.0)
        if isinstance(self.resolution_m, list | tuple):
            self.resolution_m = [float(r) for r in self.resolution_m]
            if len(self.resolution_m) == 1:
                self.resolution_m = self.resolution_m[0]
        if self.categories is not None:
            self.categories = list(self.categories)
        if self.frequencies is not None:
            self.frequencies = [str(f) for f in self.frequencies]
        if self.background_ping_ranges is not None:
            self.background_ping_ranges = {
                str(name): [(int(a), int(b)) for a, b in ranges] for name, ranges in self.background_ping_ranges.items()
            }
        _check_choice("styles", self.styles, CHIP_STYLES, many=True)
        _check_choice("render_level", self.render_level, RENDER_LEVELS)
        _check_choice("value_var", self.value_var, tuple(LEVEL_VARS))
        _check_choice("resample_method", self.resample_method, RESAMPLE_METHODS)
        _check_choice("across_axis", self.across_axis, ("ground", "slant"))
        _check_choice("normalization_scope", self.normalization_scope, NORMALIZATION_SCOPES)
        _check_choice("bit_depth", self.bit_depth, (8, 16))
        _check_choice("min_confidence", self.min_confidence, tuple(CONFIDENCE_RANK))
        _check_choice("layout", self.layout, LAYOUTS)
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetSpec:
        """Build from a mapping (e.g. parsed YAML); unknown keys are an error."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown DatasetSpec key(s): {unknown}")
        return cls(**dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetSpec:
        """Load from a YAML (or JSON) file holding one mapping."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-ready mapping that :meth:`from_dict` accepts."""
        out = dataclasses.asdict(self)
        out["styles"] = list(self.styles)
        if self.background_ping_ranges is not None:
            out["background_ping_ranges"] = {k: [list(r) for r in v] for k, v in self.background_ping_ranges.items()}
        return out

    def size_for(self, style: str) -> int:
        """Output size of ``style``."""
        return self.size_px if style == "chip" else self.swath_size_px

    @property
    def is_pixel(self) -> bool:
        """Whether the render level crops native sample/ping pixels."""
        return self.render_level in PIXEL_LEVELS

    @property
    def split_enabled(self) -> bool:
        """Whether a validation split is assigned."""
        return self.val_fraction > 0


def _check_choice(name: str, value, allowed: Sequence, *, many: bool = False) -> None:
    bad = [v for v in value if v not in allowed] if many else ([] if value in allowed else [value])
    if bad:
        raise ValueError(f"{name}: {bad} not in {list(allowed)}")


def _resolve_options(spec: DatasetSpec) -> DatasetSpec:
    """Drop options that don't apply to the render level, with a warning each."""
    changes = {}
    if spec.resolution_m is not None and spec.is_pixel:
        logger.warning(
            f"resolution_m={spec.resolution_m} has no effect under render level {spec.render_level!r} "
            "(native pixel crop) -- ignoring it",
        )
        changes["resolution_m"] = None
    if spec.value_var != "radiometric" and spec.is_pixel:
        logger.warning(
            f"value_var={spec.value_var} has no effect on {spec.render_level!r} images -- the render level already "
            "selects the values for pixel-space styles",
        )
    if spec.across_axis != "ground" and spec.render_level != "geometric":
        logger.warning(f"across_axis={spec.across_axis} only applies to render level geometric -- ignoring it")
        changes["across_axis"] = "ground"
    if spec.box_size_m is not None and spec.is_pixel:
        logger.warning("box_size_m only applies to geometric/georef render levels -- ignoring it, using box_size")
        changes["box_size_m"] = None
    return replace(spec, **changes) if changes else spec


# ============================================================
# Results
# ============================================================

@dataclass
class SkippedTarget:
    """A target (or one style of it) left out of the dataset, and why."""

    target_id: str
    reason: str
    style: str | None = None


@dataclass
class StyleResult:
    """What one style directory received."""

    style: str
    directory: Path
    images: list[Path] = field(default_factory=list)
    n_target_images: int = 0
    n_background_images: int = 0
    n_annotations: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetResult:
    """Summary of a :func:`build_target_dataset` run.

    Attributes
    ----------
    out_dir : Path
    styles : dict
        ``{relative style directory: StyleResult}`` (``"chip"``, or
        ``"res_0.05m/chip"`` with several resolutions). Styles that
        produced no image are absent.
    skipped : list of SkippedTarget
    n_targets : int
        Targets at or above ``min_confidence``.
    n_matched : int
        Of those, correlated to a ping.
    manifest_path : Path or None
    """

    out_dir: Path
    styles: dict[str, StyleResult] = field(default_factory=dict)
    skipped: list[SkippedTarget] = field(default_factory=list)
    n_targets: int = 0
    n_matched: int = 0
    manifest_path: Path | None = None

    @property
    def n_images(self) -> int:
        """Images written across every style."""
        return sum(len(s.images) for s in self.styles.values())


# ============================================================
# Per-style output
# ============================================================

def _split_for(key: str, val_fraction: float) -> str:
    """Deterministic ``"train"``/``"val"`` from a hash of ``key``."""
    if val_fraction <= 0:
        return "train"
    unit = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2.0**64
    return "val" if unit < val_fraction else "train"


class _StyleWriter:
    """Accumulates and writes one style's images, annotations and logs."""

    def __init__(self, style: str, style_dir: Path, spec: DatasetSpec):
        self.style = style
        self.dir = style_dir
        self.spec = spec
        self.stream = spec.normalization_scope == "image"
        self.pending: list[tuple] = []
        self.entries: list[tuple[CocoImageEntry, str]] = []
        self.thumbnails: list[np.ndarray] = []
        self.captions: list[str] = []
        self.sheet_boxes: list[list[Target]] = []
        self.aug_log: list[dict] = []
        self.result = StyleResult(style=style, directory=style_dir)

    def add(self, window, targets, label, freq_key, file_name, aug_idx, split, log_entry, *, background=False) -> None:
        """Queue (shared scope) or write (image scope) one image."""
        if self.spec.split_enabled:
            log_entry = {**log_entry, "split": split}
        self.aug_log.append(log_entry)
        item = (window, targets, label, freq_key, file_name, aug_idx, split, background)
        if self.stream:
            self._emit(*item)
        else:
            self.pending.append(item)

    def _image_path(self, file_name: str, split: str) -> Path:
        if self.spec.layout == "ultralytics":
            d = self.dir / "images" / split
            d.mkdir(parents=True, exist_ok=True)
            return d / file_name
        return self.dir / file_name

    def _emit(self, window, targets, label, freq_key, file_name, aug_idx, split, background, reference_stats=None) -> None:
        spec = self.spec
        img_8bit, _meta = to_uint8(window, gamma=spec.gamma, pct_low=spec.pct_low, pct_high=spec.pct_high, reference_stats=reference_stats)
        path = self._image_path(file_name, split)
        if spec.bit_depth == 16:
            # The contact sheet still uses the 8-bit render.
            img_16bit, _meta = to_uint16(window, gamma=spec.gamma, pct_low=spec.pct_low, pct_high=spec.pct_high, reference_stats=reference_stats)
            save_grayscale_png(img_16bit, path)
        else:
            save_grayscale_png(img_8bit, path)
        height, width = img_8bit.shape
        for co in targets:
            self.entries.append((CocoImageEntry(target=co, file_name=file_name, width=width, height=height,
                                                bbox_size=co.extra_attributes.get("bbox_size_px")), split))
        if not targets:
            self.entries.append((CocoImageEntry(target=None, file_name=file_name, width=width, height=height), split))
        if aug_idx == 0:  # contact sheet: first draw only
            self.thumbnails.append(img_8bit)
            self.sheet_boxes.append(targets)
            n_co = len(targets)
            self.captions.append(f"{label} ({freq_key}, {n_co} target{'s' if n_co != 1 else ''})")
        self.result.images.append(path)
        self.result.split_counts[split] = self.result.split_counts.get(split, 0) + 1
        if background:
            self.result.n_background_images += 1
        else:
            self.result.n_target_images += 1

    def finish(self, contact_sheet: ContactSheetFn | None) -> StyleResult | None:
        """Write everything still pending plus annotations; ``None`` if no image was produced."""
        spec = self.spec
        if self.pending:
            reference_stats = reference_stats_for_scope(spec.normalization_scope, [p[0] for p in self.pending], spec.pct_low, spec.pct_high)
            for item in self.pending:
                self._emit(*item, reference_stats=reference_stats)
            self.pending = []
        if not self.result.images:
            logger.warning(f"{self.style}: no windows produced at render level {spec.render_level!r} -- skipping its COCO/YOLO export")
            return None

        box = (spec.box_size, spec.box_size)
        if contact_sheet is not None:
            suptitle = f"{self.style}: {len(self.thumbnails)} chips" + (
                f" (first of {spec.n_augment} draws each)" if spec.n_augment > 1 else ""
            )
            contact_sheet(self.thumbnails, self.captions, boxes=self.sheet_boxes, box_size=box, suptitle=suptitle,
                          path=self.dir / "contact_sheet.png")
        if spec.n_augment > 1 or spec.random_flip or spec.n_background:
            (self.dir / "augmentations.json").write_text(json.dumps(self.aug_log, indent=1))

        # Writers apply their own min_confidence filter; pass the spec's so
        # catalogs whose targets are never "verified" still get annotations.
        writer_kwargs = dict(box_size=box, min_confidence=spec.min_confidence, categories=spec.categories)
        if spec.layout == "flat":
            entries = [e for e, _split in self.entries]
            coco = write_coco(entries, self.dir / "targets.coco.json", **writer_kwargs)
            logger.debug(f"{self.style}: COCO export: {len(coco['images'])} image(s), {len(coco['annotations'])} annotation(s)")
            yolo = write_yolo(entries, self.dir / "targets.yolo", **writer_kwargs)
            logger.debug(f"{self.style}: YOLO export: {len(yolo['images'])} image(s) -> {self.dir / 'targets.yolo'}")
            self.result.n_annotations = len(coco["annotations"])
            if spec.split_enabled:
                self._write_split_lists()
        else:
            self._write_ultralytics(writer_kwargs)
        logger.debug(f"{self.style}: wrote {len(self.result.images)} image(s) to {self.dir}")
        return self.result

    def _write_split_lists(self) -> None:
        split_dir = self.dir / "splits"
        split_dir.mkdir(exist_ok=True)
        by_split: dict[str, list[str]] = defaultdict(list)
        for entry, split in self.entries:
            if entry.file_name not in by_split[split]:
                by_split[split].append(entry.file_name)
        for split in ("train", "val"):
            (split_dir / f"{split}.txt").write_text("".join(f"{name}\n" for name in by_split.get(split, [])))

    def _write_ultralytics(self, writer_kwargs: dict) -> None:
        names = None
        for split in ("train", "val"):
            entries = [e for e, s in self.entries if s == split]
            if not entries:
                continue
            coco = write_coco(entries, self.dir / f"{split}.coco.json", **writer_kwargs)
            self.result.n_annotations += len(coco["annotations"])
            yolo = write_yolo(entries, self.dir, labels_dir=self.dir / "labels" / split, data_yaml=False, **writer_kwargs)
            names = yolo["names"]
        val = "images/val" if (self.dir / "images" / "val").is_dir() else "images/train"
        names_block = "".join(f"  {i}: {n}\n" for i, n in enumerate(names or ["target"]))
        (self.dir / "data.yaml").write_text(
            f"path: {self.dir.resolve()}\ntrain: images/train\nval: {val}\nnames:\n{names_block}nc: {len(names or ['target'])}\n"
        )
        logger.debug(f"{self.style}: Ultralytics dataset -> {self.dir / 'data.yaml'}")


# ============================================================
# Target windows
# ============================================================

def _select_targets(targets: Sequence[Target], spec: DatasetSpec) -> tuple[list[Target], list[SkippedTarget]]:
    """Confidence filter, plus the categories check (before any rendering)."""
    min_rank = CONFIDENCE_RANK[spec.min_confidence]
    selected, skipped = [], []
    for t in targets:
        if CONFIDENCE_RANK[t.position_confidence] >= min_rank:
            selected.append(t)
        else:
            skipped.append(SkippedTarget(t.target_id, f"below min_confidence={spec.min_confidence}"))
    if spec.categories is not None:
        unknown = sorted({t.category for t in selected if t.category is not None} - set(spec.categories))
        if unknown:
            raise ValueError(f"target categories {unknown} are not in categories {spec.categories}")
    return selected, skipped


def _correlate(targets, index, max_match_dist_m) -> tuple[list[tuple], list[SkippedTarget]]:
    matched, skipped = [], []
    for t in targets:
        m = correlate_target(t, index, max_match_dist_m)
        if m is None:
            logger.warning(f"target {t.target_id}: no ping within {max_match_dist_m}m of its recorded position, skipping")
            skipped.append(SkippedTarget(t.target_id, f"no ping within {max_match_dist_m} m"))
            continue
        logger.debug(f"target {t.target_id} -> {m.path.name} {m.freq_key} ping {m.ping_idx} ({m.dist_m:.1f}m)")
        matched.append((t, m.path, m.freq_key, m.ping_idx))
    return matched, skipped


def _group(matched) -> dict[tuple, dict[int, list]]:
    """``{(path, freq_key): {channel: [(target, ping)]}}``; dual-swath needs both channels together."""
    groups: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for t, path, freq_key, ping_idx in matched:
        channel_idx = t.channel if t.channel is not None else 0
        groups[(path, freq_key)][channel_idx].append((t, ping_idx))
    return groups


@dataclass
class _Run:
    """Shared state of one dataset build at one resolution."""

    spec: DatasetSpec
    out_dir: Path
    writers: dict[str, _StyleWriter]
    rng: np.random.Generator
    result: DatasetResult
    context: ContextFn | None
    skipped_styles: set = field(default_factory=set)

    @property
    def geo(self) -> bool:
        return not self.spec.is_pixel

    def skip(self, target_id: str, style: str, reason: str) -> None:
        if (target_id, style) not in self.skipped_styles:
            self.skipped_styles.add((target_id, style))
            self.result.skipped.append(SkippedTarget(target_id, reason, style))

    def save_context(self, image, target, coordinate_system, resolution_m=None) -> None:
        if self.context is None:
            return
        self.context(image, target, coordinate_system=coordinate_system, resolution_m=resolution_m,
                     box_size=(self.spec.box_size, self.spec.box_size),
                     path=self.out_dir / "context" / f"context_{target.target_id}.png")


def _geo_window(run: _Run, view, channel_idx: int, style: str, size_px: int, jitter_px: int, orig_bbox: dict):
    """One geometric/georef window from a loaded neighborhood, with original bboxes restored."""
    spec = run.spec
    rendered = extract_geo_style_window(
        view.ds, center_sample=view.target.sample, channel_idx=channel_idx, local_ping=view.local_ping, style=style,
        level=spec.render_level, size_px=size_px, jitter_px=jitter_px, rng=run.rng, mps=view.mps,
        items_by_channel=view.items, lo=view.lo, primary=view.target, resolution_m_override=spec.resolution_m,
        backscatter_var=LEVEL_VARS[spec.value_var], resample_method=spec.resample_method,
        swath_fill=spec.swath_fill and spec.render_level == "geometric",
    )
    if rendered is None:
        return None
    return rendered.image, [replace(ct, bbox=orig_bbox.get(ct.target_id, ct.bbox)) for ct in rendered.targets]


def _load_geo_views(run: _Run, ds_full, t, ping_idx, items_by_channel, swath_geo_styles) -> dict:
    spec = run.spec
    common = dict(
        jitter_px=spec.jitter_px, resolution_m=spec.resolution_m, with_db="db" in (spec.render_level, spec.value_var),
        across_axis=spec.across_axis, per_sample_latlon=spec.render_level == "georef", swath_fill=spec.swath_fill,
    )
    views = {}
    if "chip" in spec.styles:
        views["chip"] = load_geo_neighborhood(ds_full, t, ping_idx, items_by_channel, ["chip"], size_px=spec.size_px, **common)
    if swath_geo_styles:
        views["swath"] = load_geo_neighborhood(
            ds_full, t, ping_idx, items_by_channel, swath_geo_styles, size_px=spec.swath_size_px, **common,
        )
    return views


def _pixel_context(run: _Run, masked: xr.Dataset, t: Target, channel_idx: int, local_ping: int) -> None:
    """Full loaded neighborhood, both channels combined, box at the target's true position."""
    var = LEVEL_VARS[run.spec.value_var]
    combined = combine_channels_mirrored(masked[var].isel(channel=0).values, masked[var].isel(channel=1).values)
    col = target_combined_col(t.sample, masked.sizes["sample"], channel_idx)
    context_target = replace(t, sample=col, extra_attributes={**t.extra_attributes, "window_row": local_ping})
    context_target = with_bbox_size_px(context_target, scale_x=1.0, scale_y=1.0)
    combined, (context_target,) = orient_pixel_window(combined, [context_target], channel_idx, "dual-swath")
    run.save_context(combined, context_target, "pixel")


def _with_box_size_m(targets: list[Target], box_size_m: float) -> list[Target]:
    """Meters placeholder box for center-only picks (same real extent at every resolution)."""
    return [
        ct if "bbox_size_px" in ct.extra_attributes or not ct.extra_attributes.get("window_resolution_m")
        else replace(ct, extra_attributes={**ct.extra_attributes, "bbox_size_px": (box_size_m / ct.extra_attributes["window_resolution_m"],) * 2})
        for ct in targets
    ]


def _render_group(run: _Run, path: Path, freq_key: str, items_by_channel: dict) -> None:
    """Every target window of one file/frequency."""
    spec = run.spec
    ds_full = read_survey(path, chunks={"ping": 512}).get_frequency(freq_key).dataset
    n_pings, n_samples_single = ds_full.sizes["ping"], ds_full.sizes["sample"]
    placed_by_channel, placed_combined = place_targets_in_file(items_by_channel, n_samples_single)

    # Geometric/georef: one neighborhood for chip, one shared by the swath
    # styles (and the geometric context view, a dual-swath extraction).
    swath_geo_styles = [s for s in spec.styles if s != "chip"] if run.geo else []
    if run.geo and spec.with_context:
        swath_geo_styles = list(dict.fromkeys(swath_geo_styles + ["dual-swath"]))
    native_spans = {"chip": spec.size_px, "single-swath": n_samples_single, "dual-swath": 2 * n_samples_single}
    max_native_span = 0 if run.geo else max([0] + [native_spans[s] for s in spec.styles])
    half_span = max_native_span // 2 + spec.jitter_px + 50
    orig_bbox = {tt.target_id: tt.bbox for its in items_by_channel.values() for tt, _ in its}

    for channel_idx, its in items_by_channel.items():
        for t, ping_idx in its:
            masked = views = None
            lo = local_ping = 0
            if not run.geo:
                # A bounded neighborhood keeps memory independent of file size.
                lo, hi = max(0, ping_idx - half_span), min(n_pings, ping_idx + half_span)
                masked = mask_nadir_gap_with_uncorrected(ds_full.isel(ping=slice(lo, hi)).load())
                if "db" in (spec.render_level, spec.value_var):
                    masked = with_db_var(masked)
                local_ping = ping_idx - lo
            else:
                try:
                    views = _load_geo_views(run, ds_full, t, ping_idx, items_by_channel, swath_geo_styles)
                except KeyError as exc:
                    logger.warning(f"target {t.target_id}: {path.name} has no {exc} (run geometric correction) -- skipping")
                    run.result.skipped.append(SkippedTarget(t.target_id, f"missing {exc}"))
                    continue
            _render_target(run, t, path, freq_key, channel_idx, ping_idx, masked, views, lo, local_ping,
                           placed_by_channel, placed_combined, orig_bbox)


def _render_target(run, t, path, freq_key, channel_idx, ping_idx, masked, views, lo, local_ping,
                   placed_by_channel, placed_combined, orig_bbox) -> None:
    spec = run.spec

    def geo_window(style, size_px, jitter_px):
        return _geo_window(run, views["chip" if style == "chip" else "swath"], channel_idx, style, size_px, jitter_px, orig_bbox)

    if spec.with_context:
        if not run.geo:
            _pixel_context(run, masked, t, channel_idx, local_ping)
        elif "dual-swath" not in spec.styles:
            # Unjittered dual-swath view; when dual-swath is exported its
            # first draw is reused below instead of a second resample.
            context_result = geo_window("dual-swath", spec.swath_size_px, 0)
            if context_result is None:
                logger.warning(f"target {t.target_id}: context/{spec.render_level} window unavailable, skipping context")
            else:
                grid, context_targets = context_result
                run.save_context(grid, context_targets[0], spec.render_level, context_targets[0].extra_attributes.get("window_resolution_m"))

    split = _split_for(f"target:{t.target_id}", spec.val_fraction)
    for aug_idx in range(spec.n_augment):
        for style in spec.styles:
            size_px = spec.size_for(style)
            if not run.geo:
                rendered = extract_pixel_style_window(
                    masked, center_sample=t.sample, channel_idx=channel_idx, local_ping=local_ping, ping_offset=lo,
                    style=style, level=spec.render_level, size_px=size_px, jitter_px=spec.jitter_px, rng=run.rng,
                    placed_by_channel=placed_by_channel, placed_combined=placed_combined,
                )
                result = None if rendered is None else (rendered.image, rendered.targets)
            else:
                result = geo_window(style, size_px, spec.jitter_px)
            if result is None:
                logger.warning(f"target {t.target_id}: {style}/{spec.render_level} window unavailable, skipping")
                run.skip(t.target_id, style, "window unavailable")
                continue
            window, co_targets = result
            if style == "dual-swath" and run.geo and spec.with_context and aug_idx == 0:
                primary = next((ct for ct in co_targets if ct.target_id == t.target_id), co_targets[0])
                run.save_context(window, primary, spec.render_level, primary.extra_attributes.get("window_resolution_m"))
            if spec.box_size_m is not None:
                co_targets = _with_box_size_m(co_targets, spec.box_size_m)
            flip_h = flip_v = False
            if spec.random_flip:
                window, co_targets, flip_h, flip_v = random_flip(window, co_targets, run.rng, edge_coords=run.geo)
            file_name = augmented_file_name(f"target_{t.target_id}", aug_idx, spec.n_augment)
            primary = next((ct for ct in co_targets if ct.target_id == t.target_id), None)
            log_entry = {
                "file_name": file_name, "target_id": t.target_id, "augmentation": aug_idx,
                "source_file": path.name, "freq_key": freq_key, "flip_h": flip_h, "flip_v": flip_v,
                "target_row": None if primary is None else round(float(primary.extra_attributes["window_row"]), 2),
                "target_col": None if primary is None else round(float(primary.sample), 2),
                "resolution_m": None if primary is None else primary.extra_attributes.get("window_resolution_m"),
                "n_targets": len(co_targets),
            }
            run.writers[style].add(window, co_targets, t.target_id, freq_key, file_name, aug_idx, split, log_entry)


# ============================================================
# Background windows
# ============================================================

def _render_backgrounds(run: _Run, index: dict, catalog: Sequence[Target], freq_keys: list[str]) -> None:
    """Screened no-target windows, from their own generator so targets are unaffected."""
    spec = run.spec
    bg_rng = np.random.default_rng(None if spec.seed is None else [spec.seed, 1])
    for style in spec.styles:
        window_spec = WindowSpec(
            style=style, level=spec.render_level, size_px=spec.size_for(style), jitter_px=spec.jitter_px,
            resolution_m=spec.resolution_m, value_var=spec.value_var, resample_method=spec.resample_method,
            across_axis=spec.across_axis, swath_fill=spec.swath_fill,
        )
        windows = sample_background_windows(
            index, catalog, window_spec, spec.n_background, bg_rng, freq_keys=freq_keys,
            ping_ranges=spec.background_ping_ranges, margin_m=spec.background_margin_m,
            max_nan_frac=spec.background_max_nan_frac, max_match_dist_m=spec.max_match_dist_m, n_draws=spec.n_augment,
        )
        for bg in windows:
            window, flip_h, flip_v = bg.window.image, False, False
            if spec.random_flip:
                window, _no_targets, flip_h, flip_v = random_flip(window, [], bg_rng, edge_coords=run.geo)
            file_name = augmented_file_name(f"bg_{bg.index}", bg.draw, spec.n_augment)
            cand = bg.candidate
            log_entry = {
                "file_name": file_name, "target_id": None, "background": bg.index, "augmentation": bg.draw,
                "source_file": cand.path.name, "freq_key": cand.freq_key, "ping": cand.ping, "channel": cand.channel,
                "center_sample": cand.sample, "across_center_m": bg.across_center_m, "flip_h": flip_h, "flip_v": flip_v,
                "target_row": None, "target_col": None, "resolution_m": bg.window.resolution_m, "n_targets": 0,
                "nan_fraction": round(bg.nan_fraction, 4),
            }
            split = _split_for(f"bg:{cand.path.name}:{cand.freq_key}:{cand.ping // spec.split_block_pings}", spec.val_fraction)
            run.writers[style].add(window, [], f"bg_{bg.index}", cand.freq_key, file_name, bg.draw, split, log_entry, background=True)
        logger.debug(f"{style}: {len(windows)} background image(s) from {len({bg.index for bg in windows})} window(s)")


# ============================================================
# Entry point
# ============================================================

def _build_one(index, catalog, matched, spec: DatasetSpec, out_dir: Path, prefix: str, result: DatasetResult,
               contact_sheet: ContactSheetFn | None, context: ContextFn | None) -> None:
    """One dataset at one resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for style in spec.styles:
        (out_dir / style).mkdir(parents=True, exist_ok=True)
    if spec.with_context:
        (out_dir / "context").mkdir(parents=True, exist_ok=True)
        if context is None:
            logger.warning("with_context is set but no context callable was given -- context views not written")
    writers = {style: _StyleWriter(style, out_dir / style, spec) for style in spec.styles}
    run = _Run(spec=spec, out_dir=out_dir, writers=writers, rng=np.random.default_rng(spec.seed), result=result, context=context)

    groups = _group(matched)
    for (path, freq_key), items_by_channel in groups.items():
        _render_group(run, path, freq_key, items_by_channel)
    if spec.n_background:
        _render_backgrounds(run, index, catalog, sorted({fk for _p, fk in groups}))
    for style, writer in writers.items():
        style_result = writer.finish(contact_sheet)
        if style_result is not None:
            result.styles[prefix + style] = style_result


def build_target_dataset(
    sources: str | Path | Sequence[str | Path],
    targets: Sequence[Target],
    spec: DatasetSpec,
    out_dir: str | Path,
    *,
    contact_sheet: ContactSheetFn | None = None,
    context: ContextFn | None = None,
) -> DatasetResult:
    """Render a COCO + YOLO dataset of target (and background) images.

    Parameters
    ----------
    sources : path or sequence of paths
        Directory of corrected ``.ss.h5`` files, or the files themselves.
    targets : sequence of Target
        The whole catalog, any confidence: only targets at or above
        ``spec.min_confidence`` are rendered and annotated, but background
        windows avoid all of them.
    spec : DatasetSpec
    out_dir : path
    contact_sheet : callable, optional
        Called once per style as ``contact_sheet(thumbnails, captions, *,
        boxes, box_size, suptitle, path)`` to draw a contact sheet (e.g.
        with :func:`sidescanpy.viz.contact_sheet.contact_sheet`).
    context : callable, optional
        Called per target as ``context(image, target, *, coordinate_system,
        resolution_m, box_size, path)`` when ``spec.with_context``.

    Returns
    -------
    DatasetResult
        Nothing is rendered when no target correlates (``n_matched == 0``).

    Raises
    ------
    ValueError
        A target's category is missing from ``spec.categories``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = list(targets)
    spec = _resolve_options(spec)
    selected, skipped = _select_targets(catalog, spec)
    logger.debug(f"{len(selected)} target(s) at or above min_confidence={spec.min_confidence!r}")
    result = DatasetResult(out_dir=out_dir, skipped=skipped, n_targets=len(selected))

    index = build_position_index(sources)
    if spec.frequencies is not None:
        index = {p: {fk: c for fk, c in ch.items() if fk in spec.frequencies} for p, ch in index.items()}
        index = {p: ch for p, ch in index.items() if ch}
        logger.debug(f"restricted to frequencies {spec.frequencies}: {len(index)} file(s) left")
    matched, unmatched = _correlate(selected, index, spec.max_match_dist_m)
    result.skipped += unmatched
    result.n_matched = len(matched)
    if not matched:
        logger.warning("no target correlated to any ping -- nothing to export")
        return result

    if isinstance(spec.resolution_m, list):
        for res in spec.resolution_m:
            prefix = f"res_{res:g}m/"
            _build_one(index, catalog, matched, replace(spec, resolution_m=res), out_dir / prefix, prefix, result, contact_sheet, context)
    else:
        _build_one(index, catalog, matched, spec, out_dir, "", result, contact_sheet, context)

    result.manifest_path = _write_manifest(result, spec)
    return result


def _write_manifest(result: DatasetResult, spec: DatasetSpec) -> Path:
    manifest = {
        "sidescanpy_version": __version__,
        "spec": spec.to_dict(),
        "n_targets": result.n_targets,
        "n_matched": result.n_matched,
        "outputs": {
            key: {
                "images": len(s.images), "target_images": s.n_target_images, "background_images": s.n_background_images,
                "annotations": s.n_annotations, "splits": s.split_counts,
            }
            for key, s in result.styles.items()
        },
        "skipped": [dataclasses.asdict(s) for s in result.skipped],
    }
    path = result.out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
