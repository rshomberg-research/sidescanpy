"""
Target dataclass and pixel-box helpers: the format-agnostic sonar contact model.

A target's position may be known in the native address space (ping/
channel/sample, lat/lon), the pixel address space (image id + bbox), both,
or neither, depending on what the source catalog actually gives up.

Pure Python (no I/O) so that catalog readers in ``io`` can build
:class:`Target` without importing a higher layer.

2026 rshomberg@uri.edu
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

PositionConfidence = Literal["unknown", "approximate", "verified"]


@dataclass
class Target:
    """One sonar target/contact, in native and/or pixel address space.

    Parameters
    ----------
    target_id : str
        Identifier from the source catalog (e.g. a filename-derived
        number). Not assumed globally unique across catalogs — see
        ``catalog_source`` for provenance.
    source_format : str
        e.g. ``"jsf_discover"``, ``"sdf_sonarwiz"``.
    catalog_source : str
        Path to the original catalog file this target came from.
    pick_timestamp : float, optional
        Unix epoch seconds, when trustworthy.
    ping, channel, sample : int, optional
        Native (ping/sample) address space. ``channel``: 0=port, 1=starboard.
    latitude, longitude : float, optional
        Degrees, WGS84.
    image_id : str, optional
        Pixel address space: identifies the rendered image this target
        was localized on (populated once a crop/render exists).
    bbox : tuple[float, float, float, float], optional
        ``(x, y, w, h)`` in ``image_id``'s pixel space.
    position_confidence : {"unknown", "approximate", "verified"}
        Whether the position (native and/or pixel) has been visually
        validated. Export code should default to filtering on this
        rather than silently shipping unlocalized or unverified boxes.
    label : str, optional
        Free text, passed through, not interpreted.
    category : str, optional
        Class name for multi-class annotation export (COCO/YOLO). ``None``
        (the default) means "uncategorized": the export writers then use
        their single default category name.
    extra_attributes : dict
        Format-specific fields that don't generalize (e.g. JSF's
        MutableTag text, or notes about data-quality workarounds
        applied while reading this target).
    """

    target_id: str
    source_format: str
    catalog_source: str
    pick_timestamp: float | None = None

    ping: int | None = None
    channel: int | None = None
    sample: int | None = None
    latitude: float | None = None
    longitude: float | None = None

    image_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    position_confidence: PositionConfidence = "unknown"
    label: str | None = None
    category: str | None = None
    extra_attributes: dict[str, Any] = field(default_factory=dict)


CONFIDENCE_RANK = {"unknown": 0, "approximate": 1, "verified": 2}


def clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[tuple[float, float, float, float], bool]:
    """Clip a ``(x, y, w, h)`` pixel box to an image's bounds.

    Shared by ``export/ml/coco.py`` and ``export/ml/yolo.py`` so both
    formats agree on exactly the same clipped box for a given target.

    Returns
    -------
    (clipped_bbox, was_clipped)
    """
    x, y, w, h = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    clipped = (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))
    return clipped, clipped != bbox


def bbox_from_center(row: float, col: float, box_size: tuple[float, float]) -> tuple[float, float, float, float]:
    """Build a ``(x, y, w, h)`` pixel box of ``box_size`` (width, height)
    centered on ``(col, row)``.

    Shared by box-drawing QA visualization and COCO export so both agree
    on exactly the same box for a given target. ``box_size`` is either a
    catalog's real measured extent already converted to this window's
    pixels (``extra_attributes["bbox_size_px"]``, set by ``sidescan-export
    chip`` when the target carries a ``bbox``) or, for center-only
    catalogs (JSF/Discover picks), a caller-supplied placeholder.
    """
    w, h = box_size
    return (col - w / 2, row - h / 2, w, h)
