"""
Invertible 8-bit display transform.

Percentile-stretch + gamma-correct a backscatter array (any dtype) into
an 8-bit grayscale image suitable for training-dataset export, while
retaining enough metadata to reconstruct the original array later
(lossy — 8-bit quantization is not exactly reversible, but the scale is
recoverable). This is a **display/export transform**, not a physical
correction — it has no relationship to ``processing/radiometric.py``'s
TVG correction. Run TVG (and geometric/georef, if wanted) first; this
module only turns whatever array it's handed into 8-bit.

Normalization scope (per-image vs. survey-wide vs. dataset-wide) is
deliberately a caller decision, not baked in here: pass
``reference_stats=None`` (default) for per-image scope, where each
array is stretched against its own percentiles, or precompute one
:class:`ReferenceStats` via :func:`compute_reference_stats` over every
window a wider scope should share and pass it to every :func:`to_uint8`
call in that scope.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ArrayKind = Literal["uint16", "uint32", "float"]

_DEFAULT_PCT_LOW = 1.0
_DEFAULT_PCT_HIGH = 99.0


@dataclass(frozen=True)
class ReversalMetadata:
    """Enough to invert :func:`to_uint8` back to the original array's
    dtype and approximate scale."""

    kind: ArrayKind
    p_min: float
    p_max: float
    gamma: float
    original_max: float


@dataclass(frozen=True)
class ReferenceStats:
    """Percentile bounds computed once, for reuse across many
    :func:`to_uint8` calls at survey- or dataset-wide normalization
    scope, instead of each image normalizing against only its own
    window."""

    p_min: float
    p_max: float


def compute_reference_stats(
    *arrays: np.ndarray, pct_low: float = _DEFAULT_PCT_LOW, pct_high: float = _DEFAULT_PCT_HIGH
) -> ReferenceStats:
    """Compute shared percentile bounds over one or more arrays.

    Call once over every window a survey-wide or dataset-wide
    normalization scope should share, then pass the result to every
    :func:`to_uint8` call in that scope via ``reference_stats=``. Omit
    entirely for the default per-image scope.

    Non-finite values are excluded before the percentile is computed.
    Mixing arrays of different kinds (e.g. raw counts and TVG-corrected
    dB) is the caller's decision, not validated here — only combine
    arrays it actually makes sense to normalize together.
    """
    finite_values = [a[np.isfinite(a)] for a in arrays]
    finite_values = [v for v in finite_values if v.size > 0]
    if not finite_values:
        return ReferenceStats(p_min=0.0, p_max=1.0)
    pooled = np.concatenate([v.ravel() for v in finite_values])
    p_min, p_max = np.percentile(pooled, [pct_low, pct_high])
    return ReferenceStats(p_min=float(p_min), p_max=float(p_max))


def _kind_for(array: np.ndarray) -> ArrayKind:
    if array.dtype == np.uint32:
        return "uint32"
    if array.dtype == np.uint16:
        return "uint16"
    if array.dtype in (np.float32, np.float64):
        return "float"
    raise ValueError(f"Unsupported dtype for 8-bit export: {array.dtype}")


def to_uint8(
    array: np.ndarray,
    *,
    gamma: float = 1.0,
    pct_low: float = _DEFAULT_PCT_LOW,
    pct_high: float = _DEFAULT_PCT_HIGH,
    reference_stats: ReferenceStats | None = None,
) -> tuple[np.ndarray, ReversalMetadata]:
    """Percentile-stretch + gamma-correct ``array`` into an 8-bit
    grayscale image.

    Parameters
    ----------
    array : np.ndarray
        ``uint16``, ``uint32``, or ``float32``/``float64``. ``uint32``
        is stretched over non-zero values only (zero treated as
        background/no-return, matching raw sensor-counts convention);
        the other two kinds are stretched over all finite values.
    gamma : float, default 1.0
        Power-law display contrast (``normalized ** gamma``) — matches
        this library's existing display-gamma convention
        (``viz.waterfall.plot_waterfall``/``viz.gridded``, both of which
        hand ``gamma`` straight to ``matplotlib.colors.PowerNorm``,
        whose own convention is ``normalized ** gamma``), where 1.0 is a
        no-op. Since ``gamma > 1`` on ``normalized`` in ``[0, 1]``
        pushes values *down* (darkens midtones/background, compresses
        toward black), a caller reusing a gamma value tuned against a
        ``PowerNorm``-based render (e.g. 2.5) gets the same visual
        direction here — the inverse convention (``normalized **
        (1/gamma)``, which brightens instead) was this function's
        original, undocumented-as-different implementation and produced
        washed-out/over-bright output at the same nominal gamma values
        that darkened contrast correctly elsewhere in this library; see
        CHANGELOG.md.
    pct_low, pct_high : float, default (1.0, 99.0)
        Percentile bounds for the stretch, matching this library's
        existing default elsewhere. Ignored when ``reference_stats`` is
        given.
    reference_stats : ReferenceStats, optional
        Precomputed bounds (survey-/dataset-wide normalization scope,
        see :func:`compute_reference_stats`). When given, ``pct_low``/
        ``pct_high`` are ignored — the array is stretched against
        ``reference_stats``'s bounds instead of its own. Omit for the
        default per-image scope.

    Returns
    -------
    (uint8_array, ReversalMetadata)
    """
    kind = _kind_for(array)

    if reference_stats is not None:
        p_min, p_max = reference_stats.p_min, reference_stats.p_max
        finite = array[np.isfinite(array)]
        original_max = float(finite.max()) if finite.size else 0.0
    elif kind == "uint32":
        non_zero = array[array > 0]
        if non_zero.size > 0:
            p_min, p_max = (float(v) for v in np.percentile(non_zero, [pct_low, pct_high]))
        else:
            p_min, p_max = 0.0, float(array.max()) if array.size else 1.0
        original_max = float(array.max()) if array.size else 0.0
    else:
        finite = array[np.isfinite(array)]
        if finite.size > 0:
            p_min, p_max = (float(v) for v in np.percentile(finite, [pct_low, pct_high]))
            original_max = float(finite.max())
        else:
            p_min, p_max = 0.0, 1.0
            original_max = 0.0

    normalized = (np.nan_to_num(array, nan=p_min).astype(np.float64) - p_min) / (p_max - p_min + 1e-8)
    normalized = np.clip(normalized, 0.0, 1.0)
    gamma_corrected = np.power(normalized, gamma) * 255.0
    img_8bit = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    metadata = ReversalMetadata(kind=kind, p_min=p_min, p_max=p_max, gamma=gamma, original_max=original_max)
    return img_8bit, metadata


def to_uint16(
    array: np.ndarray,
    *,
    gamma: float = 1.0,
    pct_low: float = _DEFAULT_PCT_LOW,
    pct_high: float = _DEFAULT_PCT_HIGH,
    reference_stats: ReferenceStats | None = None,
) -> tuple[np.ndarray, ReversalMetadata]:
    """16-bit counterpart of :func:`to_uint8` — identical percentile
    stretch and gamma (same bounds, same :class:`ReversalMetadata`), just
    quantized to 65536 levels instead of 256. For datasets where a model
    or analyst should see finer backscatter gradations than 8-bit keeps;
    ``to_uint16(a)[0] >> 8`` equals ``to_uint8(a)[0]`` to within one
    level of rounding."""
    img_8bit, metadata = to_uint8(array, gamma=gamma, pct_low=pct_low, pct_high=pct_high, reference_stats=reference_stats)
    normalized = (np.nan_to_num(array, nan=metadata.p_min).astype(np.float64) - metadata.p_min) / (metadata.p_max - metadata.p_min + 1e-8)
    normalized = np.clip(normalized, 0.0, 1.0)
    img_16bit = np.clip(np.power(normalized, gamma) * 65535.0, 0, 65535).astype(np.uint16)
    return img_16bit, metadata


def from_uint8(img_8bit: np.ndarray, metadata: ReversalMetadata) -> np.ndarray:
    """Reverse :func:`to_uint8` (or, given a ``uint16`` image,
    :func:`to_uint16`) back to (an approximation of) the original
    array's dtype and scale. Lossy — quantization is not exactly
    reversible."""
    full_scale = 65535.0 if img_8bit.dtype == np.uint16 else 255.0
    normalized = img_8bit.astype(np.float64) / full_scale
    reversed_data = (
        np.power(normalized, 1.0 / metadata.gamma) * (metadata.p_max - metadata.p_min) + metadata.p_min
    )

    if metadata.kind == "uint32":
        return np.clip(reversed_data, 0, 4294967295).astype(np.uint32)
    if metadata.kind == "uint16":
        return np.clip(reversed_data, 0, 65535).astype(np.uint16)
    return reversed_data.astype(np.float32)
