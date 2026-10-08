"""
Shared colormap resolution so every real sidescanpy plot renders NaN the
same way.

2026-09-28: real user report (target-crop verification artifact) that
NaN cells rendered white, not because anything filled them white, but
because matplotlib's default NaN handling is transparent, and every
export path checked composites that transparency over a white
background. User's explicit instruction: black-for-NaN is the standard
across all of sidescanpy's real outputs, not just that one demo. Fixed
at the one place each plotting function resolves its `cmap` argument to
a `Colormap` object, applied by default -- not opt-in -- rather than
duplicated per call site.
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

logger = logging.getLogger(__name__)

NAN_COLOR = "black"


def resolve_cmap(cmap: str | Colormap) -> Colormap:
    """Return a Colormap with NaN cells set to render as NAN_COLOR.

    Accepts either a registered colormap name or an existing Colormap
    instance, matching every call site's existing `cmap: str | Colormap`
    signature. Always returns a copy so the shared, registry-wide
    colormap object referenced by a bare name is never mutated in place.
    """
    resolved = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return resolved.with_extremes(bad=NAN_COLOR)
