"""
Quantitative analysis tools for sidescan data.

Each function takes data and returns a result dataclass (with
``to_dict()``); none runs a workflow or plots (see
:mod:`sidescanpy.viz.analysis` and :mod:`sidescanpy.viz.tvg`).

- :func:`characterize_values` / :func:`infer_units` — value-range summary
  and units call (:class:`ValueSummary`, :class:`UnitsResult`).
- :func:`compare_outlier_rejection` — effect of range-binned outlier
  rejection (:class:`OutlierComparisonResult`).
- :func:`analyze_nadir` — nadir type, altitude and first-bottom-return
  boundaries (:class:`NadirResult`).
- :func:`estimate_beam_geometry` — near/far beam edges
  (:class:`BeamGeometryResult`).
- :func:`estimate_usable_range` — where signal falls to the noise floor
  (:class:`UsableRangeResult`).
- :func:`fit_range_decay` — TVG-status classification from a range-decay
  fit (:class:`RangeDecayResult`); :func:`fit_tvg_curve`,
  :func:`compare_before_after`, :func:`bin_by_altitude` are its building
  blocks.
- :func:`summarize_navigation` — heading convention and nav/attitude
  stats (:class:`NavigationSummary`).
- :func:`measure_ping_geometry` — ping spacing and resolution
  (:class:`PingGeometry`).
- :func:`detect_shadow_proposals` — object proposals from acoustic
  shadows and their highlights (:class:`ShadowProposals`).

2026 rshomberg@uri.edu
"""

from sidescanpy.analysis.beam import BeamGeometryResult, estimate_beam_geometry
from sidescanpy.analysis.characterize import (
    UnitsResult,
    ValueSummary,
    characterize_values,
    infer_units,
)
from sidescanpy.analysis.geometry import PingGeometry, measure_ping_geometry
from sidescanpy.analysis.nadir import NadirResult, analyze_nadir
from sidescanpy.analysis.navigation import FieldStats, NavigationSummary, summarize_navigation
from sidescanpy.analysis.outliers import OutlierComparisonResult, compare_outlier_rejection
from sidescanpy.analysis.shadows import ShadowProposals, detect_shadow_proposals
from sidescanpy.analysis.tvg import (
    RangeDecayResult,
    TVGCandidateFit,
    TVGCandidateFitResult,
    bin_by_altitude,
    compare_before_after,
    fit_range_decay,
    fit_tvg_curve,
)
from sidescanpy.analysis.usable_range import UsableRangeResult, estimate_usable_range

__all__ = [
    "characterize_values",
    "infer_units",
    "ValueSummary",
    "UnitsResult",
    "compare_outlier_rejection",
    "OutlierComparisonResult",
    "analyze_nadir",
    "NadirResult",
    "estimate_beam_geometry",
    "BeamGeometryResult",
    "estimate_usable_range",
    "UsableRangeResult",
    "fit_range_decay",
    "RangeDecayResult",
    "fit_tvg_curve",
    "TVGCandidateFit",
    "TVGCandidateFitResult",
    "compare_before_after",
    "bin_by_altitude",
    "summarize_navigation",
    "NavigationSummary",
    "FieldStats",
    "measure_ping_geometry",
    "PingGeometry",
    "detect_shadow_proposals",
    "ShadowProposals",
]
