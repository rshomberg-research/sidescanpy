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
  (:class:`UsableRangeResult`); :func:`find_noise_knee` — peak, floor and
  noise knee of a level profile (:class:`NoiseKnee`).
- :func:`fit_tvg_models` — the TVG to apply: fixed and free spreading
  and absorption models fitted to the median level
  (:class:`TVGFitResult`); :func:`check_tvg` — how flat a TVG recipe
  leaves the data (:class:`TVGCheckResult`).
- :func:`compare_before_after`, :func:`bin_by_altitude` — TVG
  comparison helpers.
- Deprecated, removed in 0.12: :func:`fit_range_decay`
  (:class:`RangeDecayResult`), :func:`fit_tvg_curve`
  (:class:`TVGCandidateFit`, :class:`TVGCandidateFitResult`); use
  :func:`fit_tvg_models`.
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
from sidescanpy.analysis.tvg_fit import (
    TVGCheckResult,
    TVGFitResult,
    TVGModelFit,
    TVGSideFit,
    check_tvg,
    fit_tvg_models,
    tvg_model_name,
)
from sidescanpy.analysis.usable_range import (
    NoiseKnee,
    UsableRangeResult,
    estimate_usable_range,
    find_noise_knee,
)

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
    "find_noise_knee",
    "NoiseKnee",
    "fit_tvg_models",
    "TVGFitResult",
    "TVGModelFit",
    "TVGSideFit",
    "check_tvg",
    "TVGCheckResult",
    "tvg_model_name",
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
