"""
Before/after view of per-range-bin outlier rejection.

:func:`compare_outlier_rejection` summarizes a variable's distribution
before and after :func:`~sidescanpy.processing.outliers.reject_range_outliers`,
so the effect of that option can be judged before using it in a fit.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin, array_field
from sidescanpy.analysis.characterize import ValueSummary, characterize_values
from sidescanpy.processing.outliers import reject_range_outliers

logger = logging.getLogger(__name__)


@dataclass
class OutlierComparisonResult(ResultMixin):
    """Distribution summaries before and after outlier rejection.

    ``filtered_values`` (the rejected-as-NaN array, same shape as the
    input variable) is kept for plotting and left out of ``to_dict()``.
    """

    bin_width_m: float | None = None
    mad_k: float | None = None
    n_rejected: int = 0
    pct_rejected: float = float("nan")
    before: ValueSummary = field(default_factory=ValueSummary)
    after: ValueSummary = field(default_factory=ValueSummary)
    filtered_values: np.ndarray | None = array_field()


def compare_outlier_rejection(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    range_var: str = "ground_range_m",
    *,
    bin_width_m: float = 5.0,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
) -> OutlierComparisonResult:
    """Summarize ``value_var`` before and after ``reject_range_outliers``.

    Purely descriptive; nothing else consumes the filtered values.
    ``bin_width_m``, ``outlier_mad_k`` and ``outlier_min_bin_count`` are
    passed to :func:`~sidescanpy.processing.outliers.reject_range_outliers`.
    Raises ``ValueError`` if ``range_var`` is missing.
    """
    if range_var not in ds:
        raise ValueError(f"compare_outlier_rejection: {range_var!r} not found — compute geometric ranges first")

    before = characterize_values(ds, value_var=value_var)

    filtered_arr = reject_range_outliers(
        ds[value_var].values, ds[range_var].values,
        bin_width_m=bin_width_m, mad_k=outlier_mad_k, min_bin_count=outlier_min_bin_count,
    )
    filtered_var = f"_{value_var}_outliers_removed"
    filtered_ds = ds.assign(**{filtered_var: (ds[value_var].dims, filtered_arr)})
    after = characterize_values(filtered_ds, value_var=filtered_var)

    original = np.asarray(ds[value_var].values)
    n_rejected = int(np.sum(np.isfinite(original) & ~np.isfinite(filtered_arr)))
    pct_rejected = 100.0 * n_rejected / original.size if original.size else float("nan")

    return OutlierComparisonResult(
        bin_width_m=bin_width_m,
        mad_k=outlier_mad_k,
        n_rejected=n_rejected,
        pct_rejected=pct_rejected,
        before=before,
        after=after,
        filtered_values=filtered_arr,
    )
