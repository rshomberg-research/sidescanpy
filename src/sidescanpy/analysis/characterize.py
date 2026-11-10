"""
Value distribution summary and units inference.

- :func:`characterize_values` — dtype, shape and distribution statistics
  of one variable, no interpretation (:class:`ValueSummary`).
- :func:`infer_units` — likely units (dB, raw counts, normalized, …)
  from a :class:`ValueSummary`, optionally cross-checked against a
  range-decay fit (:class:`UnitsResult`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin
from sidescanpy.analysis.tvg_fit import TVGFitResult

if TYPE_CHECKING:
    from sidescanpy.analysis.tvg import RangeDecayResult

logger = logging.getLogger(__name__)


@dataclass
class ValueSummary(ResultMixin):
    """Distribution statistics of one variable (finite values only)."""

    dtype: str | None = None
    shape: list[int] = field(default_factory=list)
    n_values: int = 0
    n_finite: int = 0
    pct_nan: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    pct_zero: float = float("nan")
    pct_negative: float = float("nan")
    notes: list[str] = field(default_factory=list)


def characterize_values(ds: xr.Dataset, value_var: str = "backscatter_intensity") -> ValueSummary:
    """Data type, shape and distribution of ``ds[value_var]``."""
    arr = np.asarray(ds[value_var].values)
    finite = arr[np.isfinite(arr)]
    n_total = int(arr.size)
    n_finite = int(finite.size)

    notes: list[str] = []
    if n_finite == 0:
        notes.append(f"{value_var!r} has no finite values.")

    return ValueSummary(
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        n_values=n_total,
        n_finite=n_finite,
        pct_nan=100.0 * (n_total - n_finite) / n_total if n_total else float("nan"),
        minimum=float(finite.min()) if n_finite else float("nan"),
        maximum=float(finite.max()) if n_finite else float("nan"),
        mean=float(finite.mean()) if n_finite else float("nan"),
        median=float(np.median(finite)) if n_finite else float("nan"),
        std=float(finite.std()) if n_finite else float("nan"),
        pct_zero=100.0 * float(np.count_nonzero(finite == 0)) / n_finite if n_finite else float("nan"),
        pct_negative=100.0 * float(np.count_nonzero(finite < 0)) / n_finite if n_finite else float("nan"),
        notes=notes,
    )


@dataclass
class UnitsResult(ResultMixin):
    """Likely units of a variable.

    ``units`` is one of ``"dB"``, ``"normalized"``, ``"raw_counts"`` or
    ``"unknown"``.
    """

    units: str = "unknown"
    value_range: list[float | None] = field(default_factory=lambda: [None, None])
    pct_negative: float | None = None
    pct_zero: float | None = None
    notes: list[str] = field(default_factory=list)


def infer_units(summary: ValueSummary, decay: TVGFitResult | RangeDecayResult | None = None) -> UnitsResult:
    """Classify likely units from a value-range signature.

    - mostly negative, within (-200, 5] → ``"dB"``
    - non-negative, max in (0, 1] → ``"normalized"``
    - non-negative, max > 1 → ``"raw_counts"`` (raw ADC counts or
      uncalibrated linear amplitude; volts can't be told apart by value
      alone)
    - anything else → ``"unknown"``

    When ``decay`` (a :func:`~sidescanpy.analysis.fit_tvg_models` result,
    or a deprecated :func:`~sidescanpy.analysis.fit_range_decay` one) has
    a TVG status and a best-model RMS under 3 dB, a supporting note is
    added.
    """
    min_v, max_v = summary.minimum, summary.maximum
    pct_neg = summary.pct_negative
    pct_zero = summary.pct_zero

    notes: list[str] = []
    if min_v is None or max_v is None or not np.isfinite(min_v) or not np.isfinite(max_v):
        units = "unknown"
        notes.append("Value summary min/max not available or non-finite.")
    elif pct_neg is not None and pct_neg > 50.0 and min_v > -200.0 and max_v <= 5.0:
        units = "dB"
    elif min_v >= 0.0 and 0.0 < max_v <= 1.0:
        units = "normalized"
    elif min_v >= 0.0 and max_v > 1.0:
        units = "raw_counts"
        notes.append(
            "Non-negative, unbounded range is consistent with raw ADC counts or "
            "uncalibrated linear amplitude; not distinguishable from calibrated "
            "volts by value distribution alone."
        )
    else:
        units = "unknown"

    if isinstance(decay, TVGFitResult):
        best_rmse = decay.best_model.rms_db if decay.best_model is not None else None
        best_candidate = decay.best
    elif decay is not None:
        best_rmse, best_candidate = decay.best_rmse_db, decay.best_candidate
    if decay is not None and decay.tvg_status not in (None, "ambiguous"):
        if best_rmse is not None and best_rmse < 3.0:
            notes.append(
                f"The TVG fit ({best_candidate}, RMS {best_rmse:.2f} dB) is "
                f"consistent with genuine physical backscatter amplitude, supporting "
                f"the {units!r} unit call."
            )

    return UnitsResult(
        units=units, value_range=[min_v, max_v], pct_negative=pct_neg, pct_zero=pct_zero, notes=notes,
    )
