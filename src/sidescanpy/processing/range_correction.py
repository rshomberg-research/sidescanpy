"""
Empirical range correction for the residual TVG range-dependence.

`processing/radiometric.py`'s `apply_tvg()` implements the physically-
grounded spreading+absorption TVG model. Even after that model is applied
real data can show a
remaining, altitude-invariant, non-monotonic range-dependent residual.
This module does not attempt to model the physical cause (it is not a
mistuned absorption coefficient or a missing grazing-angle/beam-pattern
correction). Instead it fits and removes whatever
range-dependent shape is actually observed in TVG-corrected data, per
frequency.

The library ships no calibration table: a table is fit from a specific
survey and instrument, so it belongs with the project that fitted it. Fit
one with :func:`fit_range_correction`, store it with
:meth:`RangeCorrectionTable.save`, and pass it (or its path) as the
``table`` argument of :func:`apply_empirical_range_correction` /
the ``"range_correction"`` step.

This is a separate, optional correction step (``"range_correction"``)
from `apply_tvg()` — not in ``sidescanpy.processing.DEFAULT_STEPS``; opt in
by listing it in the steps passed to ``sidescanpy.processing.correct``.

2026 rshomberg@uri.edu
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_RANGE_CORRECTION, STEP_TVG, append_history, has_step
from sidescanpy.core.units import FLOOR_DB
from sidescanpy.processing.profiles import backscatter_vs_range

logger = logging.getLogger(__name__)

STEP_NAME = STEP_RANGE_CORRECTION

# Frequency matching tolerance (Hz) — real freq_hz_start values are exact
# per-instrument constants, not estimated, so an exact-ish match is
# expected; a small tolerance just absorbs float round-tripping through
# JSON, not genuine frequency drift.
_FREQ_MATCH_TOLERANCE_HZ = 1000.0


@dataclass
class _FrequencyCurve:
    freq_hz: float
    range_m: list[float]
    correction_db: list[float]
    fit_metadata: dict


class RangeCorrectionTable:
    """
    Per-frequency empirical range-correction lookup.

    Each calibrated frequency stores a piecewise-linear correction curve
    (range-bin centers -> correction_db) fit by :func:`fit_range_correction`.
    This *is* the "non-parametric smooth interpolation over the observed
    bins" approach from the design doc — deliberately not a fitted
    polynomial, since the two frequencies calibrated so far have
    qualitatively different residual shapes (a smooth bowl vs. a sharp
    near-field rise-then-plateau) that a low-order polynomial risks
    fitting poorly or overfitting.
    """

    def __init__(self) -> None:
        self._curves: dict[float, _FrequencyCurve] = {}

    def add_curve(
        self,
        freq_hz: float,
        range_m: list[float],
        correction_db: list[float],
        fit_metadata: dict | None = None,
    ) -> None:
        """Add or replace the calibrated curve for one frequency."""
        self._curves[float(freq_hz)] = _FrequencyCurve(
            freq_hz=float(freq_hz),
            range_m=[float(v) for v in range_m],
            correction_db=[float(v) for v in correction_db],
            fit_metadata=dict(fit_metadata or {}),
        )

    def frequencies(self) -> list[float]:
        return sorted(self._curves)

    def _match(
        self, freq_hz: float, tolerance_hz: float = _FREQ_MATCH_TOLERANCE_HZ
    ) -> float | None:
        best, best_diff = None, None
        for f in self._curves:
            diff = abs(f - freq_hz)
            if diff <= tolerance_hz and (best_diff is None or diff < best_diff):
                best, best_diff = f, diff
        return best

    def matched_frequency(
        self, freq_hz: float, tolerance_hz: float = _FREQ_MATCH_TOLERANCE_HZ
    ) -> float | None:
        """The calibrated frequency key matching `freq_hz`, or None."""
        return self._match(freq_hz, tolerance_hz)

    def metadata_for(
        self, freq_hz: float, tolerance_hz: float = _FREQ_MATCH_TOLERANCE_HZ
    ) -> dict | None:
        key = self._match(freq_hz, tolerance_hz)
        return None if key is None else dict(self._curves[key].fit_metadata)

    def correction_db_at(
        self,
        freq_hz: float,
        range_m: np.ndarray,
        tolerance_hz: float = _FREQ_MATCH_TOLERANCE_HZ,
    ) -> np.ndarray | None:
        """
        Evaluate the fitted correction curve for `freq_hz` at `range_m`.

        Piecewise-linear interpolation between calibrated range-bin
        centers. `range_m` values outside the calibration data's observed
        span are **clamped** to the nearest edge's correction value
        (`np.interp`'s default constant-extrapolation behavior) — a
        deliberate, documented policy, not an accident: the calibration
        data doesn't cover ranges never seen in the fitting survey, and
        clamping is the safest default versus extrapolating an unverified
        shape.

        Returns
        -------
        np.ndarray, same shape as `range_m`, or None if no calibrated
        curve matches `freq_hz` within `tolerance_hz` — caller decides
        what to do (see `apply_empirical_range_correction`, which skips
        the correction with a warning rather than guessing).
        """
        key = self._match(freq_hz, tolerance_hz)
        if key is None:
            return None
        curve = self._curves[key]
        range_arr = np.asarray(range_m, dtype=np.float64)
        return np.interp(range_arr, curve.range_m, curve.correction_db)

    def to_dict(self) -> dict:
        return {
            "curves": [
                {
                    "freq_hz": c.freq_hz,
                    "range_m": c.range_m,
                    "correction_db": c.correction_db,
                    "fit_metadata": c.fit_metadata,
                }
                for c in self._curves.values()
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RangeCorrectionTable":
        table = cls()
        for c in d.get("curves", []):
            table.add_curve(
                c["freq_hz"], c["range_m"], c["correction_db"], c.get("fit_metadata")
            )
        return table

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "RangeCorrectionTable":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_range_correction(
    ds: xr.Dataset,
    bin_width_m: float = 5.0,
    exclude_below_db: float | None = -110.0,
) -> tuple[list[float], list[float], dict]:
    """
    Fit an empirical range-correction curve from one TVG-corrected channel.

    Parameters
    ----------
    ds : xr.Dataset
        Must be TVG-corrected and have a `ground_range_m` variable — i.e.
        already run through the `"tvg"` and `"geometric"` steps.
    bin_width_m : float, default 5.0
        Passed through to `processing.profiles.backscatter_vs_range`.
    exclude_below_db : float, optional, default -110.0
        Passed through as `exclude_below_db` to exclude the
        floor-clipped samples (~-120 dB exact-zero-amplitude floor) from
        biasing the fit. `None` disables filtering.

    Returns
    -------
    (range_m, correction_db, fit_metadata)
        `range_m`/`correction_db` are bin-center arrays ready for
        `RangeCorrectionTable.add_curve()`. `correction_db[i]` is defined
        as `target_level_db - mean_backscatter[i]`, where
        `target_level_db` is the count-weighted mean across all bins —
        i.e. the fitted correction has ~zero net effect on the survey's
        average absolute level, it only flattens the range-dependent
        shape.

    Raises
    ------
    ValueError
        If `ds` isn't TVG+geometric corrected, or no samples survive
        filtering.
    """
    if not has_step(ds, STEP_TVG):
        raise ValueError("fit_range_correction expects TVG-corrected data. Run apply_tvg() first.")
    if "ground_range_m" not in ds:
        raise ValueError(
            "fit_range_correction requires 'ground_range_m' "
            "(run compute_geometric_ranges() first)."
        )

    df = backscatter_vs_range(
        ds,
        value_var="backscatter_intensity",
        range_var="ground_range_m",
        bin_width_m=bin_width_m,
        exclude_below_db=exclude_below_db,
    )
    if df.empty:
        raise ValueError(
            "No samples survived filtering; cannot fit a range correction "
            "from this dataset/ping range."
        )

    range_centers = ((df["range_min_m"] + df["range_max_m"]) / 2.0).to_numpy()
    means = df["mean"].to_numpy()
    counts = df["count"].to_numpy()

    target_level_db = float(np.average(means, weights=counts))
    correction_db = target_level_db - means

    fit_metadata = {
        "freq_hz": float(ds.attrs["freq_hz_start"]),
        "bin_width_m": bin_width_m,
        "exclude_below_db": exclude_below_db,
        "target_level_db": target_level_db,
        "n_bins": int(len(df)),
        "n_samples_total": int(counts.sum()),
        "fitted_at_utc": datetime.now(UTC).isoformat(),
    }
    return range_centers.tolist(), correction_db.tolist(), fit_metadata


def apply_empirical_range_correction(
    ds: xr.Dataset, table: "RangeCorrectionTable | str | Path | None" = None
) -> xr.Dataset:
    """
    Apply a previously-fit empirical range correction to TVG-corrected
    backscatter.

    Parameters
    ----------
    ds : xr.Dataset
        TVG-corrected (``"tvg"`` in processing_history) with a
        ``ground_range_m`` variable — i.e. already run through the
        ``"tvg"`` and ``"geometric"`` steps.
    table : RangeCorrectionTable or path
        Required. The calibration to apply, or the path of a JSON file
        written by :meth:`RangeCorrectionTable.save`. The library has no
        built-in default table.

    Returns
    -------
    xr.Dataset
        New dataset with ``backscatter_intensity`` shifted by the fitted
        curve and a ``"range_correction"`` history record holding the
        curve actually applied (so it can be inverted without the table).

        Samples with non-finite ``ground_range_m`` (masked water-column
        samples) receive a **zero** correction rather than propagating NaN
        into previously-finite backscatter values. Floor-clipped samples
        (exactly :data:`~sidescanpy.core.units.FLOOR_DB`) are left at the
        floor, as by :func:`~sidescanpy.processing.radiometric.apply_tvg`.

        If no calibrated curve matches this dataset's ``freq_hz_start``,
        logs a warning and returns ``ds`` unchanged (no history record) —
        never guesses at an uncalibrated frequency.

    Raises
    ------
    TypeError
        If ``table`` is missing or not a table / path.
    ValueError
        If ``ds`` isn't TVG+geometric corrected, or the range correction is
        already recorded in processing_history (refuses to double-apply).
    """
    if table is None:
        raise TypeError(
            "the range_correction step requires a `table` (a RangeCorrectionTable or the path "
            "of a saved one); the library has no built-in default table"
        )
    if isinstance(table, (str, Path)):
        table = RangeCorrectionTable.load(str(table))
    elif not isinstance(table, RangeCorrectionTable):
        raise TypeError(f"`table` must be a RangeCorrectionTable or a path, got {type(table).__name__}")
    if not has_step(ds, STEP_TVG):
        raise ValueError(
            "apply_empirical_range_correction expects TVG-corrected data. Run apply_tvg() first."
        )
    if "ground_range_m" not in ds:
        raise ValueError(
            "apply_empirical_range_correction requires 'ground_range_m' "
            "(run compute_geometric_ranges() first)."
        )
    if has_step(ds, STEP_RANGE_CORRECTION):
        raise ValueError(
            "range correction already applied to this dataset "
            "(found in processing_history). Refusing to double-apply."
        )

    freq_hz = ds.attrs["freq_hz_start"]
    matched = table.matched_frequency(freq_hz)
    if matched is None:
        logger.warning(
            "No calibrated range-correction curve for freq_hz_start=%s "
            "(calibrated: %s); skipping empirical range correction, "
            "dataset returned unchanged.",
            freq_hz,
            table.frequencies(),
        )
        return ds

    curve = table._curves[matched]
    correction_da = range_correction_db(ds, curve.range_m, curve.correction_db)

    ds_out = ds.copy(deep=False)
    bs = ds["backscatter_intensity"]
    # Floor-clipped samples (exactly FLOOR_DB after TVG) stay at the floor.
    ds_out = ds_out.assign(
        backscatter_intensity=xr.where(bs == FLOOR_DB, bs, bs + correction_da, keep_attrs=True).astype(bs.dtype),
    )
    logger.debug(
        "Empirical range correction applied for freq_hz_start=%s (matched calibrated freq=%s).",
        freq_hz, matched,
    )
    return append_history(
        ds_out,
        STEP_RANGE_CORRECTION,
        {
            "freq_hz_matched": matched,
            "range_m": list(curve.range_m),
            "correction_db": list(curve.correction_db),
            "fit_metadata": dict(curve.fit_metadata),
        },
        function="sidescanpy.processing.range_correction.apply_empirical_range_correction",
        version=__version__,
    )


def range_correction_db(ds: xr.Dataset, range_m, correction_db) -> xr.DataArray:
    """Correction (dB) of a piecewise-linear curve evaluated at ``ds["ground_range_m"]``.

    Clamped outside the curve's span; non-finite ground range gets 0 dB.
    Shared by the forward step and its inversion.
    """
    range_m = np.asarray(range_m, dtype=np.float64)
    correction_db = np.asarray(correction_db, dtype=np.float64)

    def _eval(gr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.interp(gr.astype(np.float64), range_m, correction_db), nan=0.0)

    return xr.apply_ufunc(_eval, ds["ground_range_m"], dask="parallelized", output_dtypes=[np.float64])
