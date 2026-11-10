"""
TVG model fit: which spreading and absorption law the data still carries.

- :func:`fit_tvg_models` — fit fixed TVG models (no TVG, absorption only,
  20/30/40 log R, each with and without absorption) and a free spreading
  coefficient to the median level of raw data, over a window from past
  the profile peak to the noise knee (:class:`TVGFitResult`).
- :func:`check_tvg` — apply a TVG recipe to the data and measure how flat
  the corrected median level is over a window (:class:`TVGCheckResult`).

The model is the sonar equation's range dependence,
``L(R) = L0 - n log10 R - 2 alpha R``, with the constant ``L0`` (source
level, gain and seabed strength together) fitted by least squares. The
best model is the TVG to apply: its ``n`` and ``alpha`` are the
``spreading_loss_db`` and ``absorption_db_per_m`` of
:func:`~sidescanpy.processing.radiometric.apply_tvg`.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from sidescanpy.analysis._common import resolve_channel_idx
from sidescanpy.analysis._result import ResultMixin
from sidescanpy.analysis.usable_range import find_noise_knee
from sidescanpy.core.history import RADIOMETRIC_STEPS, has_step
from sidescanpy.processing.profiles import median_level_profile
from sidescanpy.processing.radiometric import (
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
    center_frequency_hz,
    resolve_tvg_params,
)
from sidescanpy.processing.slicing import slice_by_range

logger = logging.getLogger(__name__)

__all__ = [
    "TVGCheckResult",
    "TVGFitResult",
    "TVGModelFit",
    "TVGSideFit",
    "check_tvg",
    "fit_tvg_models",
    "tvg_model_name",
]

# What hardware has already removed, given the TVG still to apply.
_STATUS_FOR_TERMS = {
    (False, False): "full",
    (False, True): "spreading_only",
    (True, False): "absorption_only",
    (True, True): "none",
}


def tvg_model_name(spreading_loss_db: float, with_absorption: bool) -> str:
    """Display name of a TVG model: ``"flat"``, ``"absorption only"``,
    ``"30 log R"``, ``"30 log R + abs"``, ..."""
    if spreading_loss_db == 0:
        return "absorption only" if with_absorption else "flat"
    n = f"{spreading_loss_db:g}"
    return f"{n} log R" + (" + abs" if with_absorption else "")


@dataclass
class TVGModelFit(ResultMixin):
    """One fixed TVG model fitted to the median level.

    The model level is ``offset_db - spreading_loss_db * log10(R) -
    2 * absorption_db_per_m * R``; ``offset_db`` is the least-squares
    constant and ``rms_db`` the RMS residual over the fit window.
    ``absorption_db_per_m`` is 0 for a model without absorption.
    """

    name: str
    spreading_loss_db: float
    absorption_db_per_m: float
    offset_db: float
    rms_db: float

    def level_db(self, range_m: np.ndarray) -> np.ndarray:
        """Model level (dB) at ``range_m``."""
        r = np.asarray(range_m, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.offset_db - self.spreading_loss_db * np.log10(r) - 2.0 * self.absorption_db_per_m * r


@dataclass
class TVGSideFit(ResultMixin):
    """The model fit of one side alone, over the pooled fit window.

    ``best`` and ``rms_db`` are that side's best model; ``free_n`` its
    free spreading coefficient (with absorption); ``pooled_best_rms_db``
    the RMS of the pooled best model's shape on this side (constant
    refitted); ``offset_db`` the side's median level minus the pooled
    median level, averaged over the window.
    """

    side: str
    best: str | None = None
    rms_db: float | None = None
    free_n: float | None = None
    pooled_best_rms_db: float | None = None
    offset_db: float | None = None


@dataclass
class TVGFitResult(ResultMixin):
    """TVG models fitted to the median level of raw data.

    Profile: ``range_m`` (bin centers), ``level_db`` (both sides pooled,
    or the requested ``channel``) and ``level_db_sides`` (port, starboard),
    the median of linear amplitude in dB re 1 recorded unit.

    Window: ``window_m`` is ``(start, end)``; ``window_source`` is
    ``"knee"`` (peak plus ``start_margin_m`` to the noise knee),
    ``"end_of_record"`` (no knee found, fit to the end) or ``"given"``.
    ``peak_m``, ``knee_m``, ``knee_found``, ``contrast_db``,
    ``low_contrast`` and ``rises_at_far_range`` come from
    :func:`~sidescanpy.analysis.find_noise_knee`.

    Models: ``models`` maps each name to its :class:`TVGModelFit`;
    ``best`` and ``runner_up`` are names, ranked by RMS in the window.
    ``free_n`` is the least-squares spreading coefficient with absorption
    (``free_n_rms_db`` its RMS), ``free_n_no_absorption`` the same
    without. ``side_best`` holds each side's fit (:class:`TVGSideFit`) and
    ``sides_disagree`` flags sides whose levels or shapes differ.

    Use: ``tvg_params`` is the best model as
    :class:`~sidescanpy.processing.radiometric.TVGParams` (the TVG to
    apply); ``tvg_status`` names what the recorder has already removed
    (``"none"``, ``"spreading_only"``, ``"absorption_only"``, ``"full"``),
    derived from the best model. Both are ``None`` without a fit, and both
    are included in ``to_dict()``.
    """

    range_m: np.ndarray | None = None
    level_db: np.ndarray | None = None
    level_db_sides: np.ndarray | None = None
    bin_width_m: float | None = None
    window_m: tuple[float, float] | None = None
    window_source: str | None = None
    n_bins: int = 0
    peak_m: float | None = None
    knee_m: float | None = None
    knee_found: bool = False
    contrast_db: float | None = None
    low_contrast: bool = False
    rises_at_far_range: bool = False
    start_margin_m: float | None = None
    knee_db: float | None = None
    altitude_m: float | None = None
    models: dict[str, TVGModelFit] = field(default_factory=dict)
    best: str | None = None
    runner_up: str | None = None
    free_n: float | None = None
    free_n_rms_db: float | None = None
    free_n_no_absorption: float | None = None
    free_n_no_absorption_rms_db: float | None = None
    side_best: list[TVGSideFit] = field(default_factory=list)
    sides_disagree: bool = False
    absorption_db_per_m: float | None = None
    freq_hz: float | None = None
    dithered: bool = False
    value_var: str | None = None
    range_var: str | None = None
    channel: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def best_model(self) -> TVGModelFit | None:
        """The best :class:`TVGModelFit`, or ``None``."""
        return self.models.get(self.best) if self.best else None

    @property
    def runner_up_model(self) -> TVGModelFit | None:
        """The runner-up :class:`TVGModelFit`, or ``None``."""
        return self.models.get(self.runner_up) if self.runner_up else None

    @property
    def tvg_params(self) -> TVGParams | None:
        """The best model as ``TVGParams`` (absorption 0 when it has none)."""
        best = self.best_model
        if best is None:
            return None
        return TVGParams(spreading_loss_db=best.spreading_loss_db, absorption_db_per_m=best.absorption_db_per_m)

    @property
    def tvg_status(self) -> str | None:
        """What the recorder has already removed, derived from the best model."""
        best = self.best_model
        if best is None:
            return None
        return _STATUS_FOR_TERMS[(best.spreading_loss_db > 0, best.absorption_db_per_m > 0)]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict of the fields plus ``tvg_params`` and ``tvg_status``."""
        out = super().to_dict()
        params = self.tvg_params
        out["tvg_params"] = None if params is None else {
            "spreading_loss_db": params.spreading_loss_db, "absorption_db_per_m": params.absorption_db_per_m,
        }
        out["tvg_status"] = self.tvg_status
        return out


@dataclass
class TVGCheckResult(ResultMixin):
    """How flat the median level is after a TVG recipe.

    ``level_db`` is the median level per bin (``range_m``) after
    :func:`~sidescanpy.processing.radiometric.apply_tvg`, on data cropped
    at ``crop_m``. Over ``window_m``: ``spread_db`` is its standard
    deviation, ``trend_db_per_10m`` the slope of a straight-line fit and
    ``end_to_end_db`` last bin minus first. ``window_mean_db`` is its
    mean there (``relative_db`` subtracts it). ``params`` holds the
    ``spreading_loss_db`` and ``absorption_db_per_m`` applied.
    """

    range_m: np.ndarray | None = None
    level_db: np.ndarray | None = None
    window_m: tuple[float, float] | None = None
    crop_m: float | None = None
    spread_db: float | None = None
    trend_db_per_10m: float | None = None
    end_to_end_db: float | None = None
    window_mean_db: float | None = None
    n_bins: int = 0
    params: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def relative_db(self) -> np.ndarray | None:
        """``level_db`` minus its mean over the window."""
        if self.level_db is None or self.window_mean_db is None:
            return None
        return self.level_db - self.window_mean_db

    @property
    def model_name(self) -> str:
        """Display name of the applied recipe (see :func:`tvg_model_name`)."""
        return tvg_model_name(self.params.get("spreading_loss_db", 0.0), self.params.get("absorption_db_per_m", 0.0) > 0)


# ============================================================
# fitting core
# ============================================================

def _window_mask(range_m: np.ndarray, level_db: np.ndarray, window_m) -> np.ndarray:
    return (range_m >= window_m[0]) & (range_m <= window_m[1]) & np.isfinite(level_db)


def _fit_fixed(r: np.ndarray, y: np.ndarray, n: float, alpha: float) -> tuple[float, float]:
    """Offset and RMS of ``y`` against ``-n log10 r - 2 alpha r`` with a free constant."""
    shape = -n * np.log10(r) - 2.0 * alpha * r
    offset = float(np.mean(y - shape))
    return offset, float(np.sqrt(np.mean((y - shape - offset) ** 2)))


def _fit_free_n(r: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Least-squares ``n`` and RMS of ``y + 2 alpha r = L0 - n log10 r``."""
    design = np.c_[np.ones_like(r), -np.log10(r)]
    target = y + 2.0 * alpha * r
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(coef[1]), float(np.sqrt(np.mean((design @ coef - target) ** 2)))


def _fit_models(range_m, level_db, window_m, alpha, spreading, absorption, min_bins) -> dict[str, TVGModelFit]:
    sel = _window_mask(range_m, level_db, window_m)
    if sel.sum() < min_bins:
        return {}
    r, y = range_m[sel], level_db[sel]
    fits = {}
    for n in spreading:
        for with_abs in absorption:
            a = alpha if with_abs else 0.0
            offset, rms = _fit_fixed(r, y, float(n), a)
            name = tvg_model_name(float(n), bool(with_abs))
            fits[name] = TVGModelFit(name=name, spreading_loss_db=float(n), absorption_db_per_m=float(a),
                                     offset_db=offset, rms_db=rms)
    return fits


def _ranked(models: dict[str, TVGModelFit]) -> list[str]:
    return sorted(models, key=lambda k: models[k].rms_db)


# ============================================================
# fit_tvg_models
# ============================================================

def fit_tvg_models(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    range_var: str = "slant_range_m",
    *,
    bin_width_m: float = 1.0,
    start_margin_m: float = 5.0,
    knee_db: float = 3.0,
    window_m: tuple[float, float] | None = None,
    min_window_m: float = 8.0,
    low_contrast_db: float = 6.0,
    spreading: tuple[float, ...] = (0.0, 20.0, 30.0, 40.0),
    absorption: tuple[bool, ...] = (False, True),
    absorption_db_per_m: float | None = None,
    freq_hz: float | None = None,
    channel: int | str | None = None,
    dither: bool | str = "auto",
    seed: int = 0,
    min_count: int = 50,
    min_bins: int = 3,
    side_difference_db: float = 15.0,
    side_rms_tol_db: float = 1.0,
    altitude_factor: float = 2.5,
) -> TVGFitResult:
    """Fit TVG models to the median level of raw backscatter.

    1. Profile: the median of linear amplitude in ``bin_width_m`` bins of
       ``range_var``, past the median altitude (or, without
       ``altitude_m``, outside the water column), both sides pooled (or
       ``channel``), in dB re 1
       (:func:`~sidescanpy.processing.profiles.median_level_profile`;
       integer samples are dithered first).
    2. Window: from ``start_margin_m`` past the profile peak (searched in
       the near half of the range) to the noise knee, the first range
       within ``knee_db`` of the lowest level
       (:func:`~sidescanpy.analysis.find_noise_knee`). When no knee leaves
       ``min_window_m``, the window runs to the end of the record, with a
       note; ``window_m`` overrides both ends.
    3. Models: every combination of ``spreading`` (n log R, 0 = none) and
       ``absorption`` (Francois-Garrison at the channel's center
       frequency, or ``absorption_db_per_m``), each with a least-squares
       constant, ranked by RMS residual in the window. A free ``n`` is
       fitted with and without absorption.
    4. Each side is fitted alone on the same window; ``sides_disagree``
       is set (WARNING) when the two sides' mean levels differ by more
       than ``side_difference_db``, or the pooled best model's shape fits
       a side ``side_rms_tol_db`` worse than that side's own best.

    A good fit is a hypothesis: confirm it with :func:`check_tvg` on the
    same window. Over short windows (under ~20 m) 20 log R with
    absorption, 30 log R and 40 log R differ by tenths of a dB; read
    ``free_n`` and the runner-up before trusting the label.

    Parameters
    ----------
    ds : xr.Dataset
        One channel with ``range_var`` (see
        :func:`~sidescanpy.processing.geometric.compute_geometric_ranges`),
        ``value_var`` in linear amplitude and no radiometric step applied
        (undo one with :func:`~sidescanpy.processing.invert`).
        ``ground_range_m``, when present, marks the water column.
    value_var : str, default "backscatter_intensity"
    range_var : str, default "slant_range_m"
        Slant range matches ``apply_tvg``'s gain.
    bin_width_m : float, default 1.0
    start_margin_m : float, default 5.0
    knee_db : float, default 3.0
    window_m : (float, float), optional
        Fit window ``(start, end)`` in m instead of the automatic one.
    min_window_m : float, default 8.0
    low_contrast_db : float, default 6.0
        Peak-to-floor contrast below which ``low_contrast`` is set.
    spreading : tuple of float, default (0, 20, 30, 40)
    absorption : tuple of bool, default (False, True)
    absorption_db_per_m : float, optional
        Absorption of the models with absorption. Default:
        Francois-Garrison at ``freq_hz``.
    freq_hz : float, optional
        Default: the channel's center frequency.
    channel : int or str, optional
        Fit one side (index or ``channel_id``) instead of both pooled.
    dither : bool or "auto", default "auto"
    seed : int, default 0
    min_count : int, default 50
        Samples a bin needs for a level.
    min_bins : int, default 3
        Bins the window needs for a fit.
    side_difference_db : float, default 15.0
    side_rms_tol_db : float, default 1.0
        Thresholds of ``sides_disagree``.
    altitude_factor : float, default 2.5
        A WARNING is logged when the profile peak lies beyond
        ``altitude_factor * altitude + 5 m`` or the profile starts more
        than 5 m past the altitude.

    Returns
    -------
    TVGFitResult

    Raises
    ------
    ValueError
        If ``range_var`` is missing or a radiometric step (TVG, range
        correction) is already applied.
    """
    if range_var not in ds:
        raise ValueError(f"fit_tvg_models: {range_var!r} not found — compute geometric ranges first")
    applied = [step for step in RADIOMETRIC_STEPS if has_step(ds, step)]
    if applied and value_var == "backscatter_intensity":
        raise ValueError(
            f"fit_tvg_models: {applied} already applied; fit raw data "
            "(undo the steps with sidescanpy.processing.invert first)"
        )
    channel_idx = resolve_channel_idx(ds, channel, caller="fit_tvg_models") if channel is not None else None

    freq_hz = float(freq_hz) if freq_hz is not None else center_frequency_hz(ds)
    alpha = float(absorption_db_per_m) if absorption_db_per_m is not None else float(absorption_for_frequency(freq_hz))
    altitude = None
    if "altitude_m" in ds:
        alt = np.asarray(ds["altitude_m"].values, dtype=np.float64)
        if np.isfinite(alt).any():
            altitude = float(np.nanmedian(alt))

    # One survey-wide cut at the median altitude keeps every bin's pings the
    # same (a per-ping cut biases the near bins); without altitude, the
    # water column is taken from ground_range_m.
    profile_kw = dict(bin_width_m=bin_width_m, min_count=min_count, dither=dither, seed=seed,
                      min_range_m=altitude, exclude_water_column=altitude is None)
    pooled = median_level_profile(ds, value_var, range_var, channel=channel_idx, **profile_kw)
    sides = [median_level_profile(ds, value_var, range_var, channel=s, **profile_kw) for s in (0, 1)]
    range_m = pooled["range_m"].to_numpy()
    level = pooled["level_db"].to_numpy()
    result = TVGFitResult(
        range_m=range_m, level_db=level,
        level_db_sides=np.vstack([s["level_db"].to_numpy() for s in sides]),
        bin_width_m=bin_width_m, start_margin_m=start_margin_m, knee_db=knee_db, altitude_m=altitude,
        absorption_db_per_m=alpha, freq_hz=freq_hz, dithered=bool(pooled.attrs.get("dithered", False)),
        value_var=value_var, range_var=range_var, channel=channel_idx,
    )
    notes = result.notes
    if result.dithered:
        notes.append("Integer samples: dithered by a uniform offset in [-0.5, 0.5) before taking medians.")
    if not np.isfinite(level).any():
        notes.append("No range bin has enough samples for a level; nothing to fit.")
        logger.warning("fit_tvg_models: empty profile, no fit")
        return result

    knee = find_noise_knee(range_m, level, knee_db, start_margin_m=start_margin_m,
                           min_window_m=min_window_m, low_contrast_db=low_contrast_db)
    result.peak_m, result.knee_m, result.knee_found = knee.peak_m, knee.knee_m, knee.knee_found
    result.contrast_db, result.low_contrast = knee.contrast_db, knee.low_contrast
    result.rises_at_far_range = knee.rises_at_far_range
    notes.extend(knee.notes)
    if window_m is not None:
        result.window_m = (float(window_m[0]), float(window_m[1]))
        result.window_source = "given"
    else:
        result.window_m = knee.window_m
        result.window_source = "knee" if knee.knee_found else "end_of_record"
    sel = _window_mask(range_m, level, result.window_m)
    result.n_bins = int(sel.sum())

    _check_altitude(result, range_m, level, altitude, altitude_factor)

    models = _fit_models(range_m, level, result.window_m, alpha, spreading, absorption, min_bins)
    if not models:
        notes.append(f"Fit window {result.window_m[0]:.1f}-{result.window_m[1]:.1f} m holds "
                     f"{result.n_bins} bins (fewer than {min_bins}); no fit.")
        logger.debug("fit_tvg_models: %s", notes[-1])
        return result
    result.models = models
    ranked = _ranked(models)
    result.best = ranked[0]
    result.runner_up = ranked[1] if len(ranked) > 1 else None
    r, y = range_m[sel], level[sel]
    result.free_n, result.free_n_rms_db = _fit_free_n(r, y, alpha)
    result.free_n_no_absorption, result.free_n_no_absorption_rms_db = _fit_free_n(r, y, 0.0)

    _fit_sides(result, sides, spreading, absorption, alpha, min_bins, side_difference_db, side_rms_tol_db)

    best = models[result.best]
    runner = f"; runner-up {result.runner_up} ({models[result.runner_up].rms_db:.2f} dB)" if result.runner_up else ""
    notes.append(
        f"Best {best.name} ({best.rms_db:.2f} dB RMS over {result.window_m[0]:.1f}-{result.window_m[1]:.1f} m)"
        f"{runner}; free n {result.free_n:.1f} with absorption, {result.free_n_no_absorption:.1f} without."
    )
    logger.debug("fit_tvg_models: %s", notes[-1])
    return result


def _check_altitude(result: TVGFitResult, range_m, level, altitude, factor) -> None:
    """Note and WARN when the profile peak or start sits far from the altitude."""
    if altitude is None or not np.isfinite(altitude) or altitude <= 0:
        return
    first = float(range_m[np.flatnonzero(np.isfinite(level))[0]])
    msgs = []
    if result.peak_m is not None and result.peak_m > factor * altitude + 5.0:
        msgs.append(f"the profile peak ({result.peak_m:.1f} m) is far past the altitude ({altitude:.1f} m)")
    if first > altitude + 5.0:
        msgs.append(f"the profile starts at {first:.1f} m, well past the altitude ({altitude:.1f} m)")
    if msgs:
        text = "Check the altitude: " + "; ".join(msgs) + "."
        result.notes.append(text)
        logger.warning("fit_tvg_models: %s", text)


def _fit_sides(result, sides, spreading, absorption, alpha, min_bins, side_difference_db, side_rms_tol_db) -> None:
    range_m, pooled = result.range_m, result.level_db
    best_pooled = result.models[result.best]
    disagree = []
    for s, prof in zip(("port", "starboard"), sides):
        level = prof["level_db"].to_numpy()
        side = TVGSideFit(side=s)
        result.side_best.append(side)
        fits = _fit_models(range_m, level, result.window_m, alpha, spreading, absorption, min_bins)
        if not fits:
            continue
        side.best = _ranked(fits)[0]
        side.rms_db = fits[side.best].rms_db
        sel = _window_mask(range_m, level, result.window_m)
        side.free_n = _fit_free_n(range_m[sel], level[sel], alpha)[0]
        side.pooled_best_rms_db = fits[best_pooled.name].rms_db if best_pooled.name in fits else None
        both = sel & np.isfinite(pooled)
        if both.any():
            side.offset_db = float(np.mean(level[both] - pooled[both]))
        if side.pooled_best_rms_db is not None and side.pooled_best_rms_db - side.rms_db > side_rms_tol_db:
            disagree.append(f"{s} fits {side.best} ({side.rms_db:.2f} dB) against {best_pooled.name} "
                            f"({side.pooled_best_rms_db:.2f} dB)")
    port, stbd = (sf.offset_db for sf in result.side_best)
    if port is not None and stbd is not None and abs(port - stbd) > side_difference_db:
        disagree.insert(0, f"port level {port - stbd:+.1f} dB from starboard")
    if disagree:
        result.sides_disagree = True
        text = "Sides disagree: " + "; ".join(disagree) + ". The pooled median may follow neither side."
        result.notes.append(text)
        logger.warning("fit_tvg_models: %s", text)


# ============================================================
# check_tvg
# ============================================================

def _as_params(params: TVGParams | Mapping[str, float]) -> TVGParams:
    if isinstance(params, TVGParams):
        return params
    allowed = {k: v for k, v in dict(params).items() if k in TVGParams.__dataclass_fields__}
    return TVGParams(**allowed)


def check_tvg(
    ds: xr.Dataset,
    tvg_params: TVGParams | Mapping[str, float],
    window_m: tuple[float, float],
    value_var: str = "backscatter_intensity",
    range_var: str = "slant_range_m",
    *,
    crop_m: float | None = None,
    bin_width_m: float = 1.0,
    channel: int | str | None = None,
    dither: bool | str = "auto",
    seed: int = 0,
    min_count: int = 50,
) -> TVGCheckResult:
    """Apply a TVG recipe and measure how flat the median level is.

    The data are cropped at ``crop_m`` (default: the window's far end;
    beyond it TVG only amplifies noise) with
    :func:`~sidescanpy.processing.slicing.slice_by_range`, corrected with
    :func:`~sidescanpy.processing.radiometric.apply_tvg`, and binned like
    :func:`fit_tvg_models` (median of linear amplitude, dB). If the recipe
    is right the corrected level is flat across the window: ``spread_db``
    near the fit's RMS and ``trend_db_per_10m`` near zero. Validate on
    the same window the fit used.

    Parameters
    ----------
    ds : xr.Dataset
        Raw channel, as for :func:`fit_tvg_models`.
    tvg_params : TVGParams or mapping
        The recipe, e.g. ``fit.tvg_params`` or a config's ``tvg`` block
        (``spreading_loss_db``, ``absorption_db_per_m``; an absent
        absorption means Francois-Garrison, as in ``apply_tvg``).
    window_m : (float, float)
        Where to measure flatness (normally ``fit.window_m``).
    value_var, range_var : str
    crop_m : float, optional
    bin_width_m : float, default 1.0
    channel : int or str, optional
        Measure one side only.
    dither : bool or "auto", default "auto"
        Applied before the TVG, as in the fit.
    seed : int, default 0
    min_count : int, default 50

    Returns
    -------
    TVGCheckResult
    """
    params = _as_params(tvg_params)
    window_m = (float(window_m[0]), float(window_m[1]))
    crop_m = float(crop_m) if crop_m is not None else window_m[1]
    channel_idx = resolve_channel_idx(ds, channel, caller="check_tvg") if channel is not None else None

    crop = slice_by_range(ds, max_m=crop_m)
    values = np.asarray(crop[value_var].values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if dither == "auto":
        dither = bool(finite.size) and bool(np.all(finite == np.round(finite)))
    if dither:
        values = values + np.random.default_rng(seed).uniform(-0.5, 0.5, values.shape)
    crop = crop.assign(backscatter_intensity=(crop[value_var].dims, values.astype(np.float32)))

    resolved = resolve_tvg_params(crop, params)
    corrected = apply_tvg(crop, resolved)
    linear = 10.0 ** (np.asarray(corrected["backscatter_intensity"].values, dtype=np.float64) / 20.0)
    work = corrected.assign(_after_tvg_linear=(corrected["backscatter_intensity"].dims, linear))
    prof = median_level_profile(work, "_after_tvg_linear", range_var, bin_width_m=bin_width_m,
                                channel=channel_idx, min_count=min_count, dither=False)
    range_m = prof["range_m"].to_numpy()
    level = prof["level_db"].to_numpy()
    result = TVGCheckResult(
        range_m=range_m, level_db=level, window_m=window_m, crop_m=crop_m,
        params={"spreading_loss_db": float(resolved.spreading_loss_db),
                "absorption_db_per_m": float(resolved.absorption_db_per_m)},
    )
    sel = _window_mask(range_m, level, window_m)
    result.n_bins = int(sel.sum())
    if result.n_bins < 3:
        result.notes.append(f"Window {window_m[0]:.1f}-{window_m[1]:.1f} m holds {result.n_bins} bins; no check.")
        return result
    p = level[sel]
    result.spread_db = float(np.std(p))
    result.trend_db_per_10m = float(10.0 * np.polyfit(range_m[sel], p, 1)[0])
    result.end_to_end_db = float(p[-1] - p[0])
    result.window_mean_db = float(np.mean(p))
    logger.debug("check_tvg: %s spread %.2f dB, trend %+.2f dB/10 m", result.model_name, result.spread_db,
                 result.trend_db_per_10m)
    return result
