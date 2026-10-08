"""
Radiometric correction for sidescan sonar data.

This module implements Time-Varying Gain (TVG) correction to compensate for
transmission losses in the water column. The raw backscatter_intensity values
represent echo levels received by the transducer but do not account for:

1. Geometric spreading: sound spreads in 3D, losing energy proportional to 1/R²
   (in dB: 20*log10(R) dB loss)
2. Absorption: water absorbs acoustic energy proportional to distance traveled
   (in dB: 2*alpha*R dB loss for two-way path)

TVG correction adds these losses back, so equal backscatter at different ranges
produces equal corrected values. This is essential for radiometric calibration
and mosaic generation.

The default absorption coefficient comes from the Francois-Garrison seawater
absorption model (:func:`francois_garrison_absorption_db_per_m`):

- R. E. Francois and G. R. Garrison, "Sound absorption based on ocean measurements: Part I: Pure water and magnesium sulfate contributions," J. Acoust. Soc. Am. 72(3), 896-907 (1982).
- R. E. Francois and G. R. Garrison, "Sound absorption based on ocean measurements: Part II: Boric acid contribution and equation for total absorption," J. Acoust. Soc. Am. 72(6), 1879-1890 (1982).

Input backscatter is assumed to be linear amplitude; this module converts to
dB, applies TVG, and outputs dB. The step is recorded in ``processing_history``
with its resolved parameters so it can be inverted exactly.

2026 rshomberg@uri.edu
"""

import logging
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_TVG, append_history, has_step
from sidescanpy.core.units import linear_to_db
from sidescanpy.processing.sample_axis import DEFAULT_SOUND_VEL_MS, sample_index

logger = logging.getLogger(__name__)

# "Typical coastal" water used when the caller gives no conditions.
DEFAULT_TEMPERATURE_C = 10.0
DEFAULT_SALINITY_PSU = 35.0
DEFAULT_DEPTH_M = 10.0
DEFAULT_PH = 8.0


@dataclass
class TVGParams:
    """
    Parameters for time-varying gain correction.

    Attributes
    ----------
    spreading_loss_db : float, default=20.0
        Spreading loss coefficient (dB per decade of range).
        Standard value is 20.0 for spherical spreading in 3D.
    absorption_db_per_m : float, optional
        Absorption coefficient (dB per meter).
        If None, derived from the channel frequency with
        :func:`francois_garrison_absorption_db_per_m` (Francois & Garrison,
        J. Acoust. Soc. Am. 72(3) 896-907 and 72(6) 1879-1890, 1982) and the
        four water parameters below (typical-coastal defaults where None).
    min_range_m : float, default=0.5
        Minimum range threshold (meters).
        Ranges below this are clipped to avoid log(0) singularity.
        Typical value 0.5 m accounts for transducer blanking.
    sound_velocity_ms : float, optional
        Sound velocity (m/s) used to turn sample index into slant range.
        If None, the mean of the dataset's per-ping ``sound_vel_ms``.
    temperature_c, salinity_psu, depth_m, ph : float, optional
        Water conditions for the Francois-Garrison absorption; used only
        when ``absorption_db_per_m`` is None (defaults 10 C, 35 psu, 10 m,
        pH 8 -- pass real values when known).
    """

    spreading_loss_db: float = 20.0
    absorption_db_per_m: float | None = None
    min_range_m: float = 0.5
    sound_velocity_ms: float | None = None
    temperature_c: float | None = None
    salinity_psu: float | None = None
    depth_m: float | None = None
    ph: float | None = None


def francois_garrison_absorption_db_per_m(
    frequency_hz,
    *,
    temperature_c: float = DEFAULT_TEMPERATURE_C,
    salinity_psu: float = DEFAULT_SALINITY_PSU,
    depth_m: float = DEFAULT_DEPTH_M,
    ph: float = DEFAULT_PH,
    sound_velocity_ms: float | None = None,
):
    """
    Seawater sound absorption (dB/m, one-way) after Francois & Garrison (1982).

    Sum of three relaxation / viscosity terms (``f`` in kHz, result in dB/km
    before conversion)::

        alpha = A1 P1 f1 f^2 / (f1^2 + f^2)      boric acid
              + A2 P2 f2 f^2 / (f2^2 + f^2)      magnesium sulfate
              + A3 P3 f^2                        pure water (viscosity)

    with the paper's coefficients: ``f1 = 2.8 sqrt(S/35) 10^(4 - 1245/(T+273))``,
    ``f2 = 8.17 10^(8 - 1990/(T+273)) / (1 + 0.0018 (S - 35))``, ``A3`` having
    separate polynomial branches for ``T <= 20 C`` and ``T > 20 C``, and
    depth factors ``P2``, ``P3``. Valid roughly 0.4 - 1000 kHz, -1.8 to 30 C,
    S 30-35 psu, depth 0 - 6 km.

    References
    ----------
    .. [1] R. E. Francois and G. R. Garrison, "Sound absorption based on ocean measurements: Part I: Pure water and magnesium sulfate contributions," J. Acoust. Soc. Am. 72(3), 896-907 (1982)
           (pure water and magnesium sulfate terms).
    .. [2] R. E. Francois and G. R. Garrison, "Sound absorption based on ocean measurements: Part II: Boric acid contribution and equation for total absorption," J. Acoust. Soc. Am. 72(6), 1879-1890 (1982)
           (boric acid term and the equation for total absorption).

    The defaults are "typical coastal" values (10 C, 35 psu, 10 m, pH 8),
    not a measurement: callers with real conditions should pass their own.

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency (Hz). Vectorised.
    temperature_c, salinity_psu, depth_m, ph : float
        Water conditions (degrees C, practical salinity units, metres, pH).
    sound_velocity_ms : float, optional
        Sound speed (m/s) in the relaxation-amplitude terms. Default: the
        paper's own ``c = 1412 + 3.21 T + 1.19 S + 0.0167 D``.

    Returns
    -------
    float or np.ndarray
        Absorption coefficient in dB per metre (the paper's dB/km / 1000),
        a float for scalar input.
    """
    f_khz = np.asarray(frequency_hz, dtype=np.float64) / 1000.0
    t, s, d = float(temperature_c), float(salinity_psu), float(depth_m)
    c = 1412.0 + 3.21 * t + 1.19 * s + 0.0167 * d if sound_velocity_ms is None else float(sound_velocity_ms)
    kelvin = t + 273.0

    # Boric acid term: Part II (JASA 72(6), 1879-1890), eqs. for A1, P1 (=1), f1
    a1 = 8.86 / c * 10.0 ** (0.78 * ph - 5.0)
    f1 = 2.8 * np.sqrt(s / 35.0) * 10.0 ** (4.0 - 1245.0 / kelvin)
    # Magnesium sulfate term: Part I (JASA 72(3), 896-907), eqs. for A2, P2, f2
    a2 = 21.44 * s / c * (1.0 + 0.025 * t)
    p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d * d
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / kelvin) / (1.0 + 0.0018 * (s - 35.0))
    # Pure-water term: Part I (JASA 72(3), 896-907), eqs. for A3 (two T branches at 20 C), P3
    if t <= 20.0:
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t**2 - 6.5e-10 * t**3
    p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d * d

    f2_sq = f_khz * f_khz
    # Total absorption: Part II (JASA 72(6)), sum of the three terms
    alpha_db_per_km = (
        a1 * f1 * f2_sq / (f1 * f1 + f2_sq)
        + a2 * p2 * f2 * f2_sq / (f2 * f2 + f2_sq)
        + a3 * p3 * f2_sq
    )
    alpha_db_per_m = alpha_db_per_km / 1000.0
    return float(alpha_db_per_m) if alpha_db_per_m.ndim == 0 else alpha_db_per_m


def center_frequency_hz(ds: xr.Dataset) -> float:
    """Acoustic center frequency of a channel (Hz), for frequency-dependent physics.

    The midpoint of ``freq_hz_start`` and ``freq_hz_end`` for a chirp (a
    1495-1605 kHz sweep is centered at 1550 kHz); ``freq_hz_start`` when
    the end is missing, non-positive or equal to the start. The channel
    *label* (``hz_<freq_hz_start>``) is the start frequency and is not
    affected.
    """
    start = float(ds.attrs["freq_hz_start"])
    end = ds.attrs.get("freq_hz_end")
    try:
        end = float(end)
    except (TypeError, ValueError):
        return start
    if not np.isfinite(end) or end <= 0 or end == start:
        return start
    return (start + end) / 2.0


def absorption_for_frequency(frequency_hz, **env):
    """
    Default absorption coefficient (dB/m, one-way) for a sonar frequency.

    Thin wrapper over :func:`francois_garrison_absorption_db_per_m`; ``env``
    (``temperature_c, salinity_psu, depth_m, ph, sound_velocity_ms``) is
    passed through, defaulting to its typical-coastal-water values.
    """
    return francois_garrison_absorption_db_per_m(frequency_hz, **env)


def tvg_gain_db(range_m: np.ndarray, params: TVGParams) -> np.ndarray:
    """
    Per-range TVG gain (dB): spreading loss + absorption loss.

    ``spreading_loss (dB) = spreading_loss_db * log10(range_m)``
    ``absorption_loss (dB) = 2 * absorption_db_per_m * range_m`` (factor of
    2 for the two-way path)

    Shared by ``apply_tvg()`` (xarray survey pipeline) and
    ``apply_tvg_to_amplitude()`` (plain-array path, e.g. target crop
    windows) so both apply the identical formula rather than maintaining
    it twice. Caller is responsible for clipping ``range_m`` to
    ``params.min_range_m`` first (as both callers here do) — this
    function doesn't clip on its own since it has no opinion on what a
    caller's minimum range should be.
    """
    spreading = params.spreading_loss_db * np.log10(range_m)
    absorption = 2.0 * params.absorption_db_per_m * range_m
    return spreading + absorption


def apply_tvg_to_amplitude(
    amplitude: np.ndarray,
    meters_per_sample: float,
    *,
    params: TVGParams | None = None,
    freq_hz: float | None = None,
    sample_idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply TVG correction to a plain linear-amplitude array — no xarray
    ``Dataset`` required.

    For use on data that isn't a full survey channel (e.g. a target's
    raw ping-sample window decoded straight from its catalog SQLite via
    ``sidescanpy.io.jsf.targets.decode_target_window``), where
    constructing a full ``SideScanSurvey``/``Dataset`` just to TVG-correct
    a small crop isn't warranted. Same formula and defaults as
    ``apply_tvg()`` — see that function for the physical justification.

    Parameters
    ----------
    amplitude : np.ndarray
        Linear amplitude, sample axis last (e.g. ``(n_pings, n_samples)``
        or ``(n_samples,)``).
    meters_per_sample : float
        Range covered by one sample (two-way), e.g. a target catalog's
        own ``PingMetersPerSample`` — the per-sample analog of
        ``apply_tvg()``'s ``sound_velocity / (2 * sample_rate)``.
    params : TVGParams, optional
        If ``None``, built from defaults with ``absorption_db_per_m``
        derived from ``freq_hz`` (required in that case, matching
        ``apply_tvg()``'s own default-resolution pattern).
    freq_hz : float, optional
        Used only when ``params`` is ``None`` (or has no
        ``absorption_db_per_m`` set), to derive a default via
        ``absorption_for_frequency()``.
    sample_idx : np.ndarray, optional
        Absolute sample index of each position on the last axis. Defaults
        to ``arange(n_samples)`` (the array starts at the ping's first
        recorded sample); pass it when leading samples were cropped off.

    Returns
    -------
    np.ndarray
        TVG-corrected values in dB, same shape as ``amplitude``.
    """
    if params is None:
        if freq_hz is None:
            raise ValueError(
                "apply_tvg_to_amplitude requires either `params` with "
                "absorption_db_per_m set, or `freq_hz` to derive a default"
            )
        params = TVGParams(absorption_db_per_m=absorption_for_frequency(freq_hz))
    elif params.absorption_db_per_m is None:
        if freq_hz is None:
            raise ValueError(
                "params.absorption_db_per_m is None and no `freq_hz` was "
                "given to derive a default"
            )
        params = TVGParams(
            spreading_loss_db=params.spreading_loss_db,
            absorption_db_per_m=absorption_for_frequency(freq_hz),
            min_range_m=params.min_range_m,
        )

    n_samples = amplitude.shape[-1]
    amplitude_db = linear_to_db(amplitude)
    idx = np.arange(n_samples, dtype=np.float64) if sample_idx is None \
        else np.asarray(sample_idx, dtype=np.float64)
    range_m = idx * meters_per_sample
    range_m = np.clip(range_m, params.min_range_m, None)
    gain_db = tvg_gain_db(range_m, params)
    return amplitude_db + gain_db


def apply_tvg(ds: xr.Dataset, params: TVGParams | None = None) -> xr.Dataset:
    """
    Apply time-varying gain (TVG) correction to ``backscatter_intensity``.

    Compensates for geometric spreading and acoustic absorption losses.
    Converts linear amplitude to dB (``20*log10(clip(x, 1e-6))``, see
    :mod:`sidescanpy.core.units`), adds the TVG gain for each sample's slant
    range, and returns corrected dB values in place of the input.

    Input ``backscatter_intensity`` is assumed to be linear amplitude; the
    library does not check this (what units the input data is in is the
    caller's knowledge). Exact-zero amplitude maps to the -120 dB floor.

    The step is recorded in ``processing_history`` as ``"tvg"`` with the
    parameters actually used (defaults resolved), which is what
    :func:`sidescanpy.processing.correct.invert` reads to undo it.

    Parameters
    ----------
    ds : xr.Dataset
        Frequency channel dataset; needs ``backscatter_intensity``,
        ``sound_vel_ms`` and the ``freq_hz_start``/``sample_rate_hz`` attrs.
    params : TVGParams, optional
        If None, defaults with ``absorption_db_per_m`` derived from
        ``freq_hz_start`` and ``sound_velocity_ms`` from the mean per-ping
        ``sound_vel_ms``. The caller's object is never modified.

    Returns
    -------
    xr.Dataset
        New dataset with TVG-corrected dB ``backscatter_intensity`` and the
        ``"tvg"`` history record appended.

    Raises
    ------
    ValueError
        If TVG has already been applied (prevents double application).
    """
    ds.sidescan.validate()
    if has_step(ds, STEP_TVG):
        raise ValueError("TVG has already been applied to this dataset (found in processing_history).")

    params = resolve_tvg_params(ds, params)
    logger.debug(
        "Applying TVG: freq=%s Hz, spreading=%.1f dB/dec, absorption=%.4f dB/m, "
        "min_range=%.2f m, sound_velocity=%.1f m/s",
        ds.attrs["freq_hz_start"], params.spreading_loss_db, params.absorption_db_per_m,
        params.min_range_m, params.sound_velocity_ms,
    )

    backscatter_db = linear_to_db(ds["backscatter_intensity"])
    # Keep the input dtype (float32): never upcast the big array.
    tvg_db = tvg_gain_for_dataset(ds, params).astype(backscatter_db.dtype)
    corrected_db = backscatter_db + xr.DataArray(tvg_db, dims=["sample"])

    ds_out = ds.copy(deep=False)
    ds_out = ds_out.assign(backscatter_intensity=corrected_db)
    return append_history(
        ds_out, STEP_TVG, tvg_params_record(params),
        function="sidescanpy.processing.radiometric.apply_tvg", version=__version__,
    )


def resolve_tvg_params(ds: xr.Dataset, params: TVGParams | None) -> TVGParams:
    """Fill a TVGParams' unset fields from ``ds`` (new object; input untouched).

    ``absorption_db_per_m=None`` -> Francois-Garrison absorption at the
    channel's center frequency (:func:`center_frequency_hz`); ``sound_velocity_ms=None`` -> mean of the per-ping
    ``sound_vel_ms`` (NaN or non-positive mean falls back to 1500 m/s).
    """
    params = TVGParams() if params is None else replace(params)
    if params.absorption_db_per_m is None:
        params.temperature_c = DEFAULT_TEMPERATURE_C if params.temperature_c is None else params.temperature_c
        params.salinity_psu = DEFAULT_SALINITY_PSU if params.salinity_psu is None else params.salinity_psu
        params.depth_m = DEFAULT_DEPTH_M if params.depth_m is None else params.depth_m
        params.ph = DEFAULT_PH if params.ph is None else params.ph
        params.absorption_db_per_m = francois_garrison_absorption_db_per_m(
            center_frequency_hz(ds), temperature_c=params.temperature_c, salinity_psu=params.salinity_psu,
            depth_m=params.depth_m, ph=params.ph,
        )
        logger.debug(
            "TVG absorption (Francois-Garrison, T=%.1f C, S=%.1f psu, D=%.1f m, pH=%.2f) for %s Hz: %.5f dB/m",
            params.temperature_c, params.salinity_psu, params.depth_m, params.ph,
            center_frequency_hz(ds), params.absorption_db_per_m,
        )
    elif any(v is not None for v in (params.temperature_c, params.salinity_psu, params.depth_m, params.ph)):
        logger.debug("TVG: absorption_db_per_m given explicitly; ignoring water-condition parameters")
        params.temperature_c = params.salinity_psu = params.depth_m = params.ph = None
    if params.sound_velocity_ms is None:
        # TODO: per-ping sound velocity instead of one mean per channel.
        sv = float(ds["sound_vel_ms"].mean().item())
        if np.isnan(sv) or sv <= 0:
            logger.debug("TVG: mean sound_vel_ms is %s; using %s m/s", sv, DEFAULT_SOUND_VEL_MS)
            sv = DEFAULT_SOUND_VEL_MS
        params.sound_velocity_ms = sv
    return params


def tvg_gain_for_dataset(ds: xr.Dataset, params: TVGParams) -> np.ndarray:
    """Per-sample TVG gain (dB, float64, shape ``(n_samples,)``) for ``ds``.

    Slant range of each sample is ``sample_index * c / (2 * fs)`` (absolute
    sample index, so cropped datasets get the same gain per sample), clipped
    to ``params.min_range_m``. ``params`` must be fully resolved
    (see :func:`resolve_tvg_params`).
    """
    interval_m = params.sound_velocity_ms / (2.0 * float(ds.attrs["sample_rate_hz"]))
    slant_range = np.clip(sample_index(ds) * interval_m, params.min_range_m, None)
    return tvg_gain_db(slant_range, params)


def tvg_params_record(params: TVGParams) -> dict:
    """History-record params for a resolved TVGParams."""
    record = {
        "spreading_loss_db": float(params.spreading_loss_db),
        "absorption_db_per_m": float(params.absorption_db_per_m),
        "min_range_m": float(params.min_range_m),
        "sound_velocity_ms": float(params.sound_velocity_ms),
    }
    # Water conditions behind a derived absorption (absent when it was given explicitly).
    for key in ("temperature_c", "salinity_psu", "depth_m", "ph"):
        value = getattr(params, key)
        if value is not None:
            record[key] = float(value)
    return record


def tvg_params_from_record(record_params: dict) -> TVGParams:
    """TVGParams from a ``"tvg"`` history record (current or legacy key names)."""
    sv = record_params.get("sound_velocity_ms", record_params.get("sound_velocity_used_ms"))
    if sv is None:
        raise ValueError(f"TVG history record has no sound velocity: {record_params}")
    return TVGParams(
        spreading_loss_db=float(record_params["spreading_loss_db"]),
        absorption_db_per_m=float(record_params["absorption_db_per_m"]),
        min_range_m=float(record_params.get("min_range_m", 0.5)),
        sound_velocity_ms=float(sv),
    )
