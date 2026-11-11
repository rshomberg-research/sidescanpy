"""
Run-everything driver behind ``sidescan-analyze investigate``.

Loads one frequency channel, runs every analysis in
:mod:`sidescanpy.analysis` on it, and writes ``metrics.json`` plus (unless
``plots=False``) one PNG per plot. This is a run-everything workflow, so it
lives with the CLI, not in the library; the *Investigating an unknown
survey* example walks through the same flow as a readable script. Only
:func:`save_figures` needs matplotlib.

2026 rshomberg@uri.edu
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import xarray as xr

from sidescanpy.analysis import (
    analyze_nadir,
    characterize_values,
    check_tvg,
    compare_outlier_rejection,
    estimate_beam_geometry,
    estimate_usable_range,
    fit_range_decay,
    fit_tvg_models,
    infer_units,
)
from sidescanpy.io import load_survey
from sidescanpy.processing import (
    LEGACY_RAW_VAR,
    UNCORRECTED_VAR,
    blank_nadir_noise,
    center_frequency_hz,
    compute_geometric_ranges,
    invert,
    uniform_exclusion_boundary,
)

logger = logging.getLogger(__name__)

_DPI = 150


def list_frequencies(path: Path) -> list[str]:
    """Sorted freq_keys in a ``.ss.h5`` / ``.jsf`` / ``.sdf`` / ``.xtf`` file."""
    return sorted(load_survey(path).frequencies)


def load_frequency(path: Path, freq_key: str | None = None) -> tuple[xr.Dataset, str]:
    """One frequency channel's dataset (default: first sorted freq_key)."""
    survey = load_survey(path)
    freq_key = freq_key or sorted(survey.frequencies)[0]
    if freq_key not in survey.frequencies:
        raise ValueError(f"{freq_key!r} not found. Available: {survey.frequencies}")
    return survey.get_frequency(freq_key).dataset, freq_key


def investigate(
    ds: xr.Dataset,
    *,
    value_var: str = "backscatter_intensity",
    ping_range: tuple[int, int | None] = (0, None),
    channel: int | str | None = None,
    nadir_n_samples: int = 50,
    usable_range_bin_width_m: float = 2.0,
    usable_range_method: str = "knee",
    snr_threshold: float = 3.0,
    tvg_bin_width_m: float = 1.0,
    start_margin_m: float = 5.0,
    knee_db: float = 3.0,
    window_m: tuple[float, float] | None = None,
    decay_bin_width_m: float = 5.0,
    decay_db_reference_percentile: float = 99.0,
    decay_fit_value_col: str = "mean",
    exclude_nadir_gap: bool = True,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
) -> tuple[xr.Dataset, str, dict[str, Any], dict[str, Any]]:
    """Run every analysis on one channel.

    Returns ``(ds, value_var, params, results)``: the prepared dataset
    (ping-sliced, inverted, nadir noise blanked, geometric ranges added),
    the variable analyzed, the parameters used, and the result objects
    keyed by name. ``tvg_fit`` is ``{"fit": TVGFitResult, "check":
    TVGCheckResult or None}`` (the TVG to apply, and the data after it);
    ``range_decay`` is the deprecated classifier, kept in the output for
    one release. ``channel`` restricts the usable-range, TVG-fit and
    range-decay binning to one side (port/starboard); ``None`` pools both.
    """
    if value_var in (UNCORRECTED_VAR, LEGACY_RAW_VAR) and value_var not in ds:
        # invert() below turns backscatter_intensity into the uncorrected signal.
        value_var = "backscatter_intensity"

    start, stop = ping_range
    ds = ds.isel(ping=slice(start, stop))
    if getattr(ds[value_var].data, "chunks", None) is not None:
        ds = ds.load()
    ds = invert(ds)
    ds = blank_nadir_noise(ds)
    if "ground_range_m" not in ds:
        ds = compute_geometric_ranges(ds)

    params = {
        "value_var": value_var,
        "ping_range": list(ping_range),
        "channel": channel,
        "nadir_n_samples": nadir_n_samples,
        "usable_range_bin_width_m": usable_range_bin_width_m,
        "usable_range_method": usable_range_method,
        "snr_threshold": snr_threshold,
        "tvg_bin_width_m": tvg_bin_width_m,
        "start_margin_m": start_margin_m,
        "knee_db": knee_db,
        "window_m": list(window_m) if window_m is not None else None,
        "decay_bin_width_m": decay_bin_width_m,
        "decay_db_reference_percentile": decay_db_reference_percentile,
        "decay_fit_value_col": decay_fit_value_col,
        "exclude_nadir_gap": exclude_nadir_gap,
        "reject_outliers": reject_outliers,
        "outlier_mad_k": outlier_mad_k,
    }
    outliers = dict(reject_outliers=reject_outliers, outlier_mad_k=outlier_mad_k)

    values = characterize_values(ds, value_var=value_var)
    outlier_comparison = compare_outlier_rejection(
        ds, value_var=value_var, bin_width_m=usable_range_bin_width_m, outlier_mad_k=outlier_mad_k,
    )
    nadir = analyze_nadir(ds, value_var=value_var, nadir_n_samples=nadir_n_samples)

    # Usable range masks the nadir gap per ping; the decay fit uses one
    # survey-wide boundary because its reference search is biased by
    # per-ping masking (see uniform_exclusion_boundary).
    per_ping = nadir.first_bottom_return_sample if exclude_nadir_gap else None
    uniform = uniform_exclusion_boundary(per_ping) if per_ping is not None else None

    usable_range = estimate_usable_range(
        ds, value_var=value_var, method=usable_range_method, bin_width_m=usable_range_bin_width_m,
        snr_threshold=snr_threshold, knee_db=knee_db, exclude_before_sample=per_ping, channel=channel, **outliers,
    )
    beam_geometry = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable_range.usable_range_m)
    tvg_fit = fit_tvg_models(
        ds, value_var=value_var, bin_width_m=tvg_bin_width_m, start_margin_m=start_margin_m, knee_db=knee_db,
        window_m=window_m, channel=channel,
    )
    tvg_check = None
    if tvg_fit.best is not None:
        tvg_check = check_tvg(ds, tvg_fit.tvg_params, tvg_fit.window_m, value_var=value_var,
                              bin_width_m=tvg_bin_width_m, channel=channel)
    # Deprecated classifier, kept in metrics.json for one release.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        range_decay = fit_range_decay(
            ds, value_var=value_var, bin_width_m=decay_bin_width_m,
            db_reference_percentile=decay_db_reference_percentile, usable_range_m=usable_range.usable_range_m,
            exclude_before_sample=uniform, fit_value_col=decay_fit_value_col, channel=channel, **outliers,
        )
    units = infer_units(values, tvg_fit)

    results = {
        "values": values,
        "outlier_comparison": outlier_comparison,
        "nadir": nadir,
        "beam_geometry": beam_geometry,
        "usable_range": usable_range,
        "tvg_fit": {"fit": tvg_fit, "check": tvg_check},
        "range_decay": range_decay,
        "units": units,
    }
    return ds, value_var, params, results


def save_figures(ds: xr.Dataset, value_var: str, results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render every plot for ``results`` into ``output_dir``; returns the paths."""
    import matplotlib.pyplot as plt

    from sidescanpy.viz.analysis import (
        plot_beam_geometry,
        plot_cross_track_profile,
        plot_nadir_waterfall,
        plot_nadir_zoom,
        plot_outlier_comparison,
        plot_sample_ping,
        plot_units_distribution,
        plot_usable_range,
        plot_value_histograms,
    )
    from sidescanpy.viz.tvg import plot_tvg_fit

    output_dir.mkdir(parents=True, exist_ok=True)
    nadir = results["nadir"]
    figures = {
        "value_histograms": plot_value_histograms(ds, value_var),
        "sample_ping": plot_sample_ping(ds, value_var),
        "outlier_comparison": plot_outlier_comparison(ds, results["outlier_comparison"], value_var),
        "cross_track_profile": plot_cross_track_profile(ds, value_var),
        "nadir_zoom": plot_nadir_zoom(ds, value_var, nadir=nadir, nadir_n_samples=nadir.nadir_n_samples),
        "nadir_waterfall": plot_nadir_waterfall(ds, value_var, nadir=nadir),
        "beam_geometry": plot_beam_geometry(ds, results["beam_geometry"], value_var),
        "usable_range": plot_usable_range(results["usable_range"]),
        "tvg_fit": plot_tvg_fit(results["tvg_fit"]["fit"], results["tvg_fit"]["check"]),
        "units_distribution": plot_units_distribution(ds, value_var),
    }
    paths = []
    for name, (fig, _) in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def metrics_dict(
    source_path: Path, freq_key: str, ds: xr.Dataset, value_var: str, params: dict, results: dict
) -> dict[str, Any]:
    """The ``metrics.json`` content: run context plus every result's ``to_dict()``."""
    return {
        "source_path": str(source_path),
        "freq_key": freq_key,
        "freq_hz": float(ds.attrs["freq_hz_start"]),
        "center_freq_hz": center_frequency_hz(ds),
        "n_pings": int(ds.sizes["ping"]),
        "value_var": value_var,
        "params": params,
        "results": {name: _to_dict(result) for name, result in results.items()},
    }


def _to_dict(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        return {key: _to_dict(value) for key, value in result.items()}
    return result.to_dict()


def summary_row(source_path: Path, freq_key: str, results: dict[str, Any]) -> dict[str, Any]:
    """One flat comparison-table row per source and frequency."""
    beam = results["beam_geometry"]
    fit, check = results["tvg_fit"]["fit"], results["tvg_fit"]["check"]
    return {
        "source": str(source_path),
        "freq_key": freq_key,
        "dtype": results["values"].dtype,
        "nadir": results["nadir"].nadir_type,
        "beam_geometry": beam.confidence,
        "near_angle_deg": beam.near_angle_deg_median,
        "beam_width_deg": beam.beam_width_deg,
        "usable_range_m": results["usable_range"].usable_range_m,
        "tvg_model": fit.best,
        "tvg_rms_db": fit.best_model.rms_db if fit.best_model else None,
        "tvg_spread_after_db": check.spread_db if check is not None else None,
        "tvg_status": fit.tvg_status,
        "units": results["units"].units,
    }


def run(input_path: Path, freq_key: str, output_dir: Path, *, plots: bool = True, **kwargs) -> dict[str, Any]:
    """Investigate one frequency of ``input_path``, write ``metrics.json`` (and
    the PNGs when ``plots``) under ``output_dir``, and return the summary row."""
    raw, freq_key = load_frequency(input_path, freq_key)
    ds, value_var, params, results = investigate(raw, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plots:
        save_figures(ds, value_var, results, output_dir)
    metrics = metrics_dict(input_path, freq_key, ds, value_var, params, results)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.debug("Investigation artifacts written to %s", output_dir)
    return summary_row(input_path, freq_key, results)
