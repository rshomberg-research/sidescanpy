"""
Investigation pipeline orchestration: `run_investigation()`.

Given a path to a sidescan dataset (`.ss.h5`, `.jsf`, or `.sdf`), runs
the five investigation stages (`sidescanpy.analysis`) in
order, saves each stage's standardized plots
(`sidescanpy.viz.investigation`) and a JSON metrics export, and returns
a single `InvestigationResult` object. Meant to be reused directly by
any downstream project's thin CLI wrapper rather than reimplemented
per project.

2026 rshomberg@uri.edu
"""

# FIXME: this is not in the right spot. This a a script and should not really be
# part of the module. Also, I want to get rid of the stages as those do not really make sense.

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.characterize import characterize_values, infer_units
from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.analysis.outliers import compare_outlier_rejection
from sidescanpy.analysis.tvg import fit_range_decay
from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.io.dispatch import detect_format, read_and_convert
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.processing.bottom import blank_nadir_noise, uniform_exclusion_boundary
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.inversion import LEGACY_RAW_VAR, UNCORRECTED_VAR, invert
from sidescanpy.viz.investigation import (
    plot_stage1_5_before_after,
    plot_stage1_histograms,
    plot_stage1_sample_ping,
    plot_stage2_5_beam_geometry,
    plot_stage2_cross_track_profile,
    plot_stage2_nadir_zoom,
    plot_stage2_waterfall,
    plot_stage3_usable_range,
    plot_stage4_decay_fit,
    plot_stage4_residuals,
    plot_stage5_distribution,
)

logger = logging.getLogger(__name__)

_HDF5_SUFFIXES = (".h5", ".hdf5")


@dataclass
class InvestigationResult:
    """Everything one `run_investigation()` call produced for one
    frequency channel."""

    source_path: str
    freq_key: str
    freq_hz: float
    n_pings: int
    value_var: str
    params: dict[str, Any]
    stages: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "freq_key": self.freq_key,
            "freq_hz": self.freq_hz,
            "n_pings": self.n_pings,
            "value_var": self.value_var,
            "params": self.params,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    def save_json(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    def summary_row(self) -> dict[str, Any]:
        """One flat dict per source, matching the handoff's comparison-table columns."""
        s1, s2, s3, s4, s5 = (self.stages.get(f"stage{i}") for i in range(1, 6))
        s2_5 = self.stages.get("stage2_5")
        return {
            "source": self.source_path,
            "freq_key": self.freq_key,
            "dtype": s1.dtype if s1 else None,
            "nadir": s2.nadir_type if s2 else None,
            "beam_geometry": s2_5.confidence if s2_5 else None,
            "near_angle_deg": s2_5.near_angle_deg_median if s2_5 else None,
            "beam_width_deg": s2_5.beam_width_deg if s2_5 else None,
            "usable_range_m": s3.usable_range_m if s3 else None,
            "best_fit_model": s4.best_candidate if s4 else None,
            "tvg_status": s4.tvg_status if s4 else None,
            "units": s5.units if s5 else None,
        }


def _load_channel_dataset(dataset_path: Path, freq_key: str | None):
    if dataset_path.suffix.lower() in _HDF5_SUFFIXES or dataset_path.name.endswith(".ss.h5"):
        survey = read_survey(dataset_path)
    else:
        fmt = detect_format([dataset_path])
        survey = read_and_convert([dataset_path], fmt)

    freq_key = freq_key or sorted(survey.frequencies)[0]
    if freq_key not in survey.frequencies:
        raise ValueError(f"{freq_key!r} not found. Available: {survey.frequencies}")
    return survey.get_frequency(freq_key).dataset, freq_key


def run_investigation(
    dataset_path: str | Path,
    *,
    freq_key: str | None = None,
    channel: int | str | None = None,
    value_var: str = "backscatter_intensity",
    ping_range: tuple[int, int | None] = (0, None),
    output_dir: str | Path | None = None,
    nadir_n_samples: int = 50,
    nadir_margin_samples: int = 0,
    nadir_min_extent_samples: float = 0.0,
    strong_return_min_sample_buffer: float = 0.0,
    strong_return_peak_percentile: float = 100.0,
    min_first_bottom_return_sample: float | None = None,
    stage3_bin_width_m: float = 2.0,
    stage4_bin_width_m: float = 5.0,
    stage4_min_bins: int = 6,
    stage4_db_reference_percentile: float = 99.0,
    snr_threshold: float = 3.0,
    exclude_nadir_gap: bool = True,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
    stage4_fit_value_col: str = "mean",
    stage4_r_ref_m: float | str | None = None,
    stage4_r_fit_min_m: float | str | None = None,
    stage4_r_fit_min_auto_margin_m: float = 5.0,
    stage4_r_ref_search_fraction: float = 0.5,
    stage4_r_ref_window_fraction: float = 1.0 / 3.0,
    stage4_exclude_first_n_bins: int = 0,
) -> InvestigationResult:
    """
    Run all five investigation stages against one frequency channel.

    Parameters
    ----------
    dataset_path : str or Path
        A `.ss.h5`, `.jsf`, or `.sdf` file.
    freq_key : str, optional
        e.g. ``"hz_1495000"``. Defaults to the first (sorted) frequency
        in the survey. Note this is a *frequency* channel (e.g. HF vs.
        LF) — a different axis from `channel` below (port vs.
        starboard); the two are independent and this function's own
        "one frequency channel" phrasing above refers only to this one.
    channel : int or str, optional
        Restrict **Stage 3 and Stage 4's own binning/fit** to one
        physical (port/starboard) channel — `0`/`1` by position, or
        `"port"`/`"starboard"` by `ds["channel_id"]` value. `None`
        (default) pools every channel's samples together, every existing
        caller's exact prior behavior, unchanged. Real motivation
        (2026-09-02): pooling is invisible until you look for it — port
        and starboard transducers can have genuinely different gain
        characteristics, and `usable_range_m`/a fitted decay curve/TVG
        classification built from a silent blend of both is a different
        (and less informative) result than either side's own, with
        nothing in the output signaling that a blend happened.

        **Stages 1, 1.5, 2, 2.5, and 5 are NOT restricted by this** —
        they keep seeing the full, un-sliced `ds` regardless. This isn't
        an oversight: the sidescan schema hard-requires the `channel`
        dim be exactly size 2 everywhere in this pipeline
        (`SideScanAccessor.validate()`, called by e.g.
        `compute_geometric_ranges`), and `estimate_beam_geometry`
        specifically hardcodes a `(0, "port"), (1, "starboard")`
        iteration regardless of `validate()` — those stages already
        report both channels' own geometry independently in their own
        metrics (see `estimate_beam_geometry`'s `near_ray`/`far_ray`),
        so narrowing `ds` itself for them would break real invariants
        for no benefit. Only Stage 3/4's `backscatter_vs_range`-
        based binning actually pools across `channel` via a flat
        `.ravel()`, which is the specific thing this parameter exists to
        fix. Call `run_investigation()` once per side (typically both)
        for a genuine per-channel Stage 3/4 comparison — this function
        itself still only ever returns one `InvestigationResult`.
    value_var : str, default "backscatter_intensity"
        Variable to investigate. Every recorded correction is inverted
        first (``sidescanpy.processing.invert``), so the default is the
        uncorrected signal. ``"uncorrected"`` (and the legacy name
        ``"backscatter_raw"``, when the dataset has no such variable) map to it.
    ping_range : (int, int | None), default (0, None)
        Ping slice — investigating a full multi-thousand-ping survey is
        rarely necessary and slower than a representative subset.
    output_dir : str or Path, optional
        If given, every stage's PNGs plus ``metrics.json`` are written
        there (one subdirectory per stage).
    nadir_n_samples, stage3_bin_width_m, stage4_bin_width_m,
    stage4_min_bins, stage4_db_reference_percentile, snr_threshold : reproducibility parameters
        Passed straight through to the corresponding stage functions —
        logged into the result's ``params`` and the JSON export so a
        re-run can reproduce them exactly.
        ``stage4_min_bins`` (default 6) shrinks ``stage4_bin_width_m``
        automatically when a short ``usable_range_m`` would otherwise
        leave too few bins in the decay-fit window to reliably
        distinguish the 4 candidate models (see
        `analysis.tvg.fit_range_decay`'s own ``min_bins``
        docstring for the real-data case — a survey flown very close to
        the bottom, ~15m usable range, collapsing to 2-3 bins).
        ``stage4_db_reference_percentile`` sets the self-referenced dB
        scale Stage 4 converts linear amplitude to (see
        `analysis.tvg.fit_range_decay`'s docstring) — provably
        does not affect Stage 4's classification, only what an
        intermediate absolute dB number would mean if reported
        directly.
    Before any stage runs, `processing.bottom.blank_nadir_noise()`
    always applies to `backscatter_intensity` (a
    real, no-toggle preprocessing step, like `invert`/
    `compute_geometric_ranges` above — not a debatable option). Zeros
    each ping's leading electronic-noise burst right at nadir — a
    no-op on hardware that already does this itself (EdgeTech/JSF,
    confirmed), but necessary on hardware that doesn't (Klein/SDF,
    confirmed on real data: this burst is bright and sustained enough
    to otherwise fool Stage 2's first-bottom-return detector into
    reporting sample 0 as the real bottom, on nearly every ping).
    nadir_margin_samples : int, default 0
        Passed to `blank_nadir_noise` as `margin_samples`. 0 preserves
        exact prior behavior on JSF/SDF (already validated clean at
        0). Set nonzero when `first_bottom_return_sample` visibly
        alternates between the altitude floor and a plausible real
        value across pings on a new data source — found necessary on
        one real survey: a handful of bright points
        right at the blanked boundary's edge weren't fully zeroed and
        were sustained/bright enough to fool Stage 2 into locking onto
        them. See `strong_return_min_sample_buffer`/
        `strong_return_peak_percentile` below for the second,
        complementary line of defense against the same failure mode.
    nadir_min_extent_samples : float, default 0.0
        Passed to `blank_nadir_noise` as `min_extent_samples` — a hard
        floor on the blanked nadir-noise extent, applied even where
        `detect_nadir_noise_extent` finds nothing (unlike
        `nadir_margin_samples`, which only widens an actual detection).
        0.0 preserves exact prior behavior. Use when a channel/survey
        is known to always carry at least N samples of nadir junk
        regardless of per-ping detection.
    strong_return_min_sample_buffer, strong_return_peak_percentile
        Passed to `analysis.nadir.analyze_nadir` (which passes
        them to `detect_first_bottom_return` as `min_sample_buffer`/
        `peak_percentile`) — see those functions' docstrings. Both
        default to no-ops (0.0 / 100.0); like `nadir_margin_samples`
        above, only found necessary so far on one real survey, not
        on any JSF/SDF survey checked.
    min_first_bottom_return_sample : float, optional
        Passed to `analysis.nadir.analyze_nadir` — an explicit
        floor on `first_bottom_return_sample`, independent of the
        altitude-derived one, combined via element-wise max. `None`
        (default) preserves exact prior behavior. See that function's
        docstring for when the altitude floor alone isn't enough.
    exclude_nadir_gap : bool, default True
        Stage 2's `first_bottom_return_sample` (an amplitude-based
        detector — see `analysis.nadir.analyze_nadir`'s
        docstring) is used as an early masking step before Stage 3/4's
        binning, at **two different granularities for two different
        reasons** (real-data checking drove both
        choices, not just the first):
        - **Stage 3** gets the exact **per-ping** boundary directly —
          "remove every sample that's nadir gap, run the analysis only
          on actual bottom hits," as literally as possible. A single
          survey-wide cutoff necessarily under-masks whichever pings'
          true boundary exceeds it; real data showed this leaves a
          thin, contaminated near-range bin (an anomalous near-zero
          first point on the usable-range plot) — exact per-ping
          masking removes it entirely. Stage 3 does no aggregate
          max-search across bins, so per-ping precision here has no
          downside.
        - **Stage 4** gets a single conservative value from
          `processing.bottom.uniform_exclusion_boundary` instead.
          Its reference-range auto-selection *does* do a max-mean
          search across bins, which per-ping masking exposed to a real
          altitude-correlated bias (shallower-altitude pings'
          genuinely stronger near-range returns concentrate in
          whichever bin only they still contribute to, once
          deeper-altitude pings are masked further out) — see that
          function's own docstring for the full mechanism and the
          real-data numbers this produced before the fix.
        On by default — unlike `reject_outliers` below, this isn't a
        debatable aggressiveness tradeoff, it's excluding samples that
        real-data checking confirmed are not yet flat-bottom-like
        (one real survey: altitude sample ~140, real first bottom
        return doesn't start until ~300-330 — the beam's near edge is
        never at 0 deg from vertical, so this span is never
        illuminated at all, not weak-but-real backscatter). Set False
        to reproduce the older behavior (altitude-crossing exclusion
        only, both stages).
    reject_outliers : bool, default False
        If True, Stage 3's noise-floor estimate and Stage 4's decay fit
        both run `processing.outliers.reject_range_outliers` first —
        aggressive, per-range-bin MAD-based rejection meant to strip
        point-target returns (boulders, debris, wrecks) and complex
        terrain so the noise floor / decay fit reflect a flat,
        sandy-bottom-like population instead of being pulled around by
        real but calibration-irrelevant returns. Off by default — see
        `reject_range_outliers`'s own docstring for why, including why
        this is the wrong default for anything display/export-facing
        (those bright returns are often genuine targets worth keeping).
    outlier_mad_k, outlier_min_bin_count
        Passed through to `reject_range_outliers` when
        `reject_outliers=True`; ignored otherwise.
    stage4_fit_value_col : {"mean", "median"}, default "mean"
        Passed through to `analysis.tvg.fit_range_decay` — pass
        `"median"` alongside `reject_outliers=True`, see that function's
        docstring for why (real-data check found the `"mean"` pairing
        makes the fit measurably worse, not better).
    stage4_r_ref_m, stage4_r_fit_min_m, stage4_r_fit_min_auto_margin_m,
    stage4_r_ref_search_fraction, stage4_r_ref_window_fraction
        Passed straight through to `analysis.tvg.fit_range_decay`
        as `r_ref_m`/`r_fit_min_m`/`r_fit_min_auto_margin_m`/
        `r_ref_search_fraction`/`r_ref_window_fraction` — **that
        function's own docstrings for both `r_ref_m` and `r_fit_min_m`
        are the source of truth for how these interact; read them
        before reaching for these, don't infer from the field names
        alone.** Short version: `stage4_r_fit_min_m="auto"` derives the
        fit window's near edge from the raw data's own near-field peak
        plus a margin (`stage4_r_fit_min_auto_margin_m`) instead of a
        hand-picked number, since a fixed value picked by eye on one
        mission doesn't generalize across a dataset whose near-field
        ramp varies mission to mission. `stage4_r_ref_m="auto"` (the
        recommended pairing) then places the reference *inside* that
        resolved window, `stage4_r_ref_window_fraction` (default 1/3)
        of the way from its near to far edge — resolved in that order
        specifically so the reference can't land on the window's own
        (often noisy) boundary, a real bug in the older
        `stage4_r_ref_m="mid_swath"` design (still supported, legacy)
        that pins `usable_range_m/2` independent of the window and can
        collide with it. Both string modes fall back gracefully (with a
        note, not a raise) when there isn't enough known to resolve
        them — safe to use as blanket defaults across missions/sections
        that won't all have a determinable `usable_range_m`.
    stage4_exclude_first_n_bins : int, default 0
        Passed straight through to `fit_range_decay` — drops this many
        *bins* (not samples) from the near end of the range-binned decay
        curve before `r_ref_m` auto-selection or fitting see them, for a
        bright leading-edge-specular artifact right at the first-bottom-
        return boundary (distinct from the ring-down artifact at a
        ping's very start, already handled elsewhere). `0` (default)
        is a no-op.

    Returns
    -------
    InvestigationResult
    """
    dataset_path = Path(dataset_path)
    ds, freq_key = _load_channel_dataset(dataset_path, freq_key)

    if value_var in (UNCORRECTED_VAR, LEGACY_RAW_VAR) and value_var not in ds:
        logger.debug("value_var %r not present; using the inverted backscatter_intensity", value_var)
        value_var = "backscatter_intensity"

    start, stop = ping_range
    ds = ds.isel(ping=slice(start, stop))
    if getattr(ds[value_var].data, "chunks", None) is not None:
        ds = ds.load()

    ds = invert(ds)  # investigate the uncorrected signal; no-op on uncorrected data

    ds = blank_nadir_noise(ds, margin_samples=nadir_margin_samples, min_extent_samples=nadir_min_extent_samples)

    if "ground_range_m" not in ds:
        ds = compute_geometric_ranges(ds)

    # `channel` is NOT applied by slicing `ds` here -- the sidescan
    # schema hard-requires the channel dim be exactly size 2 everywhere
    # in this pipeline (SideScanAccessor.validate(), a real, deliberate
    # invariant: sidescan sonar always has port+starboard), and several
    # stage functions (e.g. estimate_beam_geometry) hardcode a 2-channel
    # iteration regardless of validate(). `ds` keeps its full shape
    # through every stage; only estimate_usable_range/fit_range_decay (the
    # two whose own binning pools samples via `.ravel()`, the actual
    # thing `channel` exists to restrict) receive it directly below and
    # mask internally instead of slicing.
    params = {
        "channel": channel,
        "value_var": value_var,
        "ping_range": list(ping_range),
        "nadir_n_samples": nadir_n_samples,
        "nadir_margin_samples": nadir_margin_samples,
        "nadir_min_extent_samples": nadir_min_extent_samples,
        "strong_return_min_sample_buffer": strong_return_min_sample_buffer,
        "strong_return_peak_percentile": strong_return_peak_percentile,
        "min_first_bottom_return_sample": min_first_bottom_return_sample,
        "stage3_bin_width_m": stage3_bin_width_m,
        "stage4_bin_width_m": stage4_bin_width_m,
        "stage4_min_bins": stage4_min_bins,
        "stage4_db_reference_percentile": stage4_db_reference_percentile,
        "snr_threshold": snr_threshold,
        "exclude_nadir_gap": exclude_nadir_gap,
        "reject_outliers": reject_outliers,
        "outlier_mad_k": outlier_mad_k,
        "outlier_min_bin_count": outlier_min_bin_count,
        "stage4_fit_value_col": stage4_fit_value_col,
        "stage4_r_ref_m": stage4_r_ref_m,
        "stage4_r_fit_min_m": stage4_r_fit_min_m,
        "stage4_r_fit_min_auto_margin_m": stage4_r_fit_min_auto_margin_m,
        "stage4_r_ref_search_fraction": stage4_r_ref_search_fraction,
        "stage4_r_ref_window_fraction": stage4_r_ref_window_fraction,
        "stage4_exclude_first_n_bins": stage4_exclude_first_n_bins,
    }

    stage1 = characterize_values(ds, value_var=value_var)
    stage1_5 = compare_outlier_rejection(
        ds, value_var=value_var, bin_width_m=stage3_bin_width_m,
        outlier_mad_k=outlier_mad_k, outlier_min_bin_count=outlier_min_bin_count,
    )
    stage2 = analyze_nadir(
        ds, value_var=value_var, nadir_n_samples=nadir_n_samples,
        strong_return_min_sample_buffer=strong_return_min_sample_buffer,
        strong_return_peak_percentile=strong_return_peak_percentile,
        min_first_bottom_return_sample=min_first_bottom_return_sample,
    )
    exclude_before_sample_stage3 = None
    exclude_before_sample_stage4 = None
    if exclude_nadir_gap:
        per_ping_boundary = stage2.first_bottom_return_sample
        if per_ping_boundary is not None:
            # Deliberately different granularity per stage -- see
            # run_investigation's docstring and
            # uniform_exclusion_boundary()'s own docstring for why one
            # boundary doesn't serve both purposes well:
            #
            # Stage 3 gets the exact PER-PING boundary. A uniform
            # (percentile-based) cutoff necessarily under-masks
            # whichever pings' true boundary exceeds that percentile --
            # real-data check found this leaves a thin, contaminated
            # near-range bin (leftover nadir-gap samples from the
            # under-masked tail), visible as an anomalous near-zero
            # first point on the usable-range plot. Stage 3 doesn't do
            # any aggregate max-search across bins, so per-ping
            # precision here has no downside.
            #
            # Stage 4 keeps the uniform boundary. Its reference-range
            # auto-selection *does* do a max-mean search across bins,
            # which per-ping masking exposed to real altitude-correlated
            # bias (shallower-altitude pings' genuinely stronger returns
            # concentrate in whichever near bin only they contribute
            # to) -- see uniform_exclusion_boundary()'s docstring.
            exclude_before_sample_stage3 = per_ping_boundary
            exclude_before_sample_stage4 = uniform_exclusion_boundary(per_ping_boundary)
    stage3 = estimate_usable_range(
        ds, value_var=value_var, bin_width_m=stage3_bin_width_m, snr_threshold=snr_threshold,
        exclude_before_sample=exclude_before_sample_stage3,
        reject_outliers=reject_outliers, outlier_mad_k=outlier_mad_k, outlier_min_bin_count=outlier_min_bin_count,
        channel=channel,
    )
    # Stage 2.5 runs after Stage 3, not before -- its far edge is
    # triangulated from stage3's usable_range_m, so it needs that
    # result already computed.
    stage2_5 = estimate_beam_geometry(ds, stage2.first_bottom_return_sample, stage3.usable_range_m)
    stage4 = fit_range_decay(
        ds, value_var=value_var, bin_width_m=stage4_bin_width_m, min_bins=stage4_min_bins,
        db_reference_percentile=stage4_db_reference_percentile,
        usable_range_m=stage3.usable_range_m,
        exclude_before_sample=exclude_before_sample_stage4,
        reject_outliers=reject_outliers, outlier_mad_k=outlier_mad_k, outlier_min_bin_count=outlier_min_bin_count,
        fit_value_col=stage4_fit_value_col,
        r_ref_m=stage4_r_ref_m, r_fit_min_m=stage4_r_fit_min_m,
        r_fit_min_auto_margin_m=stage4_r_fit_min_auto_margin_m,
        r_ref_search_fraction=stage4_r_ref_search_fraction,
        r_ref_window_fraction=stage4_r_ref_window_fraction,
        exclude_first_n_bins=stage4_exclude_first_n_bins,
        channel=channel,
    )
    stage5 = infer_units(stage1, stage4)

    result = InvestigationResult(
        source_path=str(dataset_path),
        freq_key=freq_key,
        freq_hz=float(ds.attrs["freq_hz_start"]),
        n_pings=int(ds.sizes["ping"]),
        value_var=value_var,
        params=params,
        stages={
            "stage1": stage1, "stage1_5": stage1_5, "stage2": stage2, "stage2_5": stage2_5,
            "stage3": stage3, "stage4": stage4, "stage5": stage5,
        },
    )

    if output_dir is not None:
        _save_report_artifacts(ds, result, Path(output_dir))

    return result


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_report_artifacts(ds, result: InvestigationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    value_var = result.value_var

    s1_dir = output_dir / "stage1"
    s1_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage1_histograms(ds, value_var)
    _save_fig(fig, s1_dir / "histograms.png")
    fig, _ = plot_stage1_sample_ping(ds, value_var)
    _save_fig(fig, s1_dir / "sample_ping.png")

    s1_5_dir = output_dir / "stage1_5"
    s1_5_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage1_5_before_after(ds, result.stages["stage1_5"], value_var)
    _save_fig(fig, s1_5_dir / "before_after.png")

    s2_dir = output_dir / "stage2"
    s2_dir.mkdir(exist_ok=True)
    stage2_result = result.stages["stage2"]
    nadir_n = stage2_result.nadir_n_samples
    fig, _ = plot_stage2_cross_track_profile(ds, value_var)
    _save_fig(fig, s2_dir / "cross_track_profile.png")
    fig, _ = plot_stage2_nadir_zoom(ds, value_var, stage2=stage2_result, nadir_n_samples=nadir_n)
    _save_fig(fig, s2_dir / "nadir_zoom.png")
    fig, _ = plot_stage2_waterfall(ds, value_var, stage2=stage2_result)
    _save_fig(fig, s2_dir / "waterfall.png")

    s2_5_dir = output_dir / "stage2_5"
    s2_5_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage2_5_beam_geometry(ds, result.stages["stage2_5"], value_var)
    _save_fig(fig, s2_5_dir / "beam_geometry.png")

    s3_dir = output_dir / "stage3"
    s3_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage3_usable_range(result.stages["stage3"])
    _save_fig(fig, s3_dir / "usable_range.png")

    s4_dir = output_dir / "stage4"
    s4_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage4_decay_fit(result.stages["stage4"])
    _save_fig(fig, s4_dir / "decay_fit.png")
    fig, _ = plot_stage4_residuals(result.stages["stage4"])
    _save_fig(fig, s4_dir / "residuals.png")

    s5_dir = output_dir / "stage5"
    s5_dir.mkdir(exist_ok=True)
    fig, _ = plot_stage5_distribution(ds, value_var)
    _save_fig(fig, s5_dir / "distribution.png")

    result.save_json(output_dir / "metrics.json")
    logger.debug("Investigation artifacts written to %s", output_dir)
