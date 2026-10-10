"""
Unit tests for sidescanpy.analysis.tvg.fit_range_decay.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.analysis.tvg import _auto_near_edge_m, fit_range_decay
from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, absorption_for_frequency, tvg_gain_db
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestFitRangeDecay:
    def test_full_physics_curve_classified_as_none(self):
        """Raw data that follows the uncompensated physics model
        end-to-end should classify as TVG status 'none'."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)

        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)

        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        result = fit_range_decay(ds, bin_width_m=2.0)

        assert result.tvg_status == "none"
        assert result.best_candidate == "full_physics"

    def _make_full_physics_ds(self, n_pings=60, n_samples=300):
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)
        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)
        return ds

    def test_short_usable_range_auto_shrinks_bin_width_to_meet_min_bins(self):
        """Real-data motivation (2026-08-25): a survey flown very close
        to the bottom, with usable_range_m as short as ~15m, left the
        default bin_width_m=5.0 with only 2-3 bins in the fit window --
        not enough to reliably distinguish 4 competing single-parameter
        physics models. min_bins (default 6) now shrinks bin_width_m
        automatically for a short usable_range_m."""
        ds = self._make_full_physics_ds()

        # range_var pinned to ground_range_m: this test's numbers (and the
        # real report it reproduces) predate fit_range_decay's 2026-09-02
        # range_var default change to slant_range_m -- pinning keeps this
        # regression test decoupled from that unrelated default.
        adapted = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=5.0, usable_range_m=15.0)
        unadapted = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=5.0, usable_range_m=15.0, min_bins=0)

        assert adapted.bin_width_m_requested == 5.0
        assert adapted.bin_width_m == pytest.approx(15.0 / 6)
        assert adapted.bin_width_m < 5.0
        assert any("bin_width_m auto-reduced" in n for n in adapted.notes)
        # The exact fit-window bin count depends on where r_ref_m lands
        # (see min_bins's own docstring -- the guarantee targets the full
        # [0, usable_range_m] span, not the fit window specifically), but
        # it must be a real, substantial improvement over the unadapted
        # 5.0m-bin case this real report was about.
        assert adapted.n_bins_used > unadapted.n_bins_used
        assert unadapted.n_bins_used <= 3  # reproduces the reported "only 3 bins"
        assert adapted.n_bins_used >= 4

    def test_wide_usable_range_leaves_bin_width_unchanged(self):
        """The default bin_width_m=5.0 already gives well over min_bins
        bins across a normal 40-70m-range survey -- confirms the
        adaptation is a floor, not a universal override, and existing
        callers with plenty of natural bins see no behavior change."""
        ds = self._make_full_physics_ds()

        result = fit_range_decay(ds, bin_width_m=5.0, usable_range_m=60.0)

        assert result.bin_width_m == 5.0
        assert result.bin_width_m_requested == 5.0
        assert not any("bin_width_m auto-reduced" in n for n in result.notes)

    def test_min_bins_zero_disables_adaptation(self):
        ds = self._make_full_physics_ds()

        result = fit_range_decay(ds, bin_width_m=5.0, usable_range_m=15.0, min_bins=0)

        assert result.bin_width_m == 5.0
        assert result.bin_width_m_requested == 5.0

    def test_no_usable_range_m_leaves_bin_width_unchanged(self):
        """usable_range_m=None (Stage 3 couldn't determine one) has
        nothing to adapt against -- must not crash, must leave the
        requested bin_width_m as-is."""
        ds = self._make_full_physics_ds()

        result = fit_range_decay(ds, bin_width_m=5.0, usable_range_m=None)

        assert result.bin_width_m == 5.0
        assert result.bin_width_m_requested == 5.0

    def test_classification_ranked_by_fixed_scale_rmse_not_free_fit_rmse(self, monkeypatch):
        """Isolates fit_range_decay's own ranking logic from
        fit_tvg_curve's real fitting behavior entirely --
        injects a hand-built TVGCandidateFitResult where free-fit
        rmse_db and fixed_scale_rmse_db disagree about the winner by
        construction, and confirms fit_range_decay's classification
        follows fixed_scale_rmse_db (2026-08-24 change)."""
        import sidescanpy.analysis.tvg as tvg_mod
        from sidescanpy.analysis.tvg import TVGCandidateFit, TVGCandidateFitResult

        ds = _make_raw_dataset(n_pings=10, n_samples=100)
        ds = compute_geometric_ranges(ds)

        # "absorption_only" wins on free-fit rmse_db (0.1, lowest) but
        # needs an implausible scale=5.0 to get there; "full_physics"
        # wins on fixed_scale_rmse_db (0.2, lowest) with a free-fit
        # scale of 1.02 -- close to the literal, uncalibrated model.
        candidates = [
            TVGCandidateFit(name="flat", params=TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0),
                             scale=0.0, rmse_db=10.0, fixed_scale_rmse_db=10.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="spreading_only", params=TVGParams(absorption_db_per_m=0.0),
                             scale=1.5, rmse_db=3.0, fixed_scale_rmse_db=5.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="absorption_only", params=TVGParams(spreading_loss_db=0.0),
                             scale=5.0, rmse_db=0.1, fixed_scale_rmse_db=8.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="full_physics", params=TVGParams(),
                             scale=1.02, rmse_db=1.0, fixed_scale_rmse_db=0.2,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
        ]
        fake_result = TVGCandidateFitResult(
            candidates=candidates,
            best=min(candidates, key=lambda c: c.fixed_scale_rmse_db),
            observed_range_m=np.array([5.0, 40.0]),
            observed_loss_db=np.array([0.0, 10.0]),
        )
        monkeypatch.setattr(tvg_mod, "fit_tvg_curve", lambda *a, **kw: fake_result)

        result = tvg_mod.fit_range_decay(ds, bin_width_m=2.0)

        assert result.best_candidate == "full_physics"
        assert result.best_rmse_db == pytest.approx(0.2)
        assert result.best_free_fit_scale == pytest.approx(1.02)
        assert result.best_free_fit_rmse_db == pytest.approx(1.0)
        assert result.candidates["absorption_only"]["fixed_scale_rmse_db"] == pytest.approx(8.0)
        # runner-up by fixed_scale_rmse_db is spreading_only (5.0), not
        # absorption_only (8.0) -- confirms the whole ranking, not just best
        assert result.tvg_status == "none"  # _TVG_CANDIDATE_TO_STATUS["full_physics"]
        assert "spreading_only" in result.notes[0]

    def test_classification_invariant_to_db_reference(self):
        """The self-referenced dB conversion's reference percentile is
        a constant offset in dB space, which cancels out of every
        loss/residual computation `fit_tvg_curve` does — so
        changing it must not change any classification output, only
        `metrics["db_reference"]["value"]` itself."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        low_ref = fit_range_decay(ds, bin_width_m=2.0, db_reference_percentile=50.0)
        high_ref = fit_range_decay(ds, bin_width_m=2.0, db_reference_percentile=99.9)

        assert low_ref.db_reference["value"] != high_ref.db_reference["value"]
        assert low_ref.tvg_status == high_ref.tvg_status
        assert low_ref.best_candidate == high_ref.best_candidate
        # float32 backscatter + the +eps offset means the two reference
        # choices aren't bit-identical, just equal well past float32
        # precision -- rtol chosen loose enough to absorb that, tight
        # enough that a real reference-dependence bug would still fail it.
        assert np.isclose(low_ref.best_scale, high_ref.best_scale, rtol=1e-4)
        assert np.isclose(low_ref.best_rmse_db, high_ref.best_rmse_db, rtol=1e-4, atol=1e-4)

    def test_flat_curve_yields_negligible_residual(self):
        """Raw data with no range-dependent decay at all (hardware
        already fully compensated): every candidate's fitted scale
        collapses toward 0 with tiny RMSE -- with only noise left to
        fit, `spreading_only`/`absorption_only`/`full_physics` can
        occasionally out-fit `flat` by chance (their free scale
        overfits the noise itself), which is exactly the
        near-flat-data fit-degeneracy `fit_tvg_curve` already
        documents (see that function's `r_ref_m` docstring). So this
        checks "no real decay detected" (uniformly tiny RMSE, status
        reads as no-correction-needed-or-ambiguous), not a specific
        winning candidate name."""
        n_pings, n_samples = 30, 300
        bs = np.full((n_pings, 2, n_samples), 5.0, dtype=np.float32)
        rng = np.random.default_rng(1)
        bs += rng.normal(0, 0.05, size=bs.shape).astype(np.float32)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = fit_range_decay(ds, bin_width_m=2.0)

        assert result.tvg_status in ("full", "ambiguous")
        assert result.best_rmse_db < 0.1
        assert all(c["rmse_db"] < 0.1 for c in result.candidates.values())

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="slant_range_m"):
            fit_range_decay(ds)

    def test_exclude_first_n_bins_zero_is_exact_prior_behavior(self):
        ds = self._make_full_physics_ds()

        default = fit_range_decay(ds, bin_width_m=2.0)
        explicit_zero = fit_range_decay(ds, bin_width_m=2.0, exclude_first_n_bins=0)

        assert default.to_dict() == explicit_zero.to_dict()
        assert default.tvg_status == explicit_zero.tvg_status
        assert default.exclude_first_n_bins == 0
        assert default.n_bins_excluded_after_bottom == 0

    def test_exclude_first_n_bins_moves_r_ref_away_from_a_bright_artifact_at_the_bottom_edge(self):
        """Real-world motivation: a bright leading-edge specular
        artifact right at the first-bottom-return boundary can itself
        win r_ref_m's near-field max-mean auto-selection, corrupting
        the reference (and every candidate's fit against it) --
        distinct from the ring-down artifact at the very START of a
        ping, which reads anomalously LOW and is already handled by
        picking the max instead of the first bin (see
        fit_tvg_curve's own r_ref_m docstring).
        exclude_first_n_bins drops the artifact bin(s) before r_ref_m
        ever sees them."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)
        # altitude=10m, sample_rate=1000Hz, sound_vel=1500m/s -> 0.75m/
        # sample -> water column ends ~sample 14 (slant_range > altitude,
        # ground_range ~3.2m there). Sample 14 alone lands cleanly in
        # the first 2m-wide bin -- inject a bright artifact there.
        bs = ds["backscatter_intensity"].values.copy()
        bs[:, :, 14:15] *= 80.0
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        without_exclusion = fit_range_decay(ds, bin_width_m=2.0)
        with_exclusion = fit_range_decay(ds, bin_width_m=2.0, exclude_first_n_bins=1)

        assert without_exclusion.n_bins_excluded_after_bottom == 0
        assert with_exclusion.n_bins_excluded_after_bottom == 1
        assert any("Dropped 1 bin(s)" in n for n in with_exclusion.notes)
        # the artifact should have corrupted r_ref_m's auto-selection
        # without exclusion, and dropping it should move the pick
        assert with_exclusion.r_ref_m != without_exclusion.r_ref_m
        # underlying curve is still genuinely full_physics -- dropping
        # the artifact should recover a real, better fit against it
        assert with_exclusion.best_candidate == "full_physics"
        assert with_exclusion.best_rmse_db < without_exclusion.best_rmse_db

    def test_r_fit_min_m_passthrough_fits_both_sides_of_reference(self):
        """Combined real use case (2026-09-02): r_ref_m pinned mid-swath
        + r_fit_min_m=0.0 to fit both sides + exclude_first_n_bins to
        drop only the near-boundary artifact bins, not the whole near
        half of the swath."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        one_sided = fit_range_decay(ds, bin_width_m=2.0, r_ref_m=20.0)
        both_sides = fit_range_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m=0.0)

        assert both_sides.r_ref_m == 20.0  # unchanged
        assert both_sides.r_fit_min_m == 0.0
        assert one_sided.r_fit_min_m == 20.0  # defaults to r_ref_m
        assert both_sides.n_bins_used > one_sided.n_bins_used

    def test_r_fit_min_m_auto_resolves_to_numeric_value_and_records_which_mode_ran(self):
        """r_fit_min_m="auto": a fixed near-edge number picked by
        eye on one mission didn't generalize across a dataset whose
        near-field ramp varies mission to mission. metrics["r_fit_min_m"]
        must carry the *resolved* number (what the fit actually used);
        metrics["r_fit_min_m_override"] must keep the literal "auto" so
        the JSON export still shows which mode produced the run."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m="auto")

        assert isinstance(result.r_fit_min_m, float)
        assert result.r_fit_min_m_override == "auto"
        assert result.r_fit_min_auto_margin_m == pytest.approx(5.0)
        assert any("r_fit_min_m='auto' resolved to" in n for n in result.notes)

    def test_r_fit_min_m_auto_without_r_ref_m_or_usable_range_m_falls_back_gracefully(self):
        """Graceful fallback instead of raising --
        "auto" is a default (not just an opt-in
        override), so a single difficult mission/section with no
        determinable usable_range_m must not take down the whole
        analyze run. Falls back to the prior one-sided fit window
        (r_fit_min_m=None, starts at r_ref_m) instead."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, r_fit_min_m="auto")

        assert result.r_fit_min_m == result.r_ref_m  # one-sided fallback
        assert result.r_fit_min_m_override == "auto"
        assert any("falling back to the prior one-sided fit window" in n for n in result.notes)

    def test_r_fit_min_m_auto_uses_r_ref_search_fraction_of_usable_range_when_r_ref_m_none(self):
        """r_ref_m=None + usable_range_m given: auto mode's search bound
        falls back to r_ref_search_fraction * usable_range_m rather than
        requiring r_ref_m to be pinned."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_fit_min_m="auto")

        assert isinstance(result.r_fit_min_m, float)

    def test_r_fit_min_m_auto_clamps_to_r_ref_m_when_naive_value_would_exceed_it(self):
        """Regression: the search
        restricts by *bin center* <= r_ref_m, but the returned value is
        that bin's far edge PLUS the margin -- on data whose true
        near-field peak sits close to r_ref_m itself (seen on real
        data, whose near-field ramp varies enough that it isn't reliably
        confined to the near half of the swath), the naive auto value
        can still land past r_ref_m even though the search never looked
        beyond it. Must clamp down to r_ref_m, not silently return a
        window that doesn't contain its own reference."""
        # Hand-built ramp (distinct from _make_full_physics_ds's
        # monotonic decay, whose peak always sits at range=0): peak at
        # range~19m, deliberately close to r_ref_m=20.0m below, so
        # peak_far_edge(~19.5m) + the 5.0m default margin (~24.5m)
        # would exceed r_ref_m without the clamp.
        range_m = np.arange(60.0) + 0.5
        amplitude = np.where(range_m < 19.0, range_m, np.maximum(38.0 - range_m, 0.1)).astype(np.float32)
        ds = xr.Dataset({
            "backscatter_intensity": ("sample", amplitude),
            "slant_range_m": ("sample", range_m),
        })

        result = fit_range_decay(
            ds, bin_width_m=1.0, freq_hz=1_495_000.0, usable_range_m=40.0,
            r_ref_m=20.0, r_fit_min_m="auto", min_total_count=1,
        )

        assert result.r_fit_min_m == pytest.approx(20.0)
        assert any("clamped down to r_ref_m" in n for n in result.notes)

    def test_r_ref_m_auto_avoids_the_clamp_the_prior_test_needed(self):
        """2026-09-02, direct response to a real report: pinning
        r_ref_m independently (numeric, or the legacy "mid_swath") and
        clamping r_fit_min_m to fit around it lands the reference
        EXACTLY on the window's boundary when they collide (the prior
        test) -- often a noisy transition bin, since that's the region
        the window exists to exclude. r_ref_m="auto" fixes this at the
        root: resolve the window first, place the reference INSIDE it.
        Same pathological ramp as the clamp test above (peak close to
        where a pinned reference would land) -- this time the reference
        must NOT sit on either boundary."""
        range_m = np.arange(60.0) + 0.5
        amplitude = np.where(range_m < 19.0, range_m, np.maximum(38.0 - range_m, 0.1)).astype(np.float32)
        ds = xr.Dataset({
            "backscatter_intensity": ("sample", amplitude),
            "slant_range_m": ("sample", range_m),
        })

        result = fit_range_decay(
            ds, bin_width_m=1.0, freq_hz=1_495_000.0, usable_range_m=40.0,
            r_ref_m="auto", r_fit_min_m="auto", min_total_count=1,
        )

        assert result.r_fit_min_m < result.r_ref_m < result.r_fit_max_m  # strictly inside, not on a boundary
        assert not any("clamped down to r_ref_m" in n for n in result.notes)
        assert any("r_ref_m='auto' resolved to" in n for n in result.notes)

    def test_r_ref_m_auto_uses_window_fraction(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(
            ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="auto", r_fit_min_m=10.0,
            r_ref_window_fraction=0.25,
        )

        assert result.r_ref_m == pytest.approx(10.0 + 0.25 * (60.0 - 10.0))

    def test_r_ref_m_auto_without_usable_range_m_falls_back_gracefully(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, r_ref_m="auto")

        assert result.tvg_status is not None  # completed, did not raise
        assert result.r_ref_m_override == "auto"
        assert any("falling back to the near-field auto-pick" in n for n in result.notes)

    def test_r_fit_min_m_invalid_string_raises(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        with pytest.raises(ValueError, match="must be 'auto'"):
            fit_range_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m="near")

    def test_r_ref_m_mid_swath_resolves_to_half_usable_range(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="mid_swath")

        assert result.r_ref_m == pytest.approx(30.0)
        assert result.r_ref_m_override == "mid_swath"
        assert any("r_ref_m='mid_swath' resolved to" in n for n in result.notes)

    def test_r_ref_m_mid_swath_without_usable_range_m_falls_back_gracefully(self):
        """Graceful fallback instead of raising --
        "mid_swath" is a default (not just an opt-in
        override), so a single difficult mission/section with no
        determinable usable_range_m must not take down the whole
        analyze run. Falls back to the prior near-field auto-pick
        (r_ref_m=None) instead."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = fit_range_decay(ds, bin_width_m=2.0, r_ref_m="mid_swath")

        assert result.tvg_status is not None  # completed, did not raise
        assert result.r_ref_m_override == "mid_swath"
        assert any("falling back to the near-field auto-pick" in n for n in result.notes)

    def test_r_ref_m_invalid_string_raises(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        with pytest.raises(ValueError, match="must be 'auto' or 'mid_swath'"):
            fit_range_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="far_field")

    def test_mid_swath_paired_with_auto_keeps_reference_inside_fit_window(self):
        """The exact failure mode 'mid_swath' exists to prevent: left to
        auto-pick independently, r_ref_m (near-field peak bin itself)
        and r_fit_min_m="auto" (that same peak's far edge + margin)
        nearly always disagree about which side of the peak r_ref_m
        should be on -- r_ref_m ends up nearer than the fit window's own
        near edge, i.e. outside it. Pinning r_ref_m="mid_swath" must
        keep the resolved reference inside [r_fit_min_m, r_fit_max_m]."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        independent = fit_range_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_fit_min_m="auto")
        paired = fit_range_decay(
            ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="mid_swath", r_fit_min_m="auto",
        )

        assert independent.r_ref_m < independent.r_fit_min_m  # the bug
        assert paired.r_fit_min_m <= paired.r_ref_m <= paired.r_fit_max_m

    def _make_two_channel_decay_ds(self, n_pings=60, n_samples=300):
        """channel 0 (port) follows the full uncompensated physics decay
        curve; channel 1 (starboard) is flat (already fully
        compensated, scale=1.0 everywhere) -- classifies as "none" vs.
        "full" respectively when fit independently. Pooled together,
        neither classification is expected to hold cleanly."""
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)
        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        ref = np.nanmin(loss_db[np.isfinite(loss_db)])
        port_amplitude = 10.0 ** (-(loss_db - ref) / 20.0)
        port_amplitude = np.nan_to_num(port_amplitude, nan=0.0).astype(np.float32)
        starboard_amplitude = np.where(np.isfinite(ground_range), 1.0, 0.0).astype(np.float32)

        bs = ds["backscatter_intensity"].values.copy()
        bs[:, 0, :] = port_amplitude[:, 0, :]
        bs[:, 1, :] = starboard_amplitude[:, 1, :]
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)
        return ds

    def test_channel_none_metrics_report_no_channel_restriction(self):
        ds = self._make_two_channel_decay_ds()

        pooled = fit_range_decay(ds, bin_width_m=2.0)

        assert pooled.channel is None

    def test_channel_restricts_fit_to_one_side(self):
        ds = self._make_two_channel_decay_ds()

        port = fit_range_decay(ds, bin_width_m=2.0, channel=0)
        starboard = fit_range_decay(ds, bin_width_m=2.0, channel=1)

        assert port.channel == 0
        assert starboard.channel == 1
        assert port.tvg_status == "none"  # nothing compensated yet
        assert starboard.tvg_status == "full"  # already flat

    def test_channel_string_matches_int_position(self):
        ds = self._make_two_channel_decay_ds()

        by_string = fit_range_decay(ds, bin_width_m=2.0, channel="starboard")
        by_int = fit_range_decay(ds, bin_width_m=2.0, channel=1)

        assert by_string.tvg_status == by_int.tvg_status
        assert by_string.channel == 1

    def test_channel_invalid_string_raises(self):
        ds = self._make_two_channel_decay_ds()

        with pytest.raises(ValueError, match="not found in channel_id"):
            fit_range_decay(ds, bin_width_m=2.0, channel="midship")

    def test_channel_pooled_differs_from_either_single_channel(self):
        ds = self._make_two_channel_decay_ds()

        pooled = fit_range_decay(ds, bin_width_m=2.0)
        port = fit_range_decay(ds, bin_width_m=2.0, channel=0)
        starboard = fit_range_decay(ds, bin_width_m=2.0, channel=1)

        assert pooled.best_rmse_db != pytest.approx(port.best_rmse_db)
        assert pooled.best_rmse_db != pytest.approx(starboard.best_rmse_db)

    def test_exclude_first_n_bins_exceeding_available_bins_handled_gracefully(self):
        ds = self._make_full_physics_ds(n_pings=10, n_samples=60)

        result = fit_range_decay(ds, bin_width_m=2.0, exclude_first_n_bins=10_000)

        assert result.tvg_status == "ambiguous"
        assert "Not enough binned samples" in result.notes[0]
        assert result.exclude_first_n_bins == 10_000
        assert result.n_bins_excluded_after_bottom >= 0

    def test_reject_outliers_recovers_clean_fit_from_target_corrupted_curve(self):
        """Same full_physics curve as test_full_physics_curve_classified_as_none,
        but with a handful of huge point-target-like spikes injected --
        reject_outliers=False should be measurably degraded by them,
        reject_outliers=True should recover close to the clean fit."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)

        rng = np.random.default_rng(2)
        n_spikes = 40
        flat_idx = rng.choice(amplitude.size, size=n_spikes, replace=False)
        amplitude.flat[flat_idx] *= 50.0  # simulated hard-target/complex-terrain returns

        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        # range_var pinned to ground_range_m: the injected curve above is
        # built as an exact function of ground_range specifically, so the
        # fit must be evaluated against that same variable regardless of
        # fit_range_decay's own (unrelated) 2026-09-02 default change to
        # slant_range_m.
        with_spikes = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)
        cleaned = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0, reject_outliers=True)

        assert cleaned.outlier_n_rejected > 0
        assert cleaned.best_rmse_db < with_spikes.best_rmse_db
        assert cleaned.tvg_status == "none"

    def test_fit_value_col_median_wired_through_and_recorded(self):
        """Contract test only -- real-data checking found `"median"`
        does NOT reliably improve RMSE over `"mean"` under
        reject_outliers=True (see fit_range_decay's fit_value_col
        docstring), so this doesn't assert an RMSE direction, just that
        the parameter actually reaches fit_tvg_curve and gets
        recorded, and that classification stays correct either way."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        result = fit_range_decay(ds, bin_width_m=2.0, reject_outliers=True, fit_value_col="median")

        assert result.fit_value_col == "median"
        assert result.tvg_status == "none"
        assert result.best_candidate == "full_physics"

    def test_r_ref_m_override_is_used_instead_of_auto_selection(self):
        """Passing r_ref_m should reach fit_tvg_curve and pin
        the reported reference range exactly, not just influence it --
        on a clean full_physics curve the classification should stay
        correct at a manually-chosen mid-field reference too."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        auto = fit_range_decay(ds, bin_width_m=2.0)
        explicit_r_ref_m = auto.r_ref_m + 5.0
        overridden = fit_range_decay(ds, bin_width_m=2.0, r_ref_m=explicit_r_ref_m)

        assert overridden.r_ref_m == pytest.approx(explicit_r_ref_m)
        assert overridden.r_ref_m_override == pytest.approx(explicit_r_ref_m)
        assert overridden.r_ref_m != auto.r_ref_m
        assert overridden.best_candidate == "full_physics"

    def test_r_ref_m_override_defaults_to_none_when_auto_selecting(self):
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        result = fit_range_decay(ds, bin_width_m=2.0)

        assert result.r_ref_m_override is None

    def test_r_ref_search_fraction_wired_through_to_fit(self):
        """Contract test: r_ref_search_fraction reaches
        fit_tvg_curve's own identically-named parameter (whose
        behavior is tested directly in test_tvg.py) --
        widening it away from the 0.5 default should be able to change
        the auto-selected r_ref_m on a curve with more than one local
        peak, without raising or otherwise misbehaving."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=30.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        narrow = fit_range_decay(ds, bin_width_m=2.0, r_ref_search_fraction=0.1)
        wide = fit_range_decay(ds, bin_width_m=2.0, r_ref_search_fraction=1.0)

        assert narrow.best_candidate == "full_physics"
        assert wide.best_candidate == "full_physics"


class TestAutoNearEdgeM:
    """Direct tests of `_auto_near_edge_m`, the helper `fit_range_decay`'s
    `r_fit_min_m="auto"` delegates to -- uses `backscatter_vs_range`'s
    own documented contract ("any dataset with value_var and range_var,
    same shape, arbitrary dims") to build a minimal synthetic ramp
    directly, rather than routing through the full raw-dataset/geometric
    pipeline the rest of this file uses."""

    def _make_ramp_ds(self) -> xr.Dataset:
        # 1m-wide "pings" from 0 to 39m, amplitude rises to a clear peak
        # at range 10-11m then falls -- a deliberately non-monotonic
        # near-field ramp, distinct from the pure-decay curves the rest
        # of this file's fixtures use.
        range_m = np.arange(40.0) + 0.5
        amplitude = np.where(range_m < 11.0, range_m, np.maximum(22.0 - range_m, 0.1))
        return xr.Dataset(
            {"value": ("sample", amplitude), "range_m": ("sample", range_m)}
        )

    def test_finds_peak_bin_far_edge_plus_margin(self):
        ds = self._make_ramp_ds()

        result = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=40.0, margin_m=5.0,
        )

        # Peak bin covers [10, 11) -> far edge 11.0, + 5.0m margin.
        assert result == pytest.approx(16.0)

    def test_search_max_m_restricts_the_search_window(self):
        """A search bound tighter than where the true peak sits should
        still resolve, just against whatever's the best bin inside the
        window -- not silently ignore the bound."""
        ds = self._make_ramp_ds()

        restricted = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=5.0, margin_m=5.0,
        )

        # Within [0, 5], amplitude is still monotonically rising (peak
        # is at 10-11m, outside this window) -- best bin is the last one.
        assert restricted == pytest.approx(5.0 + 5.0)

    def test_no_bins_in_search_window_returns_none(self):
        ds = self._make_ramp_ds()

        result = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=-1.0, margin_m=5.0,
        )

        assert result is None


class TestWaterColumnExclusionIndependentOfRangeVar:
    """2026-09-02: estimate_usable_range/fit_range_decay's range_var default
    changed from ground_range_m to slant_range_m (the physics-correct
    choice for a TVG loss-vs-range fit -- matches what apply_tvg() itself
    uses). But water-column exclusion must keep working regardless --
    slant_range_m is pure time-of-flight with no NaNs of its own, so it
    can't detect the water column the way ground_range_m's own NaN
    pattern can. These confirm the decoupling fix actually holds (water
    column exclusion stays keyed to ground_range_m internally), not just
    that the range_var default flipped."""

    def test_stage3_water_column_count_matches_regardless_of_range_var(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        via_slant = estimate_usable_range(ds, range_var="slant_range_m")
        via_ground = estimate_usable_range(ds, range_var="ground_range_m")

        assert via_slant.water_column_n_excluded > 0
        assert via_slant.water_column_n_excluded == via_ground.water_column_n_excluded
        assert via_slant.range_var == "slant_range_m"

    def test_stage4_water_column_count_matches_regardless_of_range_var(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        via_slant = fit_range_decay(ds, range_var="slant_range_m", bin_width_m=2.0)
        via_ground = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)

        assert via_slant.water_column_n_excluded > 0
        assert via_slant.water_column_n_excluded == via_ground.water_column_n_excluded
        assert via_slant.range_var == "slant_range_m"

    def test_stage3_garbage_in_water_column_does_not_skew_noise_floor_with_slant_range_default(self):
        """Direct behavioral proof, not just a count match: a huge
        garbage value placed only in the water column must not appear in
        the binned curve when range_var defaults to slant_range_m --
        would have leaked straight in before this fix, biasing the
        noise-floor estimate."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)
        water_column_mask = ~np.isfinite(ds["ground_range_m"].values)
        assert water_column_mask.any(), "fixture must have a real water column for this test to mean anything"

        bs = ds["backscatter_intensity"].values.copy()
        bs[water_column_mask] = 1e6
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        result = estimate_usable_range(ds, bin_width_m=2.0)  # default range_var=slant_range_m
        binned = result.binned
        assert not binned.empty
        assert binned["mean"].max() < 200.0, (
            f"garbage water-column value leaked into the binned curve: max bin mean = {binned['mean'].max()}"
        )

    def test_stage4_garbage_in_water_column_does_not_appear_in_fit_curve_with_slant_range_default(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)
        water_column_mask = ~np.isfinite(ds["ground_range_m"].values)
        assert water_column_mask.any()

        bs = ds["backscatter_intensity"].values.copy()
        bs[water_column_mask] = 1e6
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        result = fit_range_decay(ds, bin_width_m=2.0)  # default range_var=slant_range_m
        binned = result.binned
        assert not binned.empty
        assert binned["mean"].max() < 200.0, (
            f"garbage water-column value leaked into the fit curve: max bin mean = {binned['mean'].max()}"
        )
